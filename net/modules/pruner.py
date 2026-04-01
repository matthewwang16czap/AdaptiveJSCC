import torch
import torch.nn as nn
from timm.layers import trunc_normal_


def topk_mask(score, keep_ratio):
    """
    Generates a binary mask keeping the top-k scoring tokens.
    """
    B, N = score.shape
    k = max(1, int(N * keep_ratio))
    # Find indices of top-k scores
    _, idx = torch.topk(score, k, dim=1)
    # Create binary mask
    mask = torch.zeros_like(score)
    mask.scatter_(1, idx, 1.0)
    return mask


class CNNImportance(nn.Module):
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        # Stable Initialization:
        # Start with small weights so sigmoid starts in the linear region (near 0.5)
        nn.init.constant_(self.net[-1].weight, 0)
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, x):
        return self.net(x)


class SwinTokenPruner(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.importance_net = CNNImportance(dim, hidden_dim)
        # The mask token is a learnable feature that replaces "pruned" tokens.
        # This prevents absolute zeros from breaking LayerNorm variance calculations.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x, H, W, keep_ratio):
        """
        Args:
            x: Input features (B, N, C) where N = H * W
            H, W: Spatial resolution
            keep_ratio: Fraction of tokens to keep (0.0 to 1.0)
        """
        B, N, C = x.shape
        # 1. Generate Importance Scores (2D CNN context)
        # Reshape: (B, N, C) -> (B, C, H, W)
        feat_2d = x.transpose(1, 2).reshape(B, C, H, W)
        score_2d = self.importance_net(feat_2d)
        score = score_2d.view(B, N)
        # 2. Generate Mask with Straight-Through Estimator (STE)
        # During forward, we use hard_mask (0 or 1).
        # During backward, we use the gradient of soft_mask (sigmoid).
        with torch.no_grad():
            hard_mask = topk_mask(score, keep_ratio)  # (B, N)
        soft_mask = torch.sigmoid(score)  # (B, N)
        # The STE Magic:
        # mask = hard_mask in forward pass
        # grad(mask) = grad(soft_mask) in backward pass
        mask = hard_mask + (soft_mask - soft_mask.detach())
        mask = mask.unsqueeze(-1)  # (B, N, 1)
        # 3. Replace instead of Zero-out
        # Values where mask is 1 stay as x.
        # Values where mask is 0 become self.mask_token.
        x_pruned = x * mask + self.mask_token * (1.0 - mask)
        return x_pruned, mask


class SwinChannelPruner(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.dim = dim
        # Squeeze-and-Excitation style importance network
        self.importance_net = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
        )
        # Initialize last layer to 0 so all channels start with equal "soft" probability
        nn.init.constant_(self.importance_net[-1].weight, 0)
        nn.init.constant_(self.importance_net[-1].bias, 0)

    def forward(self, x, keep_ratio):
        """
        x: (B, N, C)
        keep_ratio: float (e.g., 0.5 to keep half the channels)
        """
        B, N, C = x.shape
        # 1. Global Average Pooling to get channel-wise statistics
        # (B, N, C) -> (B, C)
        channel_stats = x.mean(dim=1)
        # 2. Generate Channel Importance Scores
        # (B, C) -> (B, C)
        scores = self.importance_net(channel_stats)
        # 3. Generate Channel Mask (STE)
        with torch.no_grad():
            # topk_mask operates on the C dimension here
            k = max(1, int(C * keep_ratio))
            _, idx = torch.topk(scores, k, dim=1)
            hard_mask = torch.zeros_like(scores)
            hard_mask.scatter_(1, idx, 1.0)
        soft_mask = torch.sigmoid(scores)
        # STE: Gradient flows through sigmoid, values come from hard_mask
        mask = hard_mask + (soft_mask - soft_mask.detach())
        # 4. Apply Mask: (B, C) -> (B, 1, C) for broadcasting
        mask = mask.unsqueeze(1).expand(-1, x.shape[1], -1)  # (B, N, C)
        x = x * mask
        return x, mask


class SwinChannelAdapter(nn.Module):
    """
    Reverse net for SwinChannelPruner.
    Injects a learnable restoration feature into channels that were pruned.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Learnable vector to represent the 'average' missing signal per channel
        self.restoration_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.restoration_token, std=0.02)
        # Refinement layer to blend restored and original features
        self.refine = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))

    def forward(self, x_pruned):
        """
        x_pruned: Features from encoder (B, N, C)
        mask: Mask used in encoder (B, 1, C) or (B, N, C)
        """
        mask = (x_pruned != 0).float()  # 1 where kept, 0 where pruned
        # Inverse mask: 1 where we pruned, 0 where we kept
        inv_mask = 1.0 - mask
        # Fill pruned slots with the restoration token
        # x_pruned already has 'keep' channels. We add 'restoration' to 'pruned' channels.
        x_restored = x_pruned + (self.restoration_token * inv_mask)
        return self.refine(x_restored)


class SwinTokenWiseChannelPruner(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.dim = dim
        # This net now processes (B, N, C) -> (B, N, C)
        # It's essentially a 1x1 conv layer logic
        self.importance_net = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
        )
        nn.init.constant_(self.importance_net[-1].weight, 0)
        nn.init.constant_(self.importance_net[-1].bias, 0)

    def forward(self, x, keep_ratio):
        B, N, C = x.shape
        # 1. Generate importance scores for EVERY token individually
        # scores shape: (B, N, C)
        scores = self.importance_net(x)
        # 2. Per-token Top-K Selection
        with torch.no_grad():
            k = max(1, int(C * keep_ratio))
            # Find the best channels for EACH token independently
            # We use dim=2 to sort along the Channel dimension
            _, idx = torch.topk(scores, k, dim=2)
            # Create a unique mask for every token: (B, N, C)
            hard_mask = torch.zeros_like(scores)
            hard_mask.scatter_(2, idx, 1.0)
        soft_mask = torch.sigmoid(scores)
        # STE for gradient flow
        mask = hard_mask + (soft_mask - soft_mask.detach())
        # 3. Apply the token-specific mask
        # (B, N, C) * (B, N, C)
        x = x * mask
        return x, mask


class SwinTokenWiseChannelAdapter(nn.Module):
    """
    Reverse net for SwinTokenWiseChannelPruner.
    Handles spatially varying channel pruning.
    """

    def __init__(self, dim, reduction=4):
        super().__init__()
        self.dim = dim
        self.restoration_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.restoration_token, std=0.02)
        # Dynamic gate: uses the mask itself to modulate restoration intensity
        self.adaptation_gate = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )

    def forward(self, x_pruned):
        """
        x_pruned: (B, N, C)
        mask: (B, N, C) - The token-wise mask from encoder
        """
        mask = (x_pruned != 0).float()  # 1 where kept, 0 where pruned
        inv_mask = 1.0 - mask
        # Step 1: Broad restoration
        res_feat = self.restoration_token * inv_mask
        # Step 2: Combine
        x_combined = x_pruned + res_feat
        # Step 3: Dynamic Adaptation
        # We let the model 're-weight' the restored features based on local context
        gate = self.adaptation_gate(x_combined)
        return x_combined * gate


class DummyStage(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        return self.block(x)


class SwinWithCNNPruning(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = DummyStage(96, 96)
        self.stage2 = DummyStage(96, 192)
        self.stage3 = DummyStage(192, 384)
        self.stage4 = DummyStage(384, 768)
        # No keep_ratio stored here anymore
        self.prune1 = SwinChannelPruner(96)
        self.prune2 = SwinChannelPruner(192)
        self.prune3 = SwinChannelPruner(384)
        self.prune4 = SwinChannelPruner(768)

    def forward(self, x, H, W, ratios=[0.9, 0.7, 0.5, 0.3]):
        """
        ratios: List of floats for each pruning stage
        """
        x = self.stage1(x)
        x, _ = self.prune1(x, ratios[0])
        x = self.stage2(x)
        x, _ = self.prune2(x, ratios[1])
        x = self.stage3(x)
        x, _ = self.prune3(x, ratios[2])
        x = self.stage4(x)
        x, _ = self.prune4(x, ratios[3])
        return x


if __name__ == "__main__":
    B, H, W, C = 2, 16, 16, 96
    N = H * W
    x = torch.randn(B, N, C)
    model = SwinWithCNNPruning()
    # You can now pass different ratios for every forward pass
    custom_ratios = [0.8, 0.6, 0.4, 0.2]
    out = model(x, H, W, ratios=custom_ratios)
    print(f"Output shape: {out.shape}")
