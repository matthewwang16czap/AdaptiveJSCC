import torch
import torch.nn as nn


# Top-K mask
def topk_mask(score, keep_ratio):
    B, N = score.shape
    k = max(1, int(N * keep_ratio))
    _, idx = torch.topk(score, k, dim=1)
    mask = torch.zeros_like(score)
    mask.scatter_(1, idx, 1.0)
    return mask


# CNN Importance Network
class CNNImportance(nn.Module):
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class SwinTokenPrunerCNN(nn.Module):
    def __init__(self, dim, keep_ratio=0.7, hidden_dim=64, soft_training=False):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.soft_training = soft_training
        self.importance_net = CNNImportance(dim, hidden_dim)

    def forward(self, x, H, W):
        """
        x: (B, N, C) where N = H * W
        Returns: (B, N, C) - Same shape, pruned tokens zeroed out.
        """
        B, N, C = x.shape
        # 1. Generate Importance Scores
        # Permute to (B, C, H, W) for CNN processing
        feat_2d = x.transpose(1, 2).reshape(B, C, H, W)
        score_2d = self.importance_net(feat_2d)
        score = score_2d.view(B, N)  # (B, N)
        # 2. Generate Binary Mask with Gradient Flow (STE)
        if self.training and self.soft_training:
            # Soft mask for pure gradient-based exploration
            mask = torch.sigmoid(score)
        else:
            # Hard Top-K selection
            with torch.no_grad():
                hard_mask = topk_mask(score, self.keep_ratio)
            # Straight-Through Estimator:
            # Forward pass uses hard_mask, backward pass uses sigmoid gradients
            soft_mask = torch.sigmoid(score)
            mask = hard_mask + (soft_mask - soft_mask.detach())
        # 3. Apply Mask (B, N, 1)
        mask = mask.unsqueeze(-1)
        # Dropout-like behavior: zero out pruned tokens
        # Optional: Scale by (1/keep_ratio) to keep activation magnitude consistent
        x = (x * mask) / self.keep_ratio
        return x


class DummyStage(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),  # Adjusts dimension for the next stage
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
        self.prune1 = SwinTokenPrunerCNN(96, keep_ratio=0.9)
        self.prune2 = SwinTokenPrunerCNN(192, keep_ratio=0.7)
        self.prune3 = SwinTokenPrunerCNN(384, keep_ratio=0.5)
        self.prune4 = SwinTokenPrunerCNN(768, keep_ratio=0.3)

    def forward(self, x, H, W):
        x = self.stage1(x)
        x = self.prune1(x, H, W)
        x = self.stage2(x)
        x = self.prune2(x, H, W)
        x = self.stage3(x)
        x = self.prune3(x, H, W)
        x = self.stage4(x)
        x = self.prune4(x, H, W)
        return x


if __name__ == "__main__":
    B, H, W, C = 2, 16, 16, 96
    N = H * W
    x = torch.randn(B, N, C)
    model = SwinWithCNNPruning()
    out = model(x, H, W)
    print(out.shape)  # MUST be (B, N, C)
