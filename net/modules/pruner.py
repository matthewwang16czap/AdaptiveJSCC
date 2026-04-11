import torch
import torch.nn as nn


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


# class CNNImportance(nn.Module):
#     def __init__(self, in_channels, hidden_dim=64):
#         super().__init__()
#         self.local_feat = nn.Sequential(
#             nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
#         )
#         # Global context branch
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#         self.global_fc = nn.Linear(in_channels, hidden_dim)
#         self.final_conv = nn.Conv2d(hidden_dim, 1, 1)

#     def forward(self, x):
#         local_x = self.local_feat(x)  # (B, hidden, H, W)
#         global_x = self.global_fc(self.global_pool(x).flatten(1))  # (B, hidden)

#         # Combine local and global
#         out = local_x + global_x.view(x.shape[0], -1, 1, 1)
#         return self.final_conv(torch.relu(out))


# class EncoderTokenPruner(nn.Module):
#     def __init__(self, dim, hidden_ratio=0.5):
#         super().__init__()
#         # Using your CNN-based importance scorer
#         self.importance_net = CNNImportance(
#             in_channels=dim, hidden_dim=int(dim * hidden_ratio)
#         )

#     def forward(self, x, H, W, keep_ratio):
#         B, N, C = x.shape
#         # 1. Reshape to 2D for CNN importance scoring
#         # (B, H*W, C) -> (B, C, H, W)
#         feat_2d = x.transpose(1, 2).view(B, C, H, W)
#         # 2. Generate Importance Map (B, 1, H, W)
#         scores_2d = self.importance_net(feat_2d)
#         scores = scores_2d.view(B, -1)  # (B, N)
#         # 3. Generate Hard Mask (Non-differentiable part)
#         with torch.no_grad():
#             hard_mask = topk_mask(scores, keep_ratio)  # (B, N)
#         # 4. Straight-Through Estimator (STE) for gradients
#         soft_mask = torch.sigmoid(scores)
#         mask = hard_mask + (soft_mask - soft_mask.detach())
#         # 5. Apply Mask (B, N, 1)
#         x = x * mask.unsqueeze(-1).contiguous()
#         return x


class AttentionScorer(nn.Module):
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # A learned "Importance Prototype" that looks for semantic objects
        self.importance_query = nn.Parameter(torch.randn(1, 1, dim))
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        nn.init.trunc_normal_(self.importance_query, std=0.02)

    def forward(self, x):
        B, N, C = x.shape
        res = x
        x = self.norm(x)
        # Cross-attention: Learned Query vs. All Patches
        q = self.q_proj(self.importance_query).expand(B, -1, -1)  # (B, 1, C)
        k = self.k_proj(x)  # (B, N, C)
        v = self.v_proj(x)  # (B, N, C)
        scaling = C**-0.5
        attn = (q @ k.transpose(-2, -1)) * scaling
        attn_weights = attn.softmax(dim=-1)  # (B, 1, N)
        # Global context vector based on what the Query found "important"
        context = attn_weights @ v  # (B, 1, C)
        # Combine local token features with global context to decide importance
        # We add the global context to every patch feature before scoring
        scores = self.mlp(res + context)
        return scores.squeeze(-1)  # (B, N)


class EncoderTokenPruner(nn.Module):
    def __init__(self, dim, hidden_ratio=0.5):
        super().__init__()
        # Swapping CNN for Attention-based scoring
        self.importance_net = AttentionScorer(
            dim=dim, hidden_dim=int(dim * hidden_ratio)
        )

    def forward(self, x, H, W, keep_ratio):
        """
        x: (B, N, C)
        H, W: Height and Width (maintained for interface compatibility)
        keep_ratio: float between 0 and 1
        """
        B, N, C = x.shape
        # 1. Generate Importance Scores (B, N)
        # No more reshaping to 2D needed!
        scores = self.importance_net(x)
        # 2. Generate Hard Mask (Top-K)
        with torch.no_grad():
            hard_mask = topk_mask(scores, keep_ratio)  # (B, N)
        # 3. Straight-Through Estimator (STE)
        # Using a sigmoid to allow gradient flow back to the AttentionScorer
        soft_mask = torch.sigmoid(scores)
        mask = hard_mask + (soft_mask - soft_mask.detach())
        # 4. Apply Mask (B, N, 1)
        # Contiguous helps with memory layout for subsequent Transformer blocks
        x = x * mask.unsqueeze(-1).contiguous()
        return x


class EncoderChannelPruner(nn.Module):
    def __init__(self, dim, hidden_ratio=1):
        super().__init__()
        self.dim = dim
        # Squeeze-and-Excitation style importance network
        self.importance_net = nn.Sequential(
            nn.Linear(dim, dim * hidden_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(dim * hidden_ratio, dim),
        )
        # Initialize last layer to 0 so all channels start with equal "soft" probability
        nn.init.constant_(self.importance_net[-1].weight, 0)
        nn.init.constant_(self.importance_net[-1].bias, 0)

    def forward(self, x, keep_ratio):
        B, N, C = x.shape
        # 1. Global Average Pooling to get channel-wise statistics  (B, N, C) -> (B, C)
        channel_stats = x.mean(dim=1)
        # 2. Generate Channel Importance Scores (B, C) -> (B, C)
        scores = self.importance_net(channel_stats)
        # 3. Generate Channel Mask (STE)
        with torch.no_grad():
            hard_mask = topk_mask(scores, keep_ratio)  # (B, C)
        soft_mask = torch.sigmoid(scores)
        # STE: Gradient flows through sigmoid, values come from hard_mask
        mask = hard_mask + (soft_mask - soft_mask.detach())
        # 4. Apply Mask: (B, C) -> (B, 1, C) for broadcasting
        x = x * mask.unsqueeze(1).contiguous()
        return x


class EncoderIntermediatePrunerAdapter(nn.Module):
    def __init__(self, dim, hidden_ratio=0.5):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        hidden_dim = dim * hidden_ratio
        self.adapter = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim)
        )
        # Initialize as identity: weights to 0 so out = identity + 0
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, x):
        # x: (B, N, C)
        identity = x
        out = self.norm(x)
        out = self.adapter(out)
        # Returns dense features so the next frozen SwinBlock isn't 'confused' by zeros or unexpected distributions.
        return identity + out


class DecoderPrunerAdapter(nn.Module):
    def __init__(self, dim, hidden_ratio=1):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        # Learnable 'bias' for the pruned regions
        self.hole_filler = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.hole_filler, std=0.02)
        # Bottleneck to extract context from the non-zero features and generate modulation parameters (Scale and Shift)
        hidden_dim = dim * hidden_ratio
        self.context_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim * 2),  # Outputs both scale and shift
        )
        # Initialize the last layer to zero so the adapter starts as an identity
        nn.init.zeros_(self.context_net[-1].weight)
        nn.init.zeros_(self.context_net[-1].bias)

    def forward(self, x, H, W):
        """
        x: (B, N, C) - Sparse input features
        """
        # 1. Create binary mask (B, N, C)
        mask = (x != 0).float()
        inv_mask = 1.0 - mask
        # 2. Initial Infilling
        # We fill the 'holes' with a learnable token to prevent zero-values from killing the activation variance.
        x_filled = x + (self.hole_filler * inv_mask)
        # 3. Contextual Modulation
        # We apply LayerNorm to stabilize the sparse signal
        res = self.norm(x_filled)
        # Generate scale (gamma) and shift (beta) based on the filled features
        modulation = self.context_net(res)
        gamma, beta = modulation.chunk(2, dim=-1)
        # 4. Apply Modulation (Similar to AdaLN used in DiT/Latent Diffusion)
        # This helps the frozen decoder 'see' a distribution it recognizes.
        out = x_filled * (1 + gamma) + beta
        return out


class DecoderIntermediatePrunerAdapter(nn.Module):
    def __init__(self, dim, hidden_ratio=1.5):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        # Bottleneck structure to refine contextual relationships
        hidden_dim = int(dim * hidden_ratio)
        self.refine_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        # Initialize last layer to zero:
        # The adapter starts as an identity mapping (identity + 0)
        nn.init.zeros_(self.refine_net[-1].weight)
        nn.init.zeros_(self.refine_net[-1].bias)

    def forward(self, x):
        """
        x: (B, N, C) - Features processed by a previous decoder block
        """
        # Residual connection ensures we don't destroy the frozen weights' progress
        identity = x
        # 1. Normalize for stability
        out = self.norm(x)
        # 2. Refine features
        # This allows the model to learn small adjustments to the
        # reconstructed patches based on the surrounding context.
        out = self.refine_net(out)
        # 3. Add back to original signal
        return identity + out
