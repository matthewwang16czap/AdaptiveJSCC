import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Top-K mask
# -----------------------------
def topk_mask(score, keep_ratio):
    B, N = score.shape
    k = max(1, int(N * keep_ratio))

    _, idx = torch.topk(score, k, dim=1)

    mask = torch.zeros_like(score)
    mask.scatter_(1, idx, 1.0)

    return mask


# -----------------------------
# CNN Importance Network
# -----------------------------
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
        """
        x: (B, C, H, W)
        """
        return self.net(x)  # (B,1,H,W)


# -----------------------------
# Swin-Compatible Pruner (CNN-based)
# -----------------------------
class SwinTokenPrunerCNN(nn.Module):
    def __init__(self, dim, keep_ratio=0.7, hidden_dim=64, soft_training=True):
        super().__init__()

        self.keep_ratio = keep_ratio
        self.soft_training = soft_training

        self.importance_net = CNNImportance(dim, hidden_dim)

    def forward(self, x, H, W):
        """
        x: (B, N, C)
        H, W: spatial size
        """
        B, N, C = x.shape
        assert N == H * W, "N must equal H*W"

        # -------------------------
        # Token → 2D
        # -------------------------
        feat_2d = x.permute(0, 2, 1).contiguous().view(B, C, H, W)

        # -------------------------
        # CNN importance
        # -------------------------
        score_2d = self.importance_net(feat_2d)  # (B,1,H,W)

        # -------------------------
        # Back to token
        # -------------------------
        score = score_2d.view(B, -1)  # (B,N)

        # -------------------------
        # Masking
        # -------------------------
        if self.training and self.soft_training:
            mask = torch.sigmoid(score)
        else:
            with torch.no_grad():
                hard_mask = topk_mask(score, self.keep_ratio)

            soft_mask = torch.sigmoid(score)
            mask = hard_mask + soft_mask - soft_mask.detach()

        mask = mask.unsqueeze(-1)  # (B,N,1)

        # -------------------------
        # Apply pruning (stable version)
        # -------------------------
        x_pruned = x * mask + x * (1 - mask) * 0.1

        return x_pruned, mask.squeeze(-1), score
