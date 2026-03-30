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
    def __init__(self, dim, hidden_dim=64, soft_training=False):
        super().__init__()
        # keep_ratio removed from __init__
        self.soft_training = soft_training
        self.importance_net = CNNImportance(dim, hidden_dim)

    def forward(self, x, H, W, keep_ratio):
        """
        x: (B, N, C) where N = H * W
        keep_ratio: float, dynamic ratio for this forward pass
        """
        B, N, C = x.shape
        # 1. Generate Importance Scores
        feat_2d = x.transpose(1, 2).reshape(B, C, H, W)
        score_2d = self.importance_net(feat_2d)
        score = score_2d.view(B, N)
        # 2. Generate Binary Mask with Gradient Flow (STE)
        if self.training and self.soft_training:
            mask = torch.sigmoid(score)
        else:
            with torch.no_grad():
                # Use the passed keep_ratio here
                hard_mask = topk_mask(score, keep_ratio)
            soft_mask = torch.sigmoid(score)
            mask = hard_mask + (soft_mask - soft_mask.detach())
        # 3. Apply Mask (B, N, 1)
        mask = mask.unsqueeze(-1)
        # Scale by (1/keep_ratio) using the dynamic ratio
        if self.training and self.soft_training:
            x = (x * mask) / (keep_ratio + 1e-6)
        else:
            actual_keep_ratio = mask.mean()
            x = (x * mask) / (actual_keep_ratio + 1e-6)
        return x, mask


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
        self.prune1 = SwinTokenPrunerCNN(96)
        self.prune2 = SwinTokenPrunerCNN(192)
        self.prune3 = SwinTokenPrunerCNN(384)
        self.prune4 = SwinTokenPrunerCNN(768)

    def forward(self, x, H, W, ratios=[0.9, 0.7, 0.5, 0.3]):
        """
        ratios: List of floats for each pruning stage
        """
        x = self.stage1(x)
        x, _ = self.prune1(x, H, W, ratios[0])
        x = self.stage2(x)
        x, _ = self.prune2(x, H, W, ratios[1])
        x = self.stage3(x)
        x, _ = self.prune3(x, H, W, ratios[2])
        x = self.stage4(x)
        x, _ = self.prune4(x, H, W, ratios[3])
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
