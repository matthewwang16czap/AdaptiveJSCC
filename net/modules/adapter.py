import torch
from torch import nn
from einops import rearrange


class LinearAdapter(nn.Module):
    def __init__(self, dim, snr_adaptive=False):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.snr_adaptive = snr_adaptive
        if snr_adaptive:
            self.snr_scale = nn.Linear(1, dim)

    def forward(self, x, snr=None):
        y = self.linear(x)
        if self.snr_adaptive:
            assert snr is not None
            gamma = self.snr_scale(snr.view(-1, 1)).unsqueeze(1)
            y = gamma * y
        return x + y


class MLPAdapter(nn.Module):
    def __init__(self, dim, hidden_ratio=4, snr_adaptive=False):
        super().__init__()
        hidden_dim = dim * hidden_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim)
        )
        # initialize last layer to zero so adapter starts as identity
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.snr_adaptive = snr_adaptive
        if snr_adaptive:
            self.snr_scale = nn.Linear(1, dim)

    def forward(self, x, snr=None):
        y = self.mlp(x)
        if self.snr_adaptive:
            assert snr is not None
            gamma = self.snr_scale(snr.view(-1, 1)).unsqueeze(1)
            y = gamma * y
        return x + y


class ILAdapter(nn.Module):
    def __init__(self, dim, adapter_dim=8, kernel_size=3):
        super().__init__()
        # 1. Main SDS Branch (Spatial Downsampling Sub-network)
        self.adapter_down = nn.Linear(dim, adapter_dim)
        self.act = nn.GELU()
        # Depthwise Separable Conv (DWC + PW)
        self.adapter_conv = nn.Sequential(
            nn.Conv2d(
                adapter_dim,
                adapter_dim,
                kernel_size,
                padding=kernel_size // 2,
                groups=adapter_dim,
                bias=False,
            ),
            nn.BatchNorm2d(adapter_dim),
            self.act,
            nn.Conv2d(adapter_dim, adapter_dim, 1, bias=False),  # Pointwise
        )
        self.adapter_up = nn.Linear(adapter_dim, dim)
        # 2. RSDS Branch (Residual Spatial Downsampling)
        # We use a Depthwise Conv with Near-One Initialization to preserve the frozen signal
        self.rsds_conv = nn.Conv2d(
            dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=False
        )
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Xavier for Linear layers
        for m in [self.adapter_down, self.adapter_up]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        # Near-Ones Initialization for the RSDS path (The "Adaptation" secret sauce)
        # This ensures that at Epoch 0, the adapter acts like an Identity layer.
        nn.init.normal_(self.rsds_conv.weight, mean=1.0, std=0.001)

    def forward(self, x, H, W):
        # x shape: (B, L, D) where L = H * W
        B, L, D = x.shape
        identity = x
        # Process Main Branch (SDS)
        # Downsample channels
        x_sds = self.adapter_down(x)  # (B, L, adapter_dim)
        x_sds = self.act(x_sds)
        # Spatial processing (Convert to 2D)
        x_sds = rearrange(x_sds, "b (h w) d -> b d h w", h=H, w=W)
        x_sds = self.adapter_conv(x_sds)
        x_sds = rearrange(x_sds, "b d h w -> b (h w) d")
        # Upsample channels
        x_sds = self.adapter_up(x_sds)
        # Process Residual Branch (RSDS)
        # This path corrects the noise in the original feature map
        identity = rearrange(identity, "b (h w) d -> b d h w", h=H, w=W)
        identity = self.rsds_conv(identity)
        identity = rearrange(identity, "b d h w -> b (h w) d")
        return x_sds + identity
