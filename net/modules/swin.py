import torch
from torch import nn
from timm.layers import to_2tuple
from .mlp import Mlp
from .window import window_partition, window_reverse, WindowAttention


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert shift_size < window_size
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
        )

    def compute_attn_mask(self, H, W, window_size, shift_size, device):
        if shift_size == 0:
            return None
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )
        w_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, window_size)
        mask_windows = mask_windows.view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        """
        Args:
            x: (B, H*W, C)
            H, W: spatial resolution of input feature
        """
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"
        # adapt window & shift at runtime
        window_size = min(self.window_size, H, W)
        shift_size = self.shift_size if window_size > self.shift_size else 0
        # save current x
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        # cyclic shift
        if shift_size > 0:
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
        # window partition
        x_windows = window_partition(x, window_size)
        x_windows = x_windows.view(-1, window_size * window_size, C)
        attn_mask = self.compute_attn_mask(H, W, window_size, shift_size, x.device)
        attn_windows = self.attn(x_windows, add_token=False, mask=attn_mask)
        # merge windows
        attn_windows = attn_windows.view(-1, window_size, window_size, C)
        x = window_reverse(attn_windows, window_size, H, W)
        # reverse cyclic shift
        if shift_size > 0:
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))
        x = x.view(B, H * W, C)
        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x
