import torch.nn as nn
import torch


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

    """

    def __init__(self, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, H // 2, W // 2


class PatchReverseMerging(nn.Module):
    r"""Patch Reverse Merging (Upsampling) Layer.

    Args:
        dim (int): Number of input channels.
        out_dim (int): Number of output channels after upsampling.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        # Expand channels for PixelShuffle
        self.increment = nn.Linear(dim, out_dim * 4, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x, H, W):
        """
        Args:
            x: (B, H*W, C)
            H, W: spatial resolution of x before upsampling
        Returns:
            x: (B, 4*H*W, out_dim)  -> spatially (2H, 2W)
        """
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"
        x = self.norm(x)
        x = self.increment(x)  # (B, H*W, 4*out_dim)
        x = x.view(B, H, W, -1)
        x = x.permute(0, 3, 1, 2)  # (B, 4*out_dim, H, W)
        x = nn.PixelShuffle(2)(x)  # (B, out_dim, 2H, 2W)
        x = x.flatten(2).permute(0, 2, 1)  # (B, 4*H*W, out_dim)
        return x, H * 2, W * 2


class PatchEmbed(nn.Module):
    """Image to Patch Embedding

    Args:
        patch_size (int or tuple): Patch spatial size (kernel + stride) at input.
        in_chans (int): Number of input channels.
        embed_dim (int): Output embedding dimension.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            x: (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        if self.norm is not None:
            x = self.norm(x)
        return x, H // self.patch_size, W // self.patch_size


class PatchUnembed(nn.Module):
    """
    Reconstruct image from patch embeddings.
    Inverse of Conv2d-based PatchEmbed.
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        # Optional token-space normalization
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None
        # Linear projection back to raw patch pixels
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * in_chans)

    def forward(self, x, H, W):
        """
        x: (B, H*W, embed_dim)
        H, W: patch grid size
        """
        B, N, C = x.shape
        assert C == self.embed_dim
        assert N == H * W
        # Token-space normalization
        if self.norm is not None:
            x = self.norm(x)
        # Project tokens → raw patch pixels
        x = self.proj(x)
        # Reassemble patches
        x = x.view(B, H, W, self.in_chans, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, self.in_chans, H * self.patch_size, W * self.patch_size)
        return x, H * self.patch_size, W * self.patch_size
