import torch
from torch import nn
from utils.universal_utils import cbr_to_keep_ratio
from .modules import (
    SwinTransformerBlock,
    MLPAdapter,
    PatchEmbed,
    PatchMerging,
    SwinTokenPruner,
    SwinChannelPruner,
    SwinTokenWiseChannelPruner,
)
from timm.layers import trunc_normal_
import math


def get_keep_ratio(stage_idx, num_stages, target_ratio, mode="last"):
    """
    Returns keep ratio for a single stage.

    Args:
        stage_idx: int, index of current stage (0-based)
        num_stages: total number of stages
        target_ratio: final keep ratio at last stage
        mode: "exp", "linear", or "cosine"
    """
    if num_stages == 1:
        return target_ratio
    if not (0 <= stage_idx < num_stages):
        raise ValueError("stage_idx out of range")
    t = stage_idx / (num_stages - 1)
    if mode == "exp":
        r = target_ratio**t
    elif mode == "linear":
        r = 1.0 - t * (1.0 - target_ratio)
    elif mode == "cosine":
        r = target_ratio + (1 - target_ratio) * (0.5 * (1 + math.cos(math.pi * t)))
    elif mode == "last":
        r = target_ratio if stage_idx == num_stages - 1 else 1.0
    else:
        raise ValueError("Unknown mode")
    return r


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_adapter=False,
        use_token_pruner=False,
        use_channel_pruner=False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        self.token_pruner = SwinTokenPruner(dim) if use_token_pruner else None
        self.channel_pruner = (
            SwinTokenWiseChannelPruner(dim) if use_channel_pruner else None
        )
        self.adapters = (
            nn.ModuleList(
                [
                    MLPAdapter(dim, hidden_ratio=1, snr_adaptive=True)
                    for _ in range(depth)
                ]
            )
            if use_adapter is True
            else None
        )

    def forward(self, x, H, W, snr, keep_ratio):
        """
        Args:
            x: (B, H*W, C)
            H, W: spatial resolution
        Returns:
            x: transformed features
            H, W: updated resolution
        """
        mask = None
        for i, blk in enumerate(self.blocks):
            x = blk(x, H, W)
            x = self.adapters[i](x, snr=snr) if self.adapters is not None else x
        x, mask = (
            self.token_pruner(x, keep_ratio)
            if self.token_pruner is not None
            else (x, mask)
        )
        x, mask = (
            self.channel_pruner(x, keep_ratio)
            if self.channel_pruner is not None
            else (x, mask)
        )
        x, H, W = self.downsample(x, H, W) if self.downsample is not None else (x, H, W)
        return x, mask, H, W


class SwinJSCC_Encoder(nn.Module):
    def __init__(
        self,
        patch_size,
        in_chans,
        embed_dims,
        depths,
        num_heads,
        window_size=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        quant_bits=None,
        use_adapter=False,
        use_token_pruner=False,
        use_channel_pruner=False,
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.patch_norm = patch_norm
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.mlp_ratio = mlp_ratio
        self.quant_bits = quant_bits
        self.use_adapter = use_adapter
        self.use_token_pruner = use_token_pruner
        self.use_channel_pruner = use_channel_pruner
        # Patch embedding
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
            norm_layer=norm_layer if patch_norm else None,
        )
        # Encoder stages
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_dim = embed_dims[i_layer]
            layer_out_dim = (
                embed_dims[i_layer + 1]
                if i_layer < self.num_layers - 1
                else embed_dims[-1]
            )
            layer_downsample = (
                PatchMerging(
                    dim=layer_dim,
                    out_dim=layer_out_dim,
                    norm_layer=norm_layer,
                )
                if i_layer < self.num_layers - 1
                else None
            )
            layer = BasicLayer(
                dim=layer_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer,
                downsample=layer_downsample,
                use_adapter=use_adapter,
                use_token_pruner=use_token_pruner,
                use_channel_pruner=use_channel_pruner,
            )
            self.layers.append(layer)
        self.norm = norm_layer(embed_dims[-1])
        self.apply(self._init_weights)

    def forward(self, x, snr, cbr):
        img_numel = x.numel()
        # Patch embedding
        x, H, W = self.patch_embed(x)
        # Generate keep ratios for each stage
        # output_feature have numel = (B, H*W, C) where H and W are downsampled by 2^(num_stages -1) each
        # as patch embed has already downsampled once.
        output_feature_numel = (
            x.shape[0]
            * self.embed_dims[-1]
            * x.shape[1]
            // (2 ** ((len(self.layers) - 1) * 2))
        )
        mask_numel = img_numel if self.use_channel_pruner else x.shape[0] * x.shape[1]
        target_ratio = cbr_to_keep_ratio(
            cbr, img_numel, output_feature_numel, mask_numel, self.quant_bits
        )
        # Backbone
        for i, layer in enumerate(self.layers):
            keep_ratio = get_keep_ratio(i, len(self.layers), target_ratio)
            x, mask, H, W = layer(x, H, W, snr, keep_ratio)
        x = self.norm(x)
        return x, mask, H, W

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}


def create_encoder(**kwargs):
    model = SwinJSCC_Encoder(**kwargs)
    return model
