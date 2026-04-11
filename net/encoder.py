import torch
from torch import nn
from utils.model_utils import compute_token_channel_keep_ratio, cbr_to_keep_ratio
from .modules import (
    SwinTransformerBlock,
    MLPSNRAdapter,
    PatchEmbed,
    PatchMerging,
    EncoderTokenPruner,
    EncoderChannelPruner,
    EncoderIntermediatePrunerAdapter,
)
from timm.layers import trunc_normal_


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
        use_snr_adapter=False,
        use_token_pruner=False,
        use_channel_pruner=False,
        module_hidden_ratio=1,
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
        self.token_pruning_adapters = (
            nn.ModuleList(
                [
                    EncoderIntermediatePrunerAdapter(
                        dim, hidden_ratio=module_hidden_ratio
                    )
                    for _ in range(depth)
                ]
            )
            if use_token_pruner
            else None
        )
        self.channel_pruning_adapters = (
            nn.ModuleList(
                [
                    EncoderIntermediatePrunerAdapter(
                        dim, hidden_ratio=module_hidden_ratio
                    )
                    for _ in range(depth)
                ]
            )
            if use_channel_pruner
            else None
        )
        self.snr_adapters = (
            nn.ModuleList(
                [
                    MLPSNRAdapter(
                        dim, hidden_ratio=module_hidden_ratio, snr_adaptive=True
                    )
                    for _ in range(depth)
                ]
            )
            if use_snr_adapter is True
            else None
        )

    def forward(self, x, H, W, snr=None):
        """
        Args:
            x: (B, H*W, C)
            H, W: spatial resolution
        Returns:
            x: transformed features
            H, W: updated resolution
        """
        if self.snr_adapters is not None and snr is None:
            raise ValueError("SNR must be provided if using SNR adapters")
        for i, blk in enumerate(self.blocks):
            x = blk(x, H, W)
            x = self.snr_adapters[i](x, snr=snr) if self.snr_adapters is not None else x
            x = (
                (self.channel_pruning_adapters[i](x))
                if self.channel_pruning_adapters is not None
                else x
            )
            x = (
                (self.token_pruning_adapters[i](x))
                if self.token_pruning_adapters is not None
                else x
            )
        x, H, W = self.downsample(x, H, W) if self.downsample is not None else (x, H, W)
        return x, H, W


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
        use_snr_adapter=False,
        use_token_pruner=False,
        use_channel_pruner=False,
        module_hidden_ratio=1,
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.patch_norm = patch_norm
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.mlp_ratio = mlp_ratio
        self.quant_bits = quant_bits
        self.use_snr_adapter = use_snr_adapter
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
                use_snr_adapter=use_snr_adapter,
                use_token_pruner=use_token_pruner,
                use_channel_pruner=use_channel_pruner,
            )
            self.layers.append(layer)
        self.channel_pruner = (
            EncoderChannelPruner(embed_dims[-1], hidden_ratio=module_hidden_ratio)
            if use_channel_pruner
            else None
        )
        self.token_pruner = (
            EncoderTokenPruner(embed_dims[-1], hidden_ratio=module_hidden_ratio)
            if use_token_pruner
            else None
        )
        self.norm = norm_layer(embed_dims[-1])
        self.apply(self._init_weights)

    def forward(self, x, snr=None, cbr=None, token_channel_balance_ratio=0.1):
        if self.use_snr_adapter and snr is None:
            raise ValueError("SNR must be provided if using SNR adapters")
        if (self.use_token_pruner or self.use_channel_pruner) and cbr is None:
            raise ValueError(
                "CBR must be provided if using token pruner or channel pruner"
            )
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
        mask_numel = 0
        # Backbone
        for i, layer in enumerate(self.layers):
            x, H, W = layer(x, H, W, snr)
        # pruner
        keep_ratio = cbr_to_keep_ratio(
            cbr, img_numel, output_feature_numel, mask_numel, self.quant_bits
        )
        token_keep_ratio, channel_keep_ratio = None, None
        if self.use_token_pruner and self.use_channel_pruner:
            token_keep_ratio, channel_keep_ratio = compute_token_channel_keep_ratio(
                keep_ratio, token_channel_balance_ratio
            )
        elif self.use_token_pruner:
            token_keep_ratio = keep_ratio
        elif self.use_channel_pruner:
            channel_keep_ratio = keep_ratio
        x = (
            self.channel_pruner(x, channel_keep_ratio)
            if self.channel_pruner is not None
            else x
        )
        x = (
            self.token_pruner(x, H, W, token_keep_ratio)
            if self.token_pruner is not None
            else x
        )
        # normalize before output
        x = self.norm(x)
        return x, H, W

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
