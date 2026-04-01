from torch import nn
from .modules import (
    SwinTransformerBlock,
    MLPAdapter,
    PatchUnembed,
    PatchReverseMerging,
    SwinTokenWiseChannelAdapter,
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
        upsample=None,
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
        # Upsampling layer
        self.upsample = upsample
        self.token_adapter = (
            SwinTokenWiseChannelAdapter(dim) if use_token_pruner else None
        )
        self.channel_adapter = (
            SwinTokenWiseChannelAdapter(dim) if use_channel_pruner else None
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

    def forward(self, x, H, W, snr):
        """
        Args:
            x: (B, H*W, C)
            H, W: spatial resolution
        Returns:
            x: updated features
            H, W: updated resolution
        """
        # Before each block, apply token/channel adapters
        x = self.token_adapter(x) if self.token_adapter is not None else x
        x = self.channel_adapter(x) if self.channel_adapter is not None else x
        # Swin blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x, H, W)
            x = self.adapters[i](x, snr=snr) if self.adapters is not None else x
        x, H, W = self.upsample(x, H, W) if self.upsample is not None else (x, H, W)
        return x, H, W


class SwinJSCC_Decoder(nn.Module):
    def __init__(
        self,
        patch_size,
        out_chans,
        embed_dims,
        depths,
        num_heads,
        window_size=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_adapter=False,
        use_token_pruner=False,
        use_channel_pruner=False,
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dims = embed_dims
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.patch_unembed = PatchUnembed(
            patch_size,
            out_chans,
            embed_dims[-1],
            norm_layer if patch_norm else None,
        )
        # Build decoder layers (low → high resolution)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_dim = embed_dims[i_layer]
            layer_out_dim = (
                embed_dims[i_layer + 1]
                if i_layer < self.num_layers - 1
                else embed_dims[-1]
            )
            layer_upsample = (
                PatchReverseMerging(
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
                upsample=layer_upsample,
                use_adapter=use_adapter,
                use_token_pruner=use_token_pruner,
                use_channel_pruner=use_channel_pruner,
            )
            self.layers.append(layer)
        self.tanh = nn.Tanh()
        self.apply(self._init_weights)

    # Forward
    def forward(self, x, snr, H, W, valid=None):
        """
        x: (B, L, C)  latent symbols
        """
        # Initial low-resolution grid
        B, L, C = x.shape
        assert H * W == L, "feature size does not match H, W"
        # Decoder backbone
        for layer in self.layers:
            x, H, W = layer(x, H, W, snr)
        # Tokens → image
        x, H, W = self.patch_unembed(x, H, W)
        # Nornalize to [0,1]
        x = self.tanh(x)
        x = 0.5 * (x + 1.0)
        if valid is not None:
            if valid.dim() == 3:
                valid = valid.unsqueeze(1)
            return x * valid
        else:
            return x

    # Init
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def create_decoder(**kwargs):
    model = SwinJSCC_Decoder(**kwargs)
    return model
