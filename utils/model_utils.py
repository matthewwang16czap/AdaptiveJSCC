import torch


def quantize_symmetric(x, bits=8):
    qmax = 2 ** (bits - 1) - 1
    max_val = x.abs().max()
    scale = max_val / qmax
    scale = scale + 1e-8
    x_q = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int32)
    return x_q, scale


def dequantize_symmetric(x_q, scale):
    return x_q.float() * scale


def freeze_model(net, config):
    if config.training:
        # base network unfreeze
        for name, param in net.named_parameters():
            if "base" in config.training_modules:
                param.requires_grad = True
            else:
                param.requires_grad = False
            # module-specific unfreeze in order
        if config.token_pruner:
            for name, param in net.named_parameters():
                if (
                    "token_pruning_adapters" in name
                    or "token_pruner" in name
                    # or "pruner_adapter" in name
                ):
                    param.requires_grad = (
                        True if "token_pruner" in config.training_modules else False
                    )
        if config.channel_pruner:
            for name, param in net.named_parameters():
                if (
                    "channel_pruning_adapters" in name
                    or "channel_pruner" in name
                    or "pruner_adapter" in name
                ):
                    param.requires_grad = (
                        True if "channel_pruner" in config.training_modules else False
                    )
        if config.snr_adapter:
            for name, param in net.encoder.named_parameters():
                if "snr_adapters" in name:
                    param.requires_grad = (
                        True if "snr_adapter" in config.training_modules else False
                    )


# size are total number of elements
def cbr_to_keep_ratio(cbr, img_size, feature_size, mask_size, quant_bits):
    if quant_bits is None:
        quant_bits = 16  # default to FP16/BF16 setup
    feature_precision_ratio = quant_bits / 8  # 8 bits int for image
    mask_precision_ratio = 1 / 8  # 1 bit per mask element
    compression_ratio = (
        feature_size * feature_precision_ratio + mask_size * mask_precision_ratio
    ) / img_size
    return cbr / compression_ratio


def keep_ratio_to_cbr(keep_ratio, img_size, feature_size, mask_size, quant_bits):
    if quant_bits is None:
        quant_bits = 16  # default to FP16/BF16 setup
    feature_precision_ratio = quant_bits / 8  # 8 bits int for image
    mask_precision_ratio = 1 / 8  # 1 bit per mask element
    compression_ratio = (
        feature_size * feature_precision_ratio + mask_size * mask_precision_ratio
    ) / img_size
    return keep_ratio * compression_ratio


def compute_token_channel_keep_ratio(keep_ratio, token_channel_balance_ratio):
    """
    Args:
        keep_ratio (float): overall keep ratio (0, 1]
        token_channel_balance_ratio (float): alpha in [0, 1]

    Returns:
        token_keep_ratio, channel_keep_ratio
    """
    keep_ratio = max(min(keep_ratio, 1.0), 1e-12)
    alpha = max(min(token_channel_balance_ratio, 1.0), 0.0)
    token_keep_ratio = keep_ratio**alpha
    channel_keep_ratio = keep_ratio ** (1.0 - alpha)
    return token_keep_ratio, channel_keep_ratio
