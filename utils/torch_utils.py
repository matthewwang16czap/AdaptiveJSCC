import numpy as np
import torch
import random
import os
from .universal_utils import makedirs


def save_model(model, save_path):
    makedirs(save_path)
    torch.save(model.state_dict(), save_path)


def load_weights(net, model_path):
    pretrained = torch.load(model_path)
    net.load_state_dict(pretrained, strict=False)
    del pretrained


def move_to_cpu(*tensors):
    for t in tensors:
        try:
            if torch.is_tensor(t):
                t.data = t.cpu()
        except Exception:
            pass


def mem_report(tag):
    torch.cuda.synchronize()
    print(
        f"{tag} allocated: {torch.cuda.memory_allocated()/1e6:.1f}MB, reserved: {torch.cuda.memory_reserved()/1e6:.1f}MB"
    )


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
                if "token_pruner" in name or "token_adapter" in name:
                    param.requires_grad = (
                        True if "token_pruner" in config.training_modules else False
                    )
        if config.channel_pruner:
            for name, param in net.named_parameters():
                if "channel_pruner" in name or "channel_adapter" in name:
                    param.requires_grad = (
                        True if "channel_pruner" in config.training_modules else False
                    )
        if config.encoder_adapter:
            for name, param in net.encoder.named_parameters():
                if "encoder_adapter" in name:
                    param.requires_grad = (
                        True if "encoder_adapter" in config.training_modules else False
                    )
        if config.decoder_adapter:
            for name, param in net.decoder.named_parameters():
                if "decoder_adapter" in name:
                    param.requires_grad = (
                        True if "decoder_adapter" in config.training_modules else False
                    )
                    param.requires_grad = False
        if config.sr:
            for name, param in net.named_parameters():
                if "sr" in name:
                    param.requires_grad = (
                        True if "sr" in config.training_modules else False
                    )
