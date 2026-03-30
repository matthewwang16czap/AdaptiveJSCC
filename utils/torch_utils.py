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
        if config.encoder_adapter:
            for name, param in net.encoder.named_parameters():
                if "adapters" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        if config.decoder_adapter:
            for name, param in net.decoder.named_parameters():
                if "adapters" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        if config.sr:
            for name, param in net.named_parameters():
                if "sr" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        # train base model
        if config.encoder_adapter + config.decoder_adapter + config.sr == False:
            for name, param in net.named_parameters():
                param.requires_grad = True
