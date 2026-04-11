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
