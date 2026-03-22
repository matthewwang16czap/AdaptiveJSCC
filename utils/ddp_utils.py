import torch
import random
import torch.distributed as dist
import os


def setup_ddp():
    """Initialize distributed environment and return device info."""
    # Get local rank (provided by torchrun)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    # Return config dict for convenience
    return {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "device_id": local_rank,
        "device": torch.device(f"cuda:{local_rank}"),
    }


def cleanup_ddp():
    dist.destroy_process_group()


def initialize_ddp():
    if "LOCAL_RANK" in os.environ:
        ddp_env = setup_ddp()
    else:
        ddp_env = {
            "rank": 0,
            "world_size": 1,
            "device_id": 0,
            "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        }
    return ddp_env


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def sample_choice_ddp(choices, device):
    if not dist.is_initialized():
        return random.choice(choices)
    if dist.get_rank() == 0:
        choice = random.choice(choices)
        choice_tensor = torch.tensor([choice], device=device)
    else:
        choice_tensor = torch.zeros(1, device=device)
    dist.broadcast(choice_tensor, src=0)
    return choice_tensor


def mean_ddp(x):
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x /= dist.get_world_size()
    return x
