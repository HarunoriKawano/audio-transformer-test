import torch
from torch.optim.lr_scheduler import CyclicLR


def get_cycle_scheduler(optimizer: torch.optim, max_lr: float, min_lr: float, warmup_steps: int, down_steps: int):
    return CyclicLR(optimizer, min_lr, max_lr, step_size_up=warmup_steps, step_size_down=down_steps, mode="triangular2")
