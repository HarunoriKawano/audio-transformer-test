import math

import torch
from torch.optim.lr_scheduler import LambdaLR


class CosineDecayScheduler:
    def __init__(self, warmup_steps: int, max_steps: int,
                 max_warmup_steps: int = 10000, min_weight: float = 0.05):
        if max_warmup_steps < warmup_steps:
            self._warmup_steps = max_warmup_steps
        else:
            self._warmup_steps = warmup_steps
        self._max_steps = max_steps
        self._min_weight = min_weight

    def __call__(self, epoch: int):
        epoch = max(epoch, 1)
        if epoch <= self._warmup_steps:
            return epoch / self._warmup_steps
        epoch -= 1
        rad = math.pi * (epoch - self._warmup_steps) / (self._max_steps - self._warmup_steps)
        weight = (math.cos(rad) + 1.) / 2

        return max(weight, self._min_weight)


def get_cosine_scheduler(
        optimizer: torch.optim, warmup_steps: int, max_steps: int,
        max_warmup_steps: int = 10000, min_weight: float = 0.05
):
    return LambdaLR(optimizer,
                    CosineDecayScheduler(warmup_steps, max_steps, max_warmup_steps, min_weight)
                    )
