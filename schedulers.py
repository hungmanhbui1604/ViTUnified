from torch.optim.lr_scheduler import LambdaLR 
import torch

def polynomial_scheduler(
    optimizer: torch.optim.Optimizer,
    total_iters: int,
    base_lr: float,
    lr_min: float,
    power: float,
) -> LambdaLR:
    """lr(t) = (base_lr - lr_min) * (1 - t / total_steps) ** power + lr_min"""

    def _lr_lambda(t: int) -> float:
        if t >= total_iters:
            return lr_min / base_lr
        decay = (1 - t / total_iters) ** power
        return (base_lr - lr_min) * decay / base_lr + lr_min / base_lr

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)