import torch
from torch.optim.lr_scheduler import LambdaLR


def polynomial_scheduler(
    optimizer: torch.optim.Optimizer,
    total_iters: int,
    lr_min: float,
    power: float,
) -> LambdaLR:
    """lr(t) = (base_lr - lr_min) * (1 - t / total_steps) ** power + lr_min"""
    def get_lr_lambda(base_lr: float):
        def _lr_lambda(t: int) -> float:
            if t >= total_iters:
                return lr_min / base_lr
            decay = (1 - t / total_iters) ** power
            return (base_lr - lr_min) * decay / base_lr + lr_min / base_lr
        return _lr_lambda

    lr_lambdas = [get_lr_lambda(pg["lr"]) for pg in optimizer.param_groups]

    return LambdaLR(optimizer, lr_lambda=lr_lambdas)


def get_scheduler(
    sched_name: str,
    optimizer: torch.optim.Optimizer,
    iters: int,
    epochs: int,
    sched_cfg: dict,
):
    total_iters = iters * epochs
    # warmup_iters = iters * sched_cfg["warmup_epochs"]

    if sched_name == "polynomial":
        return polynomial_scheduler(
            optimizer=optimizer,
            total_iters=total_iters,
            lr_min=sched_cfg["min_lr"],
            power=sched_cfg["power"],
        )

    raise ValueError("Unknown scheduler: " + sched_name)
