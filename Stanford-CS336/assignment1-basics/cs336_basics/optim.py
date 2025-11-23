import math
from collections.abc import Callable

import torch
import torch.optim as optim


class AdamW(optim.Optimizer):
    def __init__(
        self, params, lr: float, weight_decay: float, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):  # type: ignore
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                state = self.state[param]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(param)
                    state["v"] = torch.zeros_like(param)

                m, v = state["m"], state["v"]
                state["step"] += 1
                t = state["step"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                step_size_corr = (1 - beta2**t) ** 0.5 / (1 - beta1**t)
                step_size = lr * step_size_corr

                param.addcdiv_(m, v.sqrt().add_(eps), value=-step_size)
                param.mul_(1 - lr * weight_decay)

        return loss

    def set_lr(self, lr: float):
        for group in self.param_groups:
            group["lr"] = lr


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate

    elif it <= cosine_cycle_iters:
        progress_frac = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        increment_lr = (1.0 + math.cos(progress_frac * math.pi)) * (max_learning_rate - min_learning_rate) / 2.0
        return min_learning_rate + increment_lr

    else:
        return min_learning_rate
