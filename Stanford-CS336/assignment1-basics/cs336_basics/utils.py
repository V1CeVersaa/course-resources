import os
import typing
from collections.abc import Iterable

import numpy
import numpy.typing as npt
import torch
from jaxtyping import Float, Int
from torch.types import Tensor


def softmax(in_features: Tensor, dim: int = -1):
    max_value = in_features.max(dim=dim, keepdim=True).values
    exp_shift = torch.exp(in_features - max_value)
    sum_exp = exp_shift.sum(dim=dim, keepdim=True)
    return exp_shift / sum_exp


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    batch_size = inputs.shape[0]
    log_sfm = torch.nn.functional.log_softmax(inputs, dim=-1)
    target_prob = -log_sfm[torch.arange(batch_size, device=inputs.device), targets]
    return target_prob.mean()


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    counted_parameters = [p for p in parameters if p.grad is not None]

    total_norm = 0.0
    for param in counted_parameters:
        param_norm = torch.norm(param.grad, p=2)
        total_norm += param_norm.item() ** 2

    total_norm = total_norm**0.5

    if total_norm > max_l2_norm:
        scale_factor = max_l2_norm / (1e-6 + total_norm)

        for param in parameters:
            if param.grad is not None:
                param.grad.mul_(scale_factor)


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start_idx = len(dataset) - context_length
    start_indices = numpy.random.randint(0, max_start_idx, size=batch_size)

    indices = start_indices[:, None] + numpy.arange(context_length)[None, :]
    x = dataset[indices]
    y = dataset[indices + 1]

    x_tensor = torch.from_numpy(x).long().to(device)
    y_tensor = torch.from_numpy(y).long().to(device)

    return x_tensor, y_tensor


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    checkpoint = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
