from collections.abc import Iterable
import math
import os
from typing import Tuple
import typing

import numpy as np
import torch


def cosine_learning_rate_schedule(
    epoch: int, min_lr: float, max_lr: float, warmup_epochs: int, cosine_epochs: int
) -> float:
    """
    Cosine learning rate schedule function.
    """
    if epoch < warmup_epochs:
        return epoch / warmup_epochs * max_lr
    elif warmup_epochs <= epoch <= cosine_epochs:
        return min_lr + 0.5 * (max_lr - min_lr) * (
            1
            + math.cos(
                math.pi * (epoch - warmup_epochs) / (cosine_epochs - warmup_epochs)
            )
        )
    else:
        return min_lr


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float = 1.0,
    eps: float = 1e-6,
) -> None:
    """
    Gradient clipping function.
    """
    grads = [p.grad for p in parameters if p.grad is not None]

    if not grads:
        return

    stacked_grads = torch.stack([torch.norm(g.detach(), p=2) for g in grads])
    total_norm = torch.norm(stacked_grads, p=2)

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for grad in grads:
            grad *= scale


def data_loading(
    x: np.ndarray, batch_size: int, context_length: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load data from a numpy array and return a pair of tensors.
    """
    if len(x) <= context_length:
        raise ValueError(
            f"Input array length {len(x)} is less than context length {context_length}."
        )

    sample_idx = np.random.randint(0, len(x) - context_length, size=(batch_size,))

    sample_x = np.array([x[i : i + context_length] for i in sample_idx])
    sample_y = np.array([x[i + 1 : i + context_length + 1] for i in sample_idx])

    device = torch.device(device)
    return torch.from_numpy(sample_x).to(device=device), torch.from_numpy(sample_y).to(
        device=device
    )


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    """
    should dump all the state from the first three parameters into the file-like object out. You can
    use the state_dict method of both the model and the optimizer to get their relevant states and use
    torch.save(obj, out) to dump obj into out (PyTorch supports either a path or a file-like object here).
    A typical choice is to have obj be a dictionary, but you can use whatever format you want as long as
    you can load your checkpoint later.

    This function expects the following parameters:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    iteration: int
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """
    should load a checkpoint from src (path or file-like object), and then recover the model
    and optimizer states from that checkpoint. Your function should return the iteration number
    that was saved to the checkpoint. You can use torch.load(src) to recover what you saved in
    your save_checkpoint implementation, and the load_state_dict method in both the model and
    optimizers to return them to their previous states.

    This function expects the following parameters:
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    """
    chceckpoint = torch.load(src)
    model.load_state_dict(chceckpoint["model_state_dict"])
    optimizer.load_state_dict(chceckpoint["optimizer_state_dict"])

    return chceckpoint["iteration"]


def get_perplexity(loss):
    """Calculate perplexity from cross-entropy loss"""
    return math.exp(min(loss, 10))

