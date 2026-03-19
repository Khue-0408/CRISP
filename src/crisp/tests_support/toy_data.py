"""Synthetic tensors used by unit tests and smoke checks."""

from __future__ import annotations

import torch


def make_toy_batch(
    batch_size: int = 2,
    image_size: int = 32,
) -> dict[str, torch.Tensor]:
    """
    Construct a deterministic toy batch with square foreground masks.

    Returns
    -------
    dict[str, torch.Tensor]
        Batch dictionary compatible with the trainer/evaluator.
    """
    images = torch.zeros(batch_size, 3, image_size, image_size)
    masks = torch.zeros(batch_size, 1, image_size, image_size)
    lo = image_size // 4
    hi = image_size - lo
    masks[:, :, lo:hi, lo:hi] = 1.0
    return {"image": images, "mask": masks}
