"""
Sampler helpers for training and evaluation dataloaders.

This module is intentionally small but exists to keep dataloader concerns
separate from dataset definitions.

Potential future use cases
--------------------------
- distributed data parallel samplers,
- weighted sampling for class imbalance,
- center-aware or patient-aware sampling.
"""

from __future__ import annotations

from typing import Any, Optional

from torch.utils.data import DistributedSampler, RandomSampler, Sampler

from crisp.utils.dist import is_distributed


def build_train_sampler(dataset: Any, distributed: bool = False) -> Optional[Sampler]:
    """
    Build the sampler used for training.

    Parameters
    ----------
    dataset:
        Dataset instance.
    distributed:
        Whether training is running under distributed data parallel.

    Returns
    -------
    Optional[Sampler]
        A ``DistributedSampler`` if distributed, otherwise ``None``
        (PyTorch DataLoader will default to ``RandomSampler``).
    """
    if distributed or is_distributed():
        return DistributedSampler(dataset, shuffle=True)
    return None
