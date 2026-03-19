"""
Random seed control utilities.

This module exists to make multi-seed experiments and exact restarts reproducible.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    Seed Python, NumPy, and PyTorch RNGs.

    Parameters
    ----------
    seed:
        Integer random seed used across the experiment.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Enable deterministic behaviour when possible.
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
