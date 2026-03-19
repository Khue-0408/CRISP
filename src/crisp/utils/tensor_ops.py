"""
Small tensor helpers used across the codebase.

Examples include:
- safe sigmoid/logit transforms,
- shape alignment,
- interpolation wrappers,
- mask thresholding.
"""

from __future__ import annotations

import torch


def safe_logit(prob: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """
    Compute a numerically stable logit transform on probabilities.

    Clamps input to [eps, 1-eps] before computing log(p / (1-p)) to avoid
    infinities at the boundaries.
    """
    p = prob.clamp(eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


def threshold_mask(prob: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Convert probabilities into a binary mask.

    Parameters
    ----------
    prob:
        Probability tensor of any shape.
    threshold:
        Decision threshold; pixels with prob >= threshold are set to 1.

    Returns
    -------
    torch.Tensor
        Binary mask with the same shape as *prob*, dtype float32.
    """
    return (prob >= threshold).float()
