"""
Boundary posterior target construction.

CRISP defines the boundary-local posterior target:

t*(u) = (y(u) + lambda * w_b(u) * p_T(u)) / (1 + lambda * w_b(u)).

This target reduces to the hard label away from the boundary and becomes a
teacher-informed soft target near the interface. [file:1]

CRISP reference: instruct.md §5, §8.
"""

from __future__ import annotations

import torch


def compute_boundary_posterior_target(
    mask: torch.Tensor,
    boundary_weight: torch.Tensor,
    teacher_posterior: torch.Tensor,
    lambda_value: float,
) -> torch.Tensor:
    """
    Compute the CRISP boundary posterior target t*(u).

    Parameters
    ----------
    mask:
        Binary ground-truth mask y of shape [B, 1, H, W].
    boundary_weight:
        Boundary weighting field w_b of shape [B, 1, H, W].
    teacher_posterior:
        Aggregated teacher posterior p_T of shape [B, 1, H, W].
    lambda_value:
        Boundary posterior mixing strength (default 0.80).

    Returns
    -------
    torch.Tensor
        Boundary posterior target t* of shape [B, 1, H, W].

    CRISP reference
    ---------------
    instruct.md §5:
      t*(u) = (y(u) + λ w_b(u) p_T(u)) / (1 + λ w_b(u))
    """
    if mask.shape != boundary_weight.shape or mask.shape != teacher_posterior.shape:
        raise ValueError("mask, boundary_weight, and teacher_posterior must have matching shapes.")

    y = mask.float()
    p_t = teacher_posterior.detach().clamp(0.0, 1.0)
    lam_wb = lambda_value * boundary_weight  # [B, 1, H, W]
    numerator = y + lam_wb * p_t
    denominator = 1.0 + lam_wb
    t_star = numerator / denominator
    return t_star


def clip_posterior_target(
    target: torch.Tensor,
    eps_target: float,
) -> torch.Tensor:
    """
    Clip the boundary posterior target into a numerically stable open interval.

    Parameters
    ----------
    target:
        Boundary posterior target t*.
    eps_target:
        Clipping parameter epsilon (default 1e-4) to avoid degenerate 0/1 targets.

    Returns
    -------
    torch.Tensor
        Clipped target t*_eps in [eps_target, 1 - eps_target].

    CRISP reference
    ---------------
    instruct.md §8: t_eps(u) = clip(t*(u), eps, 1-eps).
    """
    return target.clamp(eps_target, 1.0 - eps_target)
