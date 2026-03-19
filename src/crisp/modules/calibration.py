"""
Calibration-specific helper functions used during training.

This module provides the restricted calibrated family mapping:
  p̃(u) = sigmoid(α̂(u) · z(u))

CRISP reference: instruct.md §6.
"""

from __future__ import annotations

import torch


def calibrate_logits_with_alpha(
    logits: torch.Tensor,
    alpha_hat: torch.Tensor,
) -> torch.Tensor:
    """
    Apply the predicted inverse-temperature field to raw logits and return probabilities.

    Parameters
    ----------
    logits:
        Raw student logits z(u), shape [B, 1, H, W].
    alpha_hat:
        Predicted inverse-temperature field α̂(u), shape [B, 1, H, W].

    Returns
    -------
    torch.Tensor
        Calibrated foreground probabilities p̃(u) = sigmoid(α̂(u) · z(u)),
        shape [B, 1, H, W].

    CRISP reference
    ---------------
    instruct.md §6: Q(z(u)) = { sigmoid(α z(u)) : α ∈ [α_min, α_max] }.
    instruct.md §10: p̃(u) = sigmoid(α̂(u) · z(u)).
    """
    return torch.sigmoid(alpha_hat * logits)
