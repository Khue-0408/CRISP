"""Native boundary-weighted consistency primitives for binary segmentation.

This module intentionally contains only the mathematical BWCR boundary field
and raw-logit consistency objective. Runtime view construction and Trainer
composition belong to separate integration layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


BWCR_LAMBDA_MIN = 0.01
BWCR_LAMBDA_MAX = 1.0
BWCR_RADIUS = 10.0
BWCR_ALPHA = 1.0


@dataclass(frozen=True)
class BWCRBoundaryField:
    """Native signed distance and its derived non-negative BWCR field.

    ``signed_distance`` is the native discrete signed distance before the
    empty/tiny-foreground fallback. ``distance`` and ``lambda_map`` include
    that fallback, and ``fallback_applied`` reports it per batch element.
    """

    signed_distance: torch.Tensor
    distance: torch.Tensor
    lambda_map: torch.Tensor
    fallback_applied: torch.Tensor


def _validate_binary_grid(tensor: torch.Tensor, *, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 4 or tensor.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B, 1, H, W]")
    if tensor.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    if not torch.all((tensor == 0) | (tensor == 1)):
        raise ValueError(f"{name} must be binary with values in {{0, 1}}")


def native_bwcr_lambda(distance: torch.Tensor) -> torch.Tensor:
    """Return the locked native BWCR linear weight for non-negative distance."""

    if not isinstance(distance, torch.Tensor):
        raise TypeError("distance must be a torch.Tensor")
    if not distance.is_floating_point():
        raise TypeError("distance must use a floating-point dtype")
    if not torch.isfinite(distance).all():
        raise ValueError("distance must contain only finite values")
    if torch.any(distance < 0):
        raise ValueError("distance must be non-negative")

    radial = torch.clamp(BWCR_RADIUS - distance, min=0.0) / BWCR_RADIUS
    return BWCR_LAMBDA_MAX * radial.pow(BWCR_ALPHA) + BWCR_LAMBDA_MIN


def native_bwcr_boundary_field(mask: torch.Tensor) -> BWCRBoundaryField:
    """Build the native discrete BWCR distance and linear boundary field.

    The input is a canonical binary source ground-truth mask. EDT computation
    is non-differentiable and performed independently for each batch element.
    Returned tensors preserve the input device and use its floating dtype (or
    ``float32`` for an integral/bool input).
    """

    _validate_binary_grid(mask, name="mask")
    output_dtype = mask.dtype if mask.is_floating_point() else torch.float32
    mask_numpy = mask.detach().to(device="cpu", dtype=torch.float64).numpy()

    signed_distances: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    fallback_flags: list[bool] = []

    for sample in mask_numpy[:, 0]:
        signed_distance = distance_transform_edt(sample) - distance_transform_edt(
            1.0 - sample
        )
        signed_distance = np.floor(signed_distance)
        signed_distance[signed_distance > 0] -= 1.0

        adjusted = signed_distance.copy()
        fallback_applied = bool(np.max(adjusted) < 1.0)
        if fallback_applied:
            adjusted[adjusted < 1.0] = BWCR_RADIUS + 1.0

        signed_distances.append(signed_distance)
        distances.append(np.abs(adjusted))
        fallback_flags.append(fallback_applied)

    signed_tensor = torch.as_tensor(
        np.stack(signed_distances)[:, None], dtype=output_dtype, device=mask.device
    )
    distance_tensor = torch.as_tensor(
        np.stack(distances)[:, None], dtype=output_dtype, device=mask.device
    )
    fallback_tensor = torch.tensor(
        fallback_flags, dtype=torch.bool, device=mask.device
    )
    return BWCRBoundaryField(
        signed_distance=signed_tensor,
        distance=distance_tensor,
        lambda_map=native_bwcr_lambda(distance_tensor),
        fallback_applied=fallback_tensor,
    )


def bwcr_consistency_loss(
    z1_inverse: torch.Tensor,
    z2_inverse: torch.Tensor,
    validity: torch.Tensor,
    lambda_map: torch.Tensor,
) -> torch.Tensor:
    """Return mean ``M * lambda(d) * (z1 - z2)^2`` over the full BHW grid."""

    if not isinstance(z1_inverse, torch.Tensor) or not isinstance(
        z2_inverse, torch.Tensor
    ):
        raise TypeError("z1_inverse and z2_inverse must be torch.Tensor values")
    if z1_inverse.ndim != 4 or z1_inverse.shape[1] != 1:
        raise ValueError("z1_inverse must have shape [B, 1, H, W]")
    if z2_inverse.shape != z1_inverse.shape:
        raise ValueError("z2_inverse must have the same shape as z1_inverse")
    if not z1_inverse.is_floating_point() or not z2_inverse.is_floating_point():
        raise TypeError("aligned logits must use floating-point dtypes")
    if not torch.isfinite(z1_inverse).all() or not torch.isfinite(z2_inverse).all():
        raise ValueError("aligned logits must contain only finite values")
    if z1_inverse.device != z2_inverse.device:
        raise ValueError("aligned logits must be on the same device")

    _validate_binary_grid(validity, name="validity")
    if validity.shape != z1_inverse.shape:
        raise ValueError("validity must have the same shape as aligned logits")
    if lambda_map.shape != z1_inverse.shape:
        raise ValueError("lambda_map must have the same shape as aligned logits")
    if not lambda_map.is_floating_point():
        raise TypeError("lambda_map must use a floating-point dtype")
    if not torch.isfinite(lambda_map).all():
        raise ValueError("lambda_map must contain only finite values")
    if torch.any(lambda_map < BWCR_LAMBDA_MIN) or torch.any(
        lambda_map > BWCR_LAMBDA_MAX + BWCR_LAMBDA_MIN
    ):
        raise ValueError("lambda_map must lie in the native BWCR range [0.01, 1.01]")
    if validity.device != z1_inverse.device or lambda_map.device != z1_inverse.device:
        raise ValueError("validity and lambda_map must share the logits device")
    if validity.requires_grad or lambda_map.requires_grad:
        raise ValueError("validity and lambda_map must not require gradients")

    squared_difference = (z1_inverse - z2_inverse).square()
    per_pixel = validity.to(squared_difference.dtype) * lambda_map.to(
        squared_difference.dtype
    ) * squared_difference
    return per_pixel.mean()
