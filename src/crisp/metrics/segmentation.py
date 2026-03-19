"""
Segmentation geometry metrics.

The main metrics needed for the CRISP replication are:
- Dice,
- Boundary-F1,
- HD95.

These metrics are used to quantify both global overlap and boundary geometry. [file:1]

CRISP reference: instruct.md §16.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice_score(
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """
    Compute Dice score between binary prediction and ground-truth mask.

    Parameters
    ----------
    pred_mask:
        Binary predicted mask [B, 1, H, W] or [H, W].
    true_mask:
        Binary ground-truth mask, same shape.
    eps:
        Smoothing epsilon.

    Returns
    -------
    torch.Tensor
        Scalar Dice score in [0, 1].
    """
    p = pred_mask.float().reshape(-1)
    t = true_mask.float().reshape(-1)
    intersection = (p * t).sum()
    return (2.0 * intersection + eps) / (p.sum() + t.sum() + eps)


def boundary_f1_score(
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
    tolerance: int = 2,
) -> torch.Tensor:
    """
    Compute Boundary-F1 score.

    Parameters
    ----------
    pred_mask:
        Binary predicted mask [H, W] or [1, H, W].
    true_mask:
        Binary ground-truth mask, same spatial shape.
    tolerance:
        Pixel tolerance for matching boundary points.

    Returns
    -------
    torch.Tensor
        Boundary-F1 score.

    Notes
    -----
    Extracts contours using morphological gradient, then matches boundary
    pixels within the tolerance distance.
    """
    def _extract_contour(mask_np: np.ndarray) -> np.ndarray:
        """Extract 1-pixel boundary via morphological gradient."""
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(mask_np, iterations=1)
        return (mask_np.astype(bool) ^ eroded).astype(np.float64)

    pred_np = pred_mask.detach().cpu().squeeze().numpy().astype(np.float64)
    true_np = true_mask.detach().cpu().squeeze().numpy().astype(np.float64)

    pred_boundary = _extract_contour(pred_np)
    true_boundary = _extract_contour(true_np)

    # If either has no boundary, handle edge cases.
    if pred_boundary.sum() == 0 and true_boundary.sum() == 0:
        return torch.tensor(1.0)
    if pred_boundary.sum() == 0 or true_boundary.sum() == 0:
        return torch.tensor(0.0)

    # Compute distance from each boundary pixel to the other set.
    # distance_transform_edt on (1 - boundary) gives distance to nearest boundary pixel.
    dist_pred_to_true = distance_transform_edt(1.0 - true_boundary)
    dist_true_to_pred = distance_transform_edt(1.0 - pred_boundary)

    # Count matches within tolerance.
    pred_matched = ((dist_pred_to_true * pred_boundary) <= tolerance).sum()
    true_matched = ((dist_true_to_pred * true_boundary) <= tolerance).sum()

    precision = pred_matched / max(pred_boundary.sum(), 1)
    recall = true_matched / max(true_boundary.sum(), 1)

    if precision + recall < 1e-10:
        return torch.tensor(0.0)

    f1 = 2.0 * precision * recall / (precision + recall)
    return torch.tensor(float(f1))


def hd95_score(
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the 95th percentile Hausdorff distance between prediction and truth.

    Parameters
    ----------
    pred_mask:
        Binary predicted mask.
    true_mask:
        Binary ground-truth mask.

    Returns
    -------
    torch.Tensor
        HD95 distance (lower is better).  Returns inf if either mask is empty.
    """
    pred_np = pred_mask.detach().cpu().squeeze().numpy().astype(bool)
    true_np = true_mask.detach().cpu().squeeze().numpy().astype(bool)

    if not pred_np.any() and not true_np.any():
        return torch.tensor(0.0)
    if not pred_np.any() or not true_np.any():
        return torch.tensor(float("inf"))

    pred_surface = np.logical_xor(pred_np, binary_erosion(pred_np, iterations=1, border_value=0))
    true_surface = np.logical_xor(true_np, binary_erosion(true_np, iterations=1, border_value=0))

    if not pred_surface.any() and not true_surface.any():
        return torch.tensor(0.0)
    if not pred_surface.any() or not true_surface.any():
        return torch.tensor(float("inf"))

    dist_to_true_surface = distance_transform_edt(~true_surface)
    dist_to_pred_surface = distance_transform_edt(~pred_surface)

    d_pred_to_true = dist_to_true_surface[pred_surface]
    d_true_to_pred = dist_to_pred_surface[true_surface]

    hd95 = max(
        np.percentile(d_pred_to_true, 95),
        np.percentile(d_true_to_pred, 95),
    )
    return torch.tensor(float(hd95))
