"""
Boundary-related utilities for CRISP.

This module is responsible for constructing the soft boundary weighting field
w_b(u) from the annotation boundary. The paper defines:

w_b(u) = exp(-d(u, ∂y)^2 / (2 * sigma_b^2)),

where d(u, ∂y) is the Euclidean distance from pixel u to the mask boundary. [file:1]

The soft field is central because CRISP localizes calibration intervention near
the boundary rather than across the entire image. [file:1]
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def extract_binary_boundary(mask: torch.Tensor) -> torch.Tensor:
    """
    Compute a binary boundary map from a binary segmentation mask.

    Uses morphological erosion via max-pooling to detect the mask boundary:
    boundary = mask XOR eroded(mask).

    Parameters
    ----------
    mask:
        Binary mask tensor of shape [B, 1, H, W].

    Returns
    -------
    torch.Tensor
        Binary boundary tensor of shape [B, 1, H, W].
    """
    assert mask.ndim == 4 and mask.shape[1] == 1, (
        f"Expected [B,1,H,W], got {mask.shape}"
    )
    # Erosion: max_pool2d on (1 - mask) with kernel 3, then invert.
    # Equivalently, eroded = 1 - max_pool2d(1 - mask, 3, stride=1, padding=1).
    eroded = 1.0 - F.max_pool2d(
        1.0 - mask.float(), kernel_size=3, stride=1, padding=1
    )
    boundary = ((mask.float() - eroded).abs() > 0.5).float()
    return boundary


def compute_distance_to_boundary(mask: torch.Tensor) -> torch.Tensor:
    """
    Compute per-pixel Euclidean distance to the nearest boundary pixel.

    Parameters
    ----------
    mask:
        Binary segmentation mask of shape [B, 1, H, W].

    Returns
    -------
    torch.Tensor
        Distance map of shape [B, 1, H, W], dtype float32.

    Notes
    -----
    Uses ``scipy.ndimage.distance_transform_edt`` per sample for stable,
    deterministic Euclidean distance computation.  Distance is computed to
    the mask boundary, not to foreground/background regions generally.
    """
    assert mask.ndim == 4 and mask.shape[1] == 1
    boundary = extract_binary_boundary(mask)
    boundary_np = boundary.squeeze(1).cpu().numpy()  # [B, H, W]

    distances = []
    for b in range(boundary_np.shape[0]):
        bnd = boundary_np[b]  # [H, W]
        if bnd.sum() == 0:
            # No boundary → all pixels are far from boundary (max distance).
            dist = np.full_like(bnd, fill_value=max(bnd.shape), dtype=np.float32)
        else:
            # EDT from non-boundary pixels to boundary pixels.
            # distance_transform_edt computes distance to nearest zero pixel.
            dist = distance_transform_edt(1.0 - bnd).astype(np.float32)
        distances.append(dist)

    dist_tensor = torch.from_numpy(np.stack(distances, axis=0)).unsqueeze(1)  # [B,1,H,W]
    return dist_tensor.to(mask.device)


def compute_boundary_weight(
    mask: torch.Tensor,
    sigma_b: float = 3.0,
    mode: Literal["gaussian_soft_field", "hard_band", "logistic_ramp"] = "gaussian_soft_field",
) -> torch.Tensor:
    """
    Compute the CRISP boundary weighting field.

    Parameters
    ----------
    mask:
        Binary segmentation mask [B, 1, H, W].
    sigma_b:
        Spread parameter controlling how far the boundary influence extends.
        Default sigma_b = 3.0 per instruct.md §3.4.
    mode:
        Boundary weighting mode. The default and paper-supported mode is a
        Gaussian soft field. Alternative modes are included for ablations.

    Returns
    -------
    torch.Tensor
        Boundary weighting field w_b with values in [0, 1], shape [B,1,H,W].

    CRISP reference
    ---------------
    instruct.md §3.1: w_b(u) = exp(-d(u,∂y)² / (2 σ_b²))
    """
    dist = compute_distance_to_boundary(mask)  # [B,1,H,W]

    if mode == "gaussian_soft_field":
        # Paper default: Gaussian decay from boundary.
        wb = torch.exp(-dist.pow(2) / (2.0 * sigma_b ** 2))
    elif mode == "hard_band":
        # Ablation: binary band within sigma_b pixels of boundary.
        wb = (dist <= sigma_b).float()
    elif mode == "logistic_ramp":
        # Ablation: smooth sigmoid transition at sigma_b.
        wb = torch.sigmoid((sigma_b - dist) / max(sigma_b * 0.3, 1e-6))
    else:
        raise ValueError(f"Unknown boundary mode: {mode}")

    return wb
