"""
Unit tests for boundary extraction and boundary weighting.

These tests validate:
- output shapes,
- value ranges,
- expected monotonic decay away from the boundary,
- stability under simple synthetic masks.
"""

import torch
from crisp.modules.boundary import (
    compute_boundary_weight,
    compute_distance_to_boundary,
    extract_binary_boundary,
)


def test_boundary_weight_shape() -> None:
    """Ensure the boundary weighting field preserves batch and spatial shape."""
    mask = torch.zeros(2, 1, 32, 32)
    mask[:, :, 8:24, 8:24] = 1.0  # centered square

    wb = compute_boundary_weight(mask, sigma_b=3.0)
    assert wb.shape == (2, 1, 32, 32), f"Shape mismatch: {wb.shape}"


def test_boundary_weight_range() -> None:
    """Boundary weights must be in [0, 1]."""
    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, 16:48, 16:48] = 1.0

    wb = compute_boundary_weight(mask, sigma_b=3.0)
    assert wb.min() >= 0.0, f"Min below 0: {wb.min()}"
    assert wb.max() <= 1.0 + 1e-6, f"Max above 1: {wb.max()}"


def test_boundary_peaks_at_boundary() -> None:
    """Boundary weight should be highest near the mask boundary."""
    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, 16:48, 16:48] = 1.0

    wb = compute_boundary_weight(mask, sigma_b=3.0)

    # Center pixel (32, 32) should have lower weight than boundary pixel (16, 16).
    center_val = wb[0, 0, 32, 32].item()
    boundary_val = wb[0, 0, 16, 16].item()
    assert boundary_val > center_val, (
        f"Boundary pixel ({boundary_val}) should be > center pixel ({center_val})"
    )


def test_extract_binary_boundary_shape() -> None:
    """Binary boundary extraction should preserve shape."""
    mask = torch.zeros(2, 1, 32, 32)
    mask[:, :, 10:22, 10:22] = 1.0
    boundary = extract_binary_boundary(mask)
    assert boundary.shape == mask.shape


def test_distance_map_non_negative() -> None:
    """Distance to boundary should be non-negative."""
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 8:24, 8:24] = 1.0
    dist = compute_distance_to_boundary(mask)
    assert dist.min() >= 0.0


def test_hard_band_mode() -> None:
    """Hard band ablation should produce binary weights."""
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 8:24, 8:24] = 1.0
    wb = compute_boundary_weight(mask, sigma_b=3.0, mode="hard_band")
    unique_vals = wb.unique()
    assert all(v in (0.0, 1.0) for v in unique_vals.tolist())
