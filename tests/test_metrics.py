"""
Unit tests for segmentation and calibration metrics.
"""

import pytest
import torch
import torch.nn as nn
from scipy.ndimage import binary_erosion, distance_transform_edt

from crisp.engine.evaluator import Evaluator
from crisp.metrics.calibration import (
    brier_score,
    boundary_area_weighted_ece,
    expected_calibration_error,
    negative_log_likelihood,
    boundary_support_mask,
    off_boundary_expected_calibration_error,
)
from crisp.metrics.segmentation import dice_score, boundary_f1_score, hd95_score
from crisp.models.base import SegmentationOutput


def _square(top: int, left: int, size: int = 4, canvas: int = 32) -> torch.Tensor:
    mask = torch.zeros(canvas, canvas)
    mask[top : top + size, left : left + size] = 1.0
    return mask


def test_boundary_f1_identical_nonempty_mask() -> None:
    mask = _square(8, 8)
    assert boundary_f1_score(mask, mask).item() == pytest.approx(1.0)


def test_boundary_f1_both_empty() -> None:
    empty = torch.zeros(32, 32)
    assert boundary_f1_score(empty, empty).item() == 1.0


def test_boundary_f1_one_empty_in_both_orders() -> None:
    empty = torch.zeros(32, 32)
    nonempty = _square(8, 8)
    assert boundary_f1_score(empty, nonempty).item() == 0.0
    assert boundary_f1_score(nonempty, empty).item() == 0.0


def test_boundary_f1_shift_at_five_pixel_tolerance() -> None:
    original = _square(120, 120, canvas=352)
    shifted = _square(120, 125, canvas=352)
    assert boundary_f1_score(original, shifted).item() == pytest.approx(1.0)
    assert boundary_f1_score(original, shifted, tolerance=4).item() < 1.0


def test_boundary_f1_shift_beyond_tolerance_decreases_score() -> None:
    original = _square(8, 8)
    within = _square(8, 13)
    beyond = _square(8, 14)
    assert boundary_f1_score(original, beyond).item() < boundary_f1_score(original, within).item()


def test_boundary_f1_disjoint_far_masks_have_no_matches() -> None:
    assert boundary_f1_score(_square(2, 2), _square(24, 24)).item() == 0.0


def test_boundary_f1_is_bounded_across_representative_cases() -> None:
    empty = torch.zeros(32, 32)
    masks = (empty, _square(2, 2), _square(8, 13), _square(24, 24))
    for pred in masks:
        for truth in masks:
            assert 0.0 <= boundary_f1_score(pred, truth).item() <= 1.0


def test_boundary_f1_scientific_regression_nonboundary_pixels_are_not_matches() -> None:
    mask = _square(8, 8)
    boundary = mask.numpy().astype(bool) ^ binary_erosion(mask.numpy().astype(bool))
    distance = distance_transform_edt(~boundary)
    historical_count = ((distance * boundary) <= 5).sum()
    assert historical_count > boundary.sum()  # The old expression counted background pixels.
    assert historical_count / boundary.sum() > 1.0
    assert boundary_f1_score(mask, mask).item() == pytest.approx(1.0)


def test_boundary_f1_edge_touching_mask_is_finite_and_bounded() -> None:
    edge_mask = _square(0, 0)
    score = boundary_f1_score(edge_mask, _square(0, 5))
    assert torch.isfinite(score)
    assert 0.0 <= score.item() <= 1.0
    assert boundary_f1_score(edge_mask, edge_mask).item() == pytest.approx(1.0)


def test_boundary_f1_is_symmetric() -> None:
    first = _square(8, 8)
    second = _square(8, 14)
    assert boundary_f1_score(first, second).item() == pytest.approx(
        boundary_f1_score(second, first).item()
    )


def test_boundary_f1_dataset_aggregation_is_mean_of_image_scores() -> None:
    class LogitsFromImage(nn.Module):
        def forward(self, images: torch.Tensor) -> SegmentationOutput:
            logits = images[:, :1]
            return SegmentationOutput(logits=logits, features=logits)

    truth = torch.stack((_square(8, 8), _square(2, 2, size=10))).unsqueeze(1)
    predicted = torch.stack((_square(8, 8), _square(20, 20, size=10))).unsqueeze(1)
    images = torch.zeros(2, 3, 32, 32)
    images[:, :1] = predicted * 20.0 - 10.0
    evaluator = Evaluator(model=LogitsFromImage(), projector=None, config={})
    score = evaluator.evaluate_dataset([{"image": images, "mask": truth}], "two_images", projector_on=False)[
        "boundary_f1"
    ]
    expected = (boundary_f1_score(predicted[0], truth[0]).item() +
                boundary_f1_score(predicted[1], truth[1]).item()) / 2.0
    assert score == pytest.approx(expected)
    assert score == pytest.approx(0.5)


def test_ece_non_negative() -> None:
    """Ensure expected calibration error is non-negative."""
    probs = torch.rand(1, 1, 16, 16)
    labels = (torch.rand(1, 1, 16, 16) > 0.5).float()
    ece = expected_calibration_error(probs, labels)
    assert ece.item() >= 0.0


def test_ece_perfect_calibration() -> None:
    """ECE should be near zero for perfectly calibrated predictions."""
    # All predictions at 0.5, 50% of labels are 1 → well-calibrated on average.
    probs = torch.full((1, 1, 100, 100), 0.5)
    labels = torch.zeros(1, 1, 100, 100)
    labels[:, :, :50, :] = 1.0
    ece = expected_calibration_error(probs, labels)
    assert ece.item() < 0.1, f"ECE too high for balanced predictions: {ece.item()}"


def test_brier_score_range() -> None:
    """Brier score should be in [0, 1]."""
    probs = torch.rand(1, 1, 16, 16)
    labels = (torch.rand(1, 1, 16, 16) > 0.5).float()
    bs = brier_score(probs, labels)
    assert 0.0 <= bs.item() <= 1.0


def test_nll_non_negative() -> None:
    """NLL should be non-negative."""
    probs = torch.rand(1, 1, 16, 16).clamp(0.01, 0.99)
    labels = (torch.rand(1, 1, 16, 16) > 0.5).float()
    nll = negative_log_likelihood(probs, labels)
    assert nll.item() >= 0.0


def test_dice_score_perfect() -> None:
    """Dice should be ~1.0 for perfect overlap."""
    mask = torch.ones(1, 1, 16, 16)
    score = dice_score(mask, mask)
    assert score.item() > 0.99


def test_dice_score_no_overlap() -> None:
    """Dice should be near 0 for no overlap."""
    pred = torch.ones(1, 1, 16, 16)
    true = torch.zeros(1, 1, 16, 16)
    score = dice_score(pred, true)
    assert score.item() < 0.01


def test_boundary_support_mask_proportion() -> None:
    """Boundary support mask should select approximately top_percent pixels."""
    wb = torch.rand(2, 1, 32, 32)
    support = boundary_support_mask(wb, top_percent=20.0)
    prop = support.mean().item()
    # Should be approximately 20% (give or take ties/rounding).
    assert 0.15 <= prop <= 0.30, f"Support proportion out of range: {prop}"


def test_boundary_support_mask_is_per_image_topk() -> None:
    """
    bECE support must be selected per-image (top-k within each image),
    not via a global threshold over the batch.
    """
    # Construct a batch where image 0 has small weights and image 1 has large weights.
    B, H, W = 2, 4, 5
    wb0 = torch.linspace(0.0, 0.1, H * W).reshape(1, 1, H, W)
    wb1 = torch.linspace(0.9, 1.0, H * W).reshape(1, 1, H, W)
    wb = torch.cat([wb0, wb1], dim=0)

    top_percent = 20.0
    support = boundary_support_mask(wb, top_percent=top_percent)
    k = max(1, int(H * W * top_percent / 100.0))
    # Each image should select exactly k pixels (no ties in constructed weights).
    assert int(support[0].sum().item()) == k
    assert int(support[1].sum().item()) == k


def test_ba_ece_avoids_zero_mass_nan() -> None:
    """BA-ECE should not produce NaNs even if a bin has near-zero boundary mass."""
    probs = torch.full((1, 1, 10, 10), 0.2)
    labels = torch.zeros_like(probs)
    wb = torch.zeros_like(probs)  # zero boundary mass everywhere
    v = boundary_area_weighted_ece(probs, labels, wb)
    assert torch.isfinite(v), "BA-ECE should be finite"


def test_off_boundary_ece_complements_boundary_support() -> None:
    """Off-boundary ECE should run and be finite."""
    probs = torch.rand(1, 1, 16, 16)
    labels = (torch.rand(1, 1, 16, 16) > 0.5).float()
    wb = torch.rand(1, 1, 16, 16)
    v = off_boundary_expected_calibration_error(probs, labels, wb, top_percent=20.0)
    assert torch.isfinite(v), "off-boundary ECE should be finite"


def test_hd95_perfect() -> None:
    """HD95 should be 0 for identical masks."""
    mask = torch.zeros(16, 16)
    mask[4:12, 4:12] = 1.0
    hd = hd95_score(mask, mask)
    assert hd.item() < 1e-6


def test_hd95_shifted_square_uses_surface_distance() -> None:
    """A one-pixel shift should yield HD95 close to one pixel, not zero from region overlap."""
    pred = torch.zeros(16, 16)
    true = torch.zeros(16, 16)
    pred[4:12, 5:13] = 1.0
    true[4:12, 4:12] = 1.0
    hd = hd95_score(pred, true)
    assert abs(hd.item() - 1.0) < 1e-6
