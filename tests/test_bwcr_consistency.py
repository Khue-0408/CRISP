from __future__ import annotations

import inspect

import pytest
import torch

import crisp.modules.bwcr_consistency as bwcr_module
from crisp.modules.bwcr_consistency import (
    bwcr_consistency_loss,
    native_bwcr_boundary_field,
    native_bwcr_lambda,
)


def test_native_linear_lambda_exact_values() -> None:
    distance = torch.tensor([0.0, 1.0, 5.0, 9.0, 10.0, 12.0])

    observed = native_bwcr_lambda(distance)

    expected = torch.tensor([1.01, 0.91, 0.51, 0.11, 0.01, 0.01])
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=1e-7)


def test_native_rectangle_signed_distance_convention() -> None:
    mask = torch.zeros(1, 1, 9, 9)
    mask[:, :, 2:7, 2:7] = 1.0

    field = native_bwcr_boundary_field(mask)

    assert not field.fallback_applied.item()
    assert field.signed_distance[0, 0, 2, 3].item() == 0.0
    assert field.signed_distance[0, 0, 3, 3].item() == 1.0
    assert field.signed_distance[0, 0, 1, 3].item() == -1.0
    assert field.distance[0, 0, 2, 3].item() == 0.0
    assert field.distance[0, 0, 3, 3].item() == 1.0
    assert field.distance[0, 0, 1, 3].item() == 1.0
    assert torch.all(field.distance >= 0)


def test_empty_mask_uses_exact_minimum_lambda_everywhere() -> None:
    field = native_bwcr_boundary_field(torch.zeros(1, 1, 8, 8))

    assert field.fallback_applied.item()
    torch.testing.assert_close(
        field.lambda_map, torch.full_like(field.lambda_map, 0.01), rtol=0.0, atol=0.0
    )


def test_single_foreground_pixel_uses_tiny_foreground_fallback() -> None:
    mask = torch.zeros(1, 1, 7, 7)
    mask[0, 0, 3, 3] = 1.0

    field = native_bwcr_boundary_field(mask)

    assert field.signed_distance.max().item() == 0.0
    assert field.fallback_applied.item()
    torch.testing.assert_close(
        field.lambda_map, torch.full_like(field.lambda_map, 0.01), rtol=0.0, atol=0.0
    )


def test_thick_object_keeps_native_structure_and_boundary_maximum() -> None:
    mask = torch.zeros(1, 1, 9, 9)
    mask[:, :, 2:7, 2:7] = 1.0

    field = native_bwcr_boundary_field(mask)

    assert not field.fallback_applied.item()
    assert field.distance[0, 0, 2, 4].item() == 0.0
    assert field.distance[0, 0, 4, 4].item() == 2.0
    assert field.lambda_map[0, 0, 2, 4].item() == pytest.approx(1.01)
    assert torch.unique(field.lambda_map).numel() > 1


def test_loss_is_binary_one_logit_raw_squared_consistency() -> None:
    z1 = torch.tensor([[[[-2.0, 3.0]]]])
    z2 = torch.tensor([[[[1.0, -1.0]]]])
    validity = torch.ones_like(z1)
    weights = torch.tensor([[[[0.51, 1.01]]]])

    observed = bwcr_consistency_loss(z1, z2, validity, weights)

    expected = torch.tensor((0.51 * 9.0 + 1.01 * 16.0) / 2.0)
    torch.testing.assert_close(observed, expected)


def test_gradients_reach_both_aligned_raw_logits_with_full_grid_denominator() -> None:
    z1 = torch.tensor([[[[2.0, -1.0]]]], requires_grad=True)
    z2 = torch.tensor([[[[0.0, 3.0]]]], requires_grad=True)
    validity = torch.ones_like(z1)
    weights = torch.ones_like(z1)

    loss = bwcr_consistency_loss(z1, z2, validity, weights)
    loss.backward()

    torch.testing.assert_close(z1.grad, torch.tensor([[[[2.0, -4.0]]]]))
    torch.testing.assert_close(z2.grad, torch.tensor([[[[-2.0, 4.0]]]]))


def test_equal_logits_have_zero_loss() -> None:
    logits = torch.randn(2, 1, 4, 5)
    validity = torch.ones_like(logits)
    weights = torch.full_like(logits, 0.51)

    assert bwcr_consistency_loss(logits, logits.clone(), validity, weights).item() == 0.0


def test_differences_on_invalid_pixels_contribute_zero() -> None:
    z1 = torch.tensor([[[[100.0, -100.0]]]])
    z2 = torch.zeros_like(z1)
    validity = torch.zeros_like(z1)
    weights = torch.full_like(z1, 1.01)

    assert bwcr_consistency_loss(z1, z2, validity, weights).item() == 0.0


def test_invalid_pixels_remain_in_full_grid_mean_denominator() -> None:
    z1 = torch.tensor([[[[2.0, 0.0], [0.0, 0.0]]]])
    z2 = torch.zeros_like(z1)
    validity = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    weights = torch.ones_like(z1)

    observed = bwcr_consistency_loss(z1, z2, validity, weights)

    assert observed.item() == pytest.approx(1.0)


def test_boundary_to_far_contribution_ratio_is_101() -> None:
    logits = torch.ones(1, 1, 1, 1)
    zero = torch.zeros_like(logits)
    validity = torch.ones_like(logits)

    boundary = bwcr_consistency_loss(
        logits, zero, validity, torch.full_like(logits, 1.01)
    )
    far = bwcr_consistency_loss(
        logits, zero, validity, torch.full_like(logits, 0.01)
    )

    assert (boundary / far).item() == pytest.approx(101.0)


def test_boundary_field_is_independent_across_batch_elements() -> None:
    empty = torch.zeros(1, 1, 9, 9)
    rectangle = torch.zeros(1, 1, 9, 9)
    rectangle[:, :, 2:7, 2:7] = 1.0
    batched = torch.cat([empty, rectangle], dim=0)

    together = native_bwcr_boundary_field(batched)
    empty_alone = native_bwcr_boundary_field(empty)
    rectangle_alone = native_bwcr_boundary_field(rectangle)

    torch.testing.assert_close(together.signed_distance[0:1], empty_alone.signed_distance)
    torch.testing.assert_close(together.signed_distance[1:2], rectangle_alone.signed_distance)
    torch.testing.assert_close(together.distance[0:1], empty_alone.distance)
    torch.testing.assert_close(together.distance[1:2], rectangle_alone.distance)
    torch.testing.assert_close(together.lambda_map[0:1], empty_alone.lambda_map)
    torch.testing.assert_close(together.lambda_map[1:2], rectangle_alone.lambda_map)
    assert together.fallback_applied.tolist() == [True, False]


@pytest.mark.parametrize(
    "mask",
    [
        torch.zeros(1, 2, 4, 4),
        torch.zeros(1, 4, 4),
        torch.tensor([[[[0.0, 0.5]]]]),
        torch.tensor([[[[0.0, float("nan")]]]]),
    ],
)
def test_malformed_masks_are_rejected(mask: torch.Tensor) -> None:
    with pytest.raises((TypeError, ValueError)):
        native_bwcr_boundary_field(mask)


def test_non_binary_validity_is_rejected() -> None:
    logits = torch.zeros(1, 1, 2, 2)
    validity = torch.full_like(logits, 0.5)
    weights = torch.ones_like(logits)

    with pytest.raises(ValueError, match="binary"):
        bwcr_consistency_loss(logits, logits, validity, weights)


def test_module_does_not_reuse_crisp_gaussian_boundary_weighting() -> None:
    source = inspect.getsource(bwcr_module)

    assert "compute_boundary_weight" not in source
    assert "gaussian_soft_field" not in source
    assert "sigma_b" not in source
    assert "sigmoid" not in source
