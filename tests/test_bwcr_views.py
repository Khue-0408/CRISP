"""Focused tests for paired BWCR source-view and inverse-alignment infrastructure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from crisp.data.bwcr_views import (
    AppearanceTransformRecord,
    BWCRAugmentationContract,
    BWCRSourceViewPair,
    BWCRViewTransformRecord,
    GeometryTransformRecord,
    apply_bwcr_view_record,
    apply_forward_geometry,
    bwcr_validity_intersection,
    bwcr_view_rng_identity,
    current_crisp_bwcr_contract,
    inverse_align_bwcr_tensor,
    inverse_bwcr_validity,
    sample_bwcr_pair,
    sample_bwcr_view_record,
)


def _current_config() -> dict:
    return {
        "image_size": 352,
        "random_hflip": True,
        "random_vflip": True,
        "random_rotate_degrees": 15,
        "random_scale_range": [0.75, 1.25],
        "color_jitter": {
            "brightness": 0.10,
            "contrast": 0.10,
            "saturation": 0.10,
            "hue": 0.02,
        },
        "random_gaussian_blur": {
            "probability": 0.10,
            "kernel_size": 3,
            "sigma": [0.1, 1.0],
        },
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    }


def _canonical_pattern() -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.linspace(0.0, 1.0, 352)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    image = torch.stack((xx, yy, 0.5 * (xx + yy)))
    mask = (((xx - 0.5).square() + (yy - 0.5).square()) <= 0.16).float().unsqueeze(0)
    return image, mask


def _identity_appearance() -> AppearanceTransformRecord:
    return AppearanceTransformRecord(1.0, 1.0, 1.0, 0.0, False, None)


def _record(geometry: GeometryTransformRecord) -> BWCRViewTransformRecord:
    return BWCRViewTransformRecord("manual", 0, geometry, _identity_appearance())


def test_transform_record_is_deterministic_and_serializable() -> None:
    contract = current_crisp_bwcr_contract(_current_config())
    kwargs = dict(
        experiment_seed=2026,
        epoch=7,
        dataset_name="Kvasir-SEG",
        image_id="sample-17",
        view_index=0,
    )
    first = sample_bwcr_view_record(contract, **kwargs)
    second = sample_bwcr_view_record(contract, **kwargs)
    assert first == second
    assert json.loads(json.dumps(first.to_dict()))["rng_identity"] == first.rng_identity


def test_sampled_record_stays_inside_current_crisp_ranges() -> None:
    contract = current_crisp_bwcr_contract(_current_config())
    record = sample_bwcr_view_record(
        contract,
        experiment_seed=2026,
        epoch=5,
        dataset_name="Kvasir-SEG",
        image_id="range-check",
        view_index=1,
    )
    geometry, appearance = record.geometry, record.appearance
    assert -15.0 <= geometry.rotation_degrees <= 15.0
    assert 0.75 <= geometry.scale <= 1.25
    assert geometry.translation_pixels == (0.0, 0.0)
    assert geometry.shear_degrees == (0.0, 0.0)
    assert 0.9 <= appearance.brightness_factor <= 1.1
    assert 0.9 <= appearance.contrast_factor <= 1.1
    assert 0.9 <= appearance.saturation_factor <= 1.1
    assert -0.02 <= appearance.hue_factor <= 0.02
    if appearance.gaussian_blur:
        assert appearance.gaussian_blur_radius is not None
        assert 0.1 <= appearance.gaussian_blur_radius <= 1.0
    else:
        assert appearance.gaussian_blur_radius is None


def test_stored_record_replays_view_without_rng_state() -> None:
    contract = current_crisp_bwcr_contract(_current_config())
    image, mask = _canonical_pattern()
    record = sample_bwcr_view_record(
        contract,
        experiment_seed=2030,
        epoch=9,
        dataset_name="CVC-ClinicDB",
        image_id="replay",
        view_index=0,
    )
    first = apply_bwcr_view_record(image, mask, record, contract)
    torch.manual_seed(999999)
    second = apply_bwcr_view_record(image, mask, record, contract)
    assert torch.equal(first.image, second.image)
    assert torch.equal(first.mask, second.mask)
    assert torch.equal(first.forward_validity, second.forward_validity)


def test_view_zero_rng_is_isolated_from_pair_sampling() -> None:
    contract = current_crisp_bwcr_contract(_current_config())
    image, mask = _canonical_pattern()
    alone = sample_bwcr_view_record(
        contract,
        experiment_seed=2027,
        epoch=2,
        dataset_name="CVC-ClinicDB",
        image_id="42",
        view_index=0,
    )
    pair = sample_bwcr_pair(
        image,
        mask,
        contract,
        experiment_seed=2027,
        epoch=2,
        dataset_name="CVC-ClinicDB",
        image_id="42",
    )
    assert pair.view_0.record == alone


def test_record_identity_is_sample_order_invariant() -> None:
    contract = current_crisp_bwcr_contract(_current_config())

    def record(image_id: str) -> BWCRViewTransformRecord:
        return sample_bwcr_view_record(
            contract,
            experiment_seed=2028,
            epoch=4,
            dataset_name="Kvasir-SEG",
            image_id=image_id,
            view_index=1,
        )

    a_then_b = {"A": record("A"), "B": record("B")}
    b_then_a = {"B": record("B"), "A": record("A")}
    assert a_then_b == b_then_a


def test_epoch_and_view_index_change_rng_identity() -> None:
    common = dict(experiment_seed=2029, dataset_name="Kvasir-SEG", image_id="same")
    epoch_0 = bwcr_view_rng_identity(epoch=0, view_index=0, **common)
    epoch_1 = bwcr_view_rng_identity(epoch=1, view_index=0, **common)
    view_1 = bwcr_view_rng_identity(epoch=0, view_index=1, **common)
    assert epoch_0 != epoch_1
    assert epoch_0 != view_1


@pytest.mark.parametrize(
    ("horizontal", "vertical"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_discrete_geometry_forward_then_inverse_is_exact(
    horizontal: bool, vertical: bool
) -> None:
    pattern = torch.arange(25, dtype=torch.float32).reshape(1, 5, 5)
    geometry = GeometryTransformRecord(horizontal, vertical, 0.0, 1.0)
    view = apply_forward_geometry(pattern, geometry, interpolation="nearest")
    restored = inverse_align_bwcr_tensor(view, geometry, interpolation="nearest")
    assert torch.equal(restored, pattern)


def test_affine_inverse_alignment_restores_orientation_with_tight_interior_error() -> None:
    axis = torch.linspace(-1.0, 1.0, 352)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    pattern = torch.exp(-40.0 * ((xx + 0.24).square() + (yy - 0.17).square())).unsqueeze(0)
    geometry = GeometryTransformRecord(False, False, 12.0, 0.9)
    view = apply_forward_geometry(pattern, geometry, interpolation="bilinear")
    restored = inverse_align_bwcr_tensor(view, geometry, interpolation="bilinear")
    validity = inverse_align_bwcr_tensor(
        apply_forward_geometry(torch.ones_like(pattern), geometry, interpolation="nearest"),
        geometry,
        interpolation="nearest",
    )
    interior = validity.bool() & (pattern > 1e-4)
    assert restored.shape == pattern.shape
    assert restored[0].argmax() == pattern[0].argmax()
    assert (restored[interior] - pattern[interior]).abs().mean() < 0.003


def test_forward_mask_and_inverse_mask_remain_binary() -> None:
    image, mask = _canonical_pattern()
    contract = BWCRAugmentationContract()
    geometry = GeometryTransformRecord(True, False, 13.0, 0.82)
    view = apply_bwcr_view_record(image, mask, _record(geometry), contract)
    restored = inverse_align_bwcr_tensor(view.mask, geometry, interpolation="nearest")
    assert set(view.mask.unique().tolist()).issubset({0.0, 1.0})
    assert set(restored.unique().tolist()).issubset({0.0, 1.0})


def test_inverse_validity_and_pair_intersection_are_binary_and_conservative() -> None:
    image, mask = _canonical_pattern()
    contract = BWCRAugmentationContract()
    record_0 = _record(GeometryTransformRecord(False, False, 15.0, 0.78))
    record_1 = _record(GeometryTransformRecord(True, False, -11.0, 0.84))
    view_0 = apply_bwcr_view_record(image, mask, record_0, contract)
    view_1 = apply_bwcr_view_record(image, mask, record_1, contract)
    pair = BWCRSourceViewPair(view_0, view_1)
    validity_0 = inverse_bwcr_validity(view_0)
    validity_1 = inverse_bwcr_validity(view_1)
    intersection = bwcr_validity_intersection(pair)
    assert (validity_0 == 0).any() and (validity_0 == 1).any()
    assert set(intersection.unique().tolist()).issubset({0.0, 1.0})
    assert torch.all(intersection <= validity_0)
    assert torch.all(intersection <= validity_1)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("random_rotate_degrees",), 20),
        (("random_scale_range",), [0.5, 1.5]),
        (("color_jitter", "hue"), 0.05),
        (("random_gaussian_blur", "probability"), 0.5),
        (("random_gaussian_blur", "sigma"), [0.5, 2.0]),
    ],
)
def test_current_crisp_range_lock_rejects_silent_substitution(
    path: tuple[str, ...], replacement: object
) -> None:
    config = _current_config()
    target = config
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment,index]
    target[path[-1]] = replacement  # type: ignore[index]
    with pytest.raises(ValueError, match="current CRISP contract"):
        current_crisp_bwcr_contract(config)


def test_checked_in_source_augmentation_config_matches_range_lock() -> None:
    text = Path("configs/data/crisp_train_test.yaml").read_text(encoding="utf-8")
    expected_lines = {
        "image_size: 352",
        "random_hflip: true",
        "random_vflip: true",
        "random_rotate_degrees: 15",
        "random_scale_range: [0.75, 1.25]",
        "  brightness: 0.10",
        "  contrast: 0.10",
        "  saturation: 0.10",
        "  hue: 0.02",
        "  probability: 0.10",
        "  sigma: [0.1, 1.0]",
    }
    assert expected_lines.issubset(set(text.splitlines()))


def test_baseline_transform_source_remains_unmodified_by_paired_rng() -> None:
    source = Path("src/crisp/data/transforms.py").read_text(encoding="utf-8")
    assert "bwcr" not in source.lower()
    assert "torch.rand(1).item() > 0.5" in source
    assert "_sample_uniform(-self.random_rotate_degrees" in source
    assert "_sample_uniform(scale_min, scale_max)" in source
