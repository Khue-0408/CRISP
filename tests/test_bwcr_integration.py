"""Integration contract for the source-only BWCR calibration control."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn

import crisp.engine.trainer as trainer_module
from crisp.data.bwcr_views import (
    AppearanceTransformRecord,
    BWCRSourceView,
    BWCRSourceViewPair,
    BWCRViewTransformRecord,
    GeometryTransformRecord,
)
from crisp.data.datasets import BinarySegmentationDataset, SampleRecord
from crisp.engine.trainer import Trainer, prepare_bwcr_training_batch
from crisp.models.base import SegmentationOutput
from crisp.models.pranet import PraNet
from crisp.modules.bwcr_consistency import bwcr_consistency_loss
from crisp.modules.bwcr_control import CONTROL_NAME, resolve_bwcr_control
from crisp.modules.losses import baseline_bce_dice_loss, pranet_native_side_losses
from crisp.registry import build_dataset
from crisp.utils.provenance import run_provenance


ROOT = Path(__file__).resolve().parents[1]


def _source_contract() -> dict:
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
            "sigma": [0.1, 1.0],
        },
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    }


def _control_config() -> dict:
    return {
        "seed": 2026,
        "training": {
            "epochs": 2,
            "optimizer": "adamw",
            "scheduler": "none",
            "mixed_precision": False,
            "phases": {
                "baseline_warmup": 25,
                "crisp_full": 65,
                "finetune": 30,
                "phase2_ramp_epochs": 10,
            },
        },
        "source_data": _source_contract(),
        "method": {
            "name": CONTROL_NAME,
            "use_crisp": False,
            "use_projector": False,
            "use_teachers": False,
        },
        "calibration_control": {
            "name": CONTROL_NAME,
            "lambda_min": 0.01,
            "lambda_max": 1.0,
            "radius": 10.0,
            "alpha": 1.0,
            "boundary_weighting": "native_linear",
            "view_geometry": "independent_inverse_aligned",
            "consistency_signal": "raw_final_logit",
        },
    }


def _canonical_batch(ids: tuple[str, ...] = ("a",)) -> dict:
    axis = torch.linspace(0.0, 1.0, 352)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    base_image = torch.stack((xx, yy, 0.5 * (xx + yy)))
    base_mask = (
        ((xx - 0.5).square() + (yy - 0.5).square()) <= 0.12
    ).float().unsqueeze(0)
    images = torch.stack([base_image.roll(index, dims=2) for index in range(len(ids))])
    masks = torch.stack([base_mask.roll(index, dims=2) for index in range(len(ids))])
    return {
        "canonical_image": images,
        "canonical_mask": masks,
        "meta": {
            "dataset_name": ["Kvasir-SEG"] * len(ids),
            "image_id": list(ids),
            "split": ["train"] * len(ids),
            "source_representation": ["canonical_pre_stochastic"] * len(ids),
        },
    }


class _TinyStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Conv2d(3, 1, 1)

    def forward(self, images: torch.Tensor) -> SegmentationOutput:
        logits = self.logits(images)
        return SegmentationOutput(logits=logits, features=images[:, :1])


class _TwoPathStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.view1_scale = nn.Parameter(torch.tensor(1.0))
        self.view2_scale = nn.Parameter(torch.tensor(2.0))
        self.calls = 0

    def forward(self, images: torch.Tensor) -> SegmentationOutput:
        scale = self.view1_scale if self.calls == 0 else self.view2_scale
        self.calls += 1
        logits = scale * images[:, :1]
        return SegmentationOutput(logits=logits, features=images[:, :1])


class _TinyPraNet(PraNet):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.encoder = nn.Conv2d(3, 4, 1)
        self.final = nn.Conv2d(4, 1, 1)
        self.side5 = nn.Conv2d(4, 1, 1)
        self.side4 = nn.Conv2d(4, 1, 1)
        self.side3 = nn.Conv2d(4, 1, 1)
        self.side_shift = 0.0

    def forward(self, images: torch.Tensor) -> SegmentationOutput:
        features = torch.tanh(self.encoder(images))
        logits = self.final(features)
        return SegmentationOutput(
            logits=logits,
            features=features,
            aux={
                "lateral_map_5": self.side5(features) + self.side_shift,
                "lateral_map_4": self.side4(features),
                "lateral_map_3": self.side3(features),
                "lateral_map_2": logits,
            },
        )


@pytest.mark.parametrize(
    "flag",
    [
        "use_crisp",
        "use_projector",
        "use_teachers",
        "use_amortization_loss",
        "use_boundary_weighted_task",
        "use_identity_regularization",
        "allow_self_ensemble_teacher",
    ],
)
def test_resolver_rejects_crisp_only_machinery(flag: str) -> None:
    config = _control_config()
    config["method"][flag] = True
    with pytest.raises(ValueError, match=flag):
        resolve_bwcr_control(config)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("lambda_min", 0.02),
        ("lambda_max", 0.9),
        ("radius", 9.0),
        ("alpha", 2.0),
        ("boundary_weighting", "gaussian_soft_field"),
        ("view_geometry", "shared"),
        ("consistency_signal", "probability"),
    ],
)
def test_resolver_rejects_noncanonical_control_identity(key: str, value: object) -> None:
    config = _control_config()
    config["calibration_control"][key] = value
    with pytest.raises(ValueError, match=key):
        resolve_bwcr_control(config)


@pytest.mark.parametrize("component", ["projector", "teacher"])
def test_trainer_rejects_constructed_crisp_component(component: str) -> None:
    extra = nn.Identity()
    projector = extra if component == "projector" else None
    teacher = extra if component == "teacher" else None
    with pytest.raises(ValueError, match="cannot receive"):
        Trainer(_TinyStudent(), projector, teacher, _control_config())


def test_bwcr_dataset_mode_returns_pre_stochastic_canonical_pair(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(np.full((19, 23, 3), 127, dtype=np.uint8)).save(image_path)
    mask = np.zeros((19, 23), dtype=np.uint8)
    mask[4:15, 6:17] = 255
    Image.fromarray(mask).save(mask_path)
    dataset = BinarySegmentationDataset(
        [SampleRecord(image_path, mask_path, "sample", "source", "train")],
        transforms=None,
        canonical_source_size=(352, 352),
    )

    sample = dataset[0]

    assert set(sample) == {"canonical_image", "canonical_mask", "meta"}
    assert sample["canonical_image"].shape == (3, 352, 352)
    assert sample["canonical_mask"].shape == (1, 352, 352)
    assert 0.0 <= sample["canonical_image"].min() <= sample["canonical_image"].max() <= 1.0
    assert set(sample["canonical_mask"].unique().tolist()) <= {0.0, 1.0}
    assert sample["meta"]["source_representation"] == "canonical_pre_stochastic"


def test_registry_bypasses_ordinary_stochastic_transform_for_bwcr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def forbidden(_config):
        raise AssertionError("ordinary stochastic train transform was constructed")

    def fake_builder(data_cfg, split, transforms, canonical_source_size=None):
        captured.update(
            split=split,
            transforms=transforms,
            canonical_source_size=canonical_source_size,
        )
        return object()

    transforms = ModuleType("crisp.data.transforms")
    transforms.build_train_transforms = forbidden
    transforms.build_eval_transforms = lambda _config: None
    monkeypatch.setitem(sys.modules, "crisp.data.transforms", transforms)
    monkeypatch.setattr(
        "crisp.data.datasets.build_manifest_train_val_dataset", fake_builder
    )
    config = _control_config()
    config["source_data"]["source_split"] = {"mode": "manifest"}

    build_dataset(config, "train")

    assert captured == {
        "split": "train",
        "transforms": None,
        "canonical_source_size": (352, 352),
    }


def test_unet_control_is_exact_view1_host_plus_unweighted_bwcr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyStudent()
    trainer = Trainer(model, None, None, _control_config())

    def forbidden(*_args, **_kwargs):
        raise AssertionError("CRISP Gaussian boundary path was entered")

    monkeypatch.setattr(trainer_module, "compute_boundary_weight", forbidden)

    step = trainer.train_one_step(_canonical_batch(), epoch=0, step=0)
    tensors = step.tensors
    assert tensors is not None
    expected_host = baseline_bce_dice_loss(
        model(tensors["bwcr_view1_image"]).logits,
        tensors["bwcr_view1_mask"],
    )["loss"]
    expected_bwcr = bwcr_consistency_loss(
        tensors["bwcr_z1_inverse"],
        tensors["bwcr_z2_inverse"],
        tensors["bwcr_validity"],
        tensors["bwcr_lambda_map"],
    )

    torch.testing.assert_close(tensors["supervised_host_loss"], expected_host)
    torch.testing.assert_close(step.loss, expected_host + expected_bwcr)
    assert step.logs["bwcr_consistency_loss"] == pytest.approx(expected_bwcr.item())
    assert not any(key.startswith("schedule/") for key in step.logs)
    assert trainer.projector is None and trainer.teacher_ensemble is None


def test_runtime_bwcr_gradient_reaches_both_live_view_forwards() -> None:
    model = _TwoPathStudent()
    trainer = Trainer(model, None, None, _control_config())

    step = trainer.train_one_step(_canonical_batch(), epoch=1, step=0)
    assert step.tensors is not None
    consistency = step.tensors["bwcr_consistency_loss"]
    gradients = torch.autograd.grad(
        consistency, (model.view1_scale, model.view2_scale), retain_graph=True
    )

    assert model.calls == 2
    assert all(gradient.abs().item() > 0.0 for gradient in gradients)


def test_pranet_preserves_view1_native_aux_and_uses_final_logit_only() -> None:
    model = _TinyPraNet()
    trainer = Trainer(model, None, None, _control_config())
    batch = _canonical_batch()

    first = trainer.train_one_step(batch, epoch=4, step=0)
    assert first.tensors is not None
    with torch.no_grad():
        output = model(first.tensors["bwcr_view1_image"])
        final = baseline_bce_dice_loss(
            output.logits, first.tensors["bwcr_view1_mask"]
        )["loss"]
        native_aux = pranet_native_side_losses(
            output.aux, first.tensors["bwcr_view1_mask"]
        )["native_aux_loss"]
    consistency = first.tensors["bwcr_consistency_loss"]
    torch.testing.assert_close(first.loss, final + native_aux + consistency)

    model.side_shift = 2.0
    second = trainer.train_one_step(batch, epoch=4, step=0)
    assert second.logs["native_aux_loss"] != pytest.approx(first.logs["native_aux_loss"])
    assert second.logs["bwcr_consistency_loss"] == pytest.approx(
        first.logs["bwcr_consistency_loss"]
    )
    assert second.logs["final_loss"] == pytest.approx(first.logs["final_loss"])


def test_runtime_uses_canonical_gt_independently_of_view_geometry() -> None:
    trainer = Trainer(_TinyStudent(), None, None, _control_config())
    batch = _canonical_batch()

    first = trainer.train_one_step(batch, epoch=1, step=0)
    second = trainer.train_one_step(batch, epoch=2, step=0)
    assert first.tensors is not None and second.tensors is not None

    assert not torch.equal(
        first.tensors["bwcr_view1_mask"], second.tensors["bwcr_view1_mask"]
    )
    torch.testing.assert_close(
        first.tensors["bwcr_lambda_map"], second.tensors["bwcr_lambda_map"]
    )


def _manual_pair(
    canonical_image: torch.Tensor, canonical_mask: torch.Tensor
) -> BWCRSourceViewPair:
    identity = GeometryTransformRecord(False, False, 0.0, 1.0)
    appearance = AppearanceTransformRecord(1.0, 1.0, 1.0, 0.0, False, None)
    first_record = BWCRViewTransformRecord("first", 1, identity, appearance)
    second_record = BWCRViewTransformRecord("second", 2, identity, appearance)
    first_validity = torch.ones_like(canonical_mask)
    second_validity = torch.ones_like(canonical_mask)
    second_validity[:, :40, :40] = 0.0
    second_image = canonical_image.clone()
    second_image[:, :40, :40] += 1.0
    return BWCRSourceViewPair(
        BWCRSourceView(canonical_image, canonical_mask, first_validity, first_record),
        BWCRSourceView(second_image, canonical_mask, second_validity, second_record),
    )


def test_runtime_validity_intersection_zeros_invalid_logit_differences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trainer_module,
        "sample_bwcr_pair",
        lambda image, mask, _contract, **_kwargs: _manual_pair(image, mask),
    )
    trainer = Trainer(_TinyStudent(), None, None, _control_config())
    with torch.no_grad():
        trainer.model.logits.weight.zero_()
        trainer.model.logits.weight[0, 0, 0, 0] = 1.0
        trainer.model.logits.bias.zero_()

    step = trainer.train_one_step(_canonical_batch(), epoch=0, step=0)
    assert step.tensors is not None

    assert torch.all(step.tensors["bwcr_validity"][:, :, :40, :40] == 0)
    assert step.logs["bwcr_consistency_loss"] == pytest.approx(0.0, abs=1e-12)


def test_runtime_pair_identity_is_repeatable_and_batch_order_independent() -> None:
    control = resolve_bwcr_control(_control_config())
    assert control is not None
    original = _canonical_batch(("a", "b"))
    repeated = prepare_bwcr_training_batch(
        original,
        contract=control.augmentation,
        experiment_seed=2026,
        epoch=7,
        device=torch.device("cpu"),
    )
    again = prepare_bwcr_training_batch(
        original,
        contract=control.augmentation,
        experiment_seed=2026,
        epoch=7,
        device=torch.device("cpu"),
    )
    assert [pair.view_0.record for pair in repeated.pairs] == [
        pair.view_0.record for pair in again.pairs
    ]
    torch.testing.assert_close(repeated.view1_images, again.view1_images)

    reversed_batch = {
        "canonical_image": original["canonical_image"].flip(0),
        "canonical_mask": original["canonical_mask"].flip(0),
        "meta": {key: list(reversed(value)) for key, value in original["meta"].items()},
    }
    reversed_views = prepare_bwcr_training_batch(
        reversed_batch,
        contract=control.augmentation,
        experiment_seed=2026,
        epoch=7,
        device=torch.device("cpu"),
    )
    assert repeated.pairs[0].view_0.record == reversed_views.pairs[1].view_0.record
    assert repeated.pairs[1].view_1.record == reversed_views.pairs[0].view_1.record


@pytest.mark.parametrize("host", ["unet", "pranet"])
def test_bwcr_config_is_only_approved_delta_from_matched_baseline(host: str) -> None:
    experiment_dir = ROOT / "configs" / "experiment"
    baseline_name = f"thesis_{host}_baseline"
    control_name = f"thesis_{host}_boundary_weighted_logit_consistency"
    baseline = (experiment_dir / f"{baseline_name}.yaml").read_text(encoding="utf-8")
    control = (experiment_dir / f"{control_name}.yaml").read_text(encoding="utf-8")
    control_block = """
calibration_control:
  name: boundary_weighted_logit_consistency
  lambda_min: 0.01
  lambda_max: 1.0
  radius: 10.0
  alpha: 1.0
  boundary_weighting: native_linear
  view_geometry: independent_inverse_aligned
  consistency_signal: raw_final_logit
"""
    expected = baseline.replace(baseline_name, control_name)
    expected = expected.replace(
        "  name: baseline\n", "  name: boundary_weighted_logit_consistency\n"
    )
    expected = expected.replace("\neval:\n", f"{control_block}\neval:\n")
    assert control == expected


def test_resolved_provenance_hash_distinguishes_bwcr_from_baseline() -> None:
    control_config = {**_control_config(), "experiment_name": "synthetic_bwcr"}
    baseline_config = deepcopy(control_config)
    baseline_config["method"]["name"] = "baseline"
    del baseline_config["calibration_control"]
    margin_config = deepcopy(baseline_config)
    margin_config["experiment_name"] = "synthetic_margin"
    margin_config["method"]["name"] = "margin_label_smoothing"
    margin_config["calibration_control"] = {
        "name": "margin_label_smoothing",
        "margin": 10.0,
        "weight": 0.1,
    }
    git = {"sha": "a" * 40, "branch": "main", "dirty": False}

    control = run_provenance(control_config, git=git, run_id="bwcr-run")
    baseline = run_provenance(baseline_config, git=git, run_id="baseline-run")
    margin = run_provenance(margin_config, git=git, run_id="margin-run")

    assert control["resolved_config"]["calibration_control"]["name"] == CONTROL_NAME
    assert control["config_sha256"] != baseline["config_sha256"]
    assert control["config_sha256"] != margin["config_sha256"]
    assert control["scientific_identity_sha256"] != baseline[
        "scientific_identity_sha256"
    ]
    assert control["scientific_identity_sha256"] != margin[
        "scientific_identity_sha256"
    ]


def test_training_log_uses_bwcr_control_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _control_config()
    config["training"]["epochs"] = 0
    config["output_dir"] = str(tmp_path)
    trainer = Trainer(
        _TinyStudent(), None, None, config, run_record={"run_id": "synthetic"}
    )
    messages: list[str] = []
    monkeypatch.setattr(
        trainer_module.logger,
        "info",
        lambda message, *args: messages.append(message % args),
    )

    trainer.fit([], None)

    assert any(
        "method=boundary_weighted_logit_consistency" in message
        for message in messages
    )


def test_target_evaluator_remains_ordinary_single_pass_inference() -> None:
    source = (ROOT / "src/crisp/engine/evaluator.py").read_text(encoding="utf-8")
    lowered = source.lower()
    assert "bwcr" not in lowered
    assert "sample_bwcr_pair" not in source
    assert "native_bwcr_boundary_field" not in source
    assert "bwcr_consistency_loss" not in source
    assert source.index("results = self.predict_batch(batch") < source.index(
        'masks = batch["mask"]'
    )


def test_bwcr_validation_precedes_training_side_effects() -> None:
    source = (ROOT / "src/crisp/scripts/train.py").read_text(encoding="utf-8")
    validation = "bwcr_control = resolve_bwcr_control(config)"
    assert source.index(validation) < source.index("seed_everything(seed)")
    assert source.index(validation) < source.index("output_dir = ensure_dir")
    assert source.index(validation) < source.index("model = build_model(config)")
    assert "if bwcr_control is not None:" in source
    assert "teacher_ensemble = None" in source


def test_no_unetpp_bwcr_config_exists() -> None:
    assert not (
        ROOT
        / "configs/experiment/thesis_unetpp_boundary_weighted_logit_consistency.yaml"
    ).exists()
