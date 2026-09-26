"""Scientific contract tests for the author-approved Margin-LS control."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import crisp.engine.trainer as trainer_module
from crisp.engine.trainer import Trainer
from crisp.models.base import SegmentationOutput
from crisp.modules.margin_label_smoothing import (
    CONTROL_NAME,
    margin_label_smoothing_penalty,
    resolve_margin_label_smoothing_control,
)
from crisp.utils.provenance import run_provenance


ROOT = Path(__file__).resolve().parents[1]


def _general_two_class_penalty(z: torch.Tensor, margin: float) -> torch.Tensor:
    logits = torch.stack((torch.zeros_like(z), z), dim=-1)
    maximum = logits.max(dim=-1, keepdim=True).values
    return torch.relu(maximum - logits - margin).sum(dim=-1)


@pytest.mark.parametrize("z", [12.0, -12.0, 4.0, -4.0, 10.0, -10.0, 10.5, -10.5])
def test_binary_reduction_matches_general_mbls_penalty(z: float) -> None:
    value = torch.tensor(z)
    expected = torch.relu(value.abs() - 10.0)
    torch.testing.assert_close(_general_two_class_penalty(value, 10.0), expected)


def test_margin_penalty_is_batch_pixel_mean_of_raw_logits() -> None:
    logits = torch.tensor([[[[12.0, -14.0], [9.0, -10.0]]]])
    expected = torch.tensor((2.0 + 4.0 + 0.0 + 0.0) / 4.0)
    torch.testing.assert_close(margin_label_smoothing_penalty(logits), expected)


def test_margin_penalty_gradients_follow_binary_contract() -> None:
    logits = torch.tensor([9.0, 12.0, -13.0], requires_grad=True)
    penalty = margin_label_smoothing_penalty(logits)
    penalty.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([0.0, 1.0 / 3.0, -1.0 / 3.0]))


@pytest.mark.parametrize("z", [10.0, -10.0])
def test_margin_penalty_is_zero_at_margin(z: float) -> None:
    assert margin_label_smoothing_penalty(torch.tensor([z])).item() == 0.0


def _control_config() -> dict:
    return {
        "training": {
            "epochs": 2,
            "optimizer": "adamw",
            "scheduler": "none",
            "mixed_precision": False,
        },
        "method": {
            "name": CONTROL_NAME,
            "use_crisp": False,
            "use_projector": False,
            "use_teachers": False,
        },
        "calibration_control": {
            "name": CONTROL_NAME,
            "margin": 10.0,
            "weight": 0.1,
        },
    }


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("margin", 0.0, "margin > 0"),
        ("margin", -1.0, "margin > 0"),
        ("margin", float("inf"), "margin > 0"),
        ("weight", -0.1, "weight >= 0"),
        ("weight", float("inf"), "weight >= 0"),
        ("margin", 9.0, "locks margin=10.0 and weight=0.1"),
        ("weight", 0.2, "locks margin=10.0 and weight=0.1"),
    ],
)
def test_control_parameters_fail_loudly(key: str, value: float, error: str) -> None:
    config = _control_config()
    config["calibration_control"][key] = value
    with pytest.raises(ValueError, match=error):
        resolve_margin_label_smoothing_control(config)


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
def test_control_rejects_crisp_only_machinery(flag: str) -> None:
    config = _control_config()
    config["method"][flag] = True
    with pytest.raises(ValueError, match=flag):
        resolve_margin_label_smoothing_control(config)


def test_control_rejects_configured_teacher_artifacts() -> None:
    config = _control_config()
    config["teacher_pool"] = {"teachers": [{"name": "not_allowed"}]}
    with pytest.raises(ValueError, match="forbids configured teacher"):
        resolve_margin_label_smoothing_control(config)


class _FixedLogitStudent(nn.Module):
    def __init__(self, value: float = 12.0) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(value))

    def forward(self, images: torch.Tensor) -> SegmentationOutput:
        b, _, h, w = images.shape
        logits = self.logit.expand(b, 1, h, w)
        return SegmentationOutput(logits=logits, features=images[:, :1])


def test_unet_style_control_is_exact_baseline_plus_weighted_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FixedLogitStudent()
    baseline_config = deepcopy(_control_config())
    baseline_config["method"]["name"] = "baseline"
    del baseline_config["calibration_control"]
    baseline = Trainer(model, None, None, baseline_config)
    control = Trainer(model, None, None, _control_config())
    batch = {
        "image": torch.zeros(2, 3, 4, 5),
        "mask": torch.zeros(2, 1, 4, 5),
    }

    def forbidden(*_args, **_kwargs):
        raise AssertionError("CRISP-only runtime path was entered")

    monkeypatch.setattr(trainer_module, "compute_boundary_weight", forbidden)
    monkeypatch.setattr(trainer_module, "solve_alpha_star", forbidden)
    baseline_step = baseline.train_one_step(batch, epoch=0, step=0)
    control_step = control.train_one_step(batch, epoch=0, step=0)
    expected_penalty = margin_label_smoothing_penalty(model(batch["image"]).logits)
    torch.testing.assert_close(control_step.loss, baseline_step.loss + 0.1 * expected_penalty)
    assert control_step.logs["bce"] == pytest.approx(baseline_step.logs["bce"])
    assert control_step.logs["dice"] == pytest.approx(baseline_step.logs["dice"])
    assert control_step.logs["margin_penalty"] == pytest.approx(2.0)
    assert control.projector is None and control.teacher_ensemble is None

    margin_contribution = control_step.loss - baseline_step.loss
    gradient = torch.autograd.grad(margin_contribution, model.logit)[0]
    assert gradient.item() == pytest.approx(0.1)


@pytest.mark.parametrize("component", ["projector", "teacher"])
def test_trainer_rejects_constructed_crisp_component(component: str) -> None:
    model = _FixedLogitStudent()
    extra = nn.Identity()
    projector = extra if component == "projector" else None
    teacher = extra if component == "teacher" else None
    with pytest.raises(ValueError, match="cannot receive"):
        Trainer(model, projector, teacher, _control_config())


def test_invalid_control_fails_before_training_side_effects() -> None:
    config = _control_config()
    config["calibration_control"]["margin"] = 0.0
    with pytest.raises(ValueError, match="margin > 0"):
        resolve_margin_label_smoothing_control(config)
    source = (ROOT / "src/crisp/scripts/train.py").read_text(encoding="utf-8")
    validation = "margin_control = resolve_margin_label_smoothing_control(config)"
    assert source.index(validation) < source.index("seed_everything(seed)")
    assert source.index(validation) < source.index("output_dir = ensure_dir")
    assert source.index(validation) < source.index("model = build_model(config)")
    assert "None if margin_control is not None else _maybe_build_teacher_ensemble(config)" in source


@pytest.mark.parametrize("host", ["unet", "pranet"])
def test_current_control_config_has_only_approved_baseline_delta(host: str) -> None:
    experiment_dir = ROOT / "configs" / "experiment"
    baseline_name = f"thesis_{host}_baseline"
    control_name = f"thesis_{host}_margin_label_smoothing"
    baseline = (experiment_dir / f"{baseline_name}.yaml").read_text(encoding="utf-8")
    control = (experiment_dir / f"{control_name}.yaml").read_text(encoding="utf-8")
    control_block = """
calibration_control:
  name: margin_label_smoothing
  margin: 10.0
  weight: 0.1
"""
    expected = baseline.replace(baseline_name, control_name)
    expected = expected.replace("  name: baseline\n", "  name: margin_label_smoothing\n")
    expected = expected.replace("\neval:\n", f"{control_block}\neval:\n")
    assert control == expected


def test_resolved_config_provenance_exposes_and_hashes_control() -> None:
    control_config = {
        **_control_config(),
        "experiment_name": "synthetic_margin_control",
        "seed": 2026,
    }
    baseline_config = deepcopy(control_config)
    baseline_config["method"]["name"] = "baseline"
    del baseline_config["calibration_control"]
    git = {"sha": "a" * 40, "branch": "main", "dirty": False}
    record = run_provenance(control_config, git=git, run_id="synthetic-run")
    baseline_record = run_provenance(baseline_config, git=git, run_id="baseline-run")
    assert record["method"]["name"] == CONTROL_NAME
    assert record["resolved_config"]["calibration_control"] == {
        "name": CONTROL_NAME,
        "margin": 10.0,
        "weight": 0.1,
    }
    assert record["config_sha256"] != baseline_record["config_sha256"]
    assert record["scientific_identity_sha256"] != baseline_record["scientific_identity_sha256"]
