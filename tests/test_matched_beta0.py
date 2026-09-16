"""The matched control changes only the amortization coefficient."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import crisp.engine.trainer as trainer_module
from crisp.engine.trainer import Trainer
from crisp.models.base import SegmentationOutput
from crisp.models.projector_head import CRISPProjectorHead
from crisp.tests_support.toy_data import make_toy_batch


ROOT = Path(__file__).resolve().parents[1]
BETA_OVERRIDE = "\ncrisp:\n  projection:\n    beta: 0.0\n"


@pytest.mark.parametrize("host", ["unet", "unetpp", "pranet"])
def test_matched_beta0_config_has_only_beta_scientific_delta(host: str) -> None:
    experiment_dir = ROOT / "configs" / "experiment"
    full_name = f"thesis_{host}_crisp"
    matched_name = f"thesis_{host}_matched_beta0"
    full = (experiment_dir / f"{full_name}.yaml").read_text(encoding="utf-8")
    matched = (experiment_dir / f"{matched_name}.yaml").read_text(encoding="utf-8")
    assert matched.count(BETA_OVERRIDE) == 1
    normalized = matched.replace(matched_name, full_name).replace(BETA_OVERRIDE, "")
    assert normalized == full
    assert "  - /crisp: default\n" in full
    assert "  - /teacher_pool: thesis_default\n" in full
    assert "  - _self_\n" in full
    assert "  use_amortization_loss: true\n" in matched
    assert "  use_projector: true\n" in matched
    assert "  use_teachers: true\n" in matched
    assert "  beta: 0.35\n" in (ROOT / "configs/crisp/default.yaml").read_text(encoding="utf-8")


class _TinyStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, 3, padding=1)
        self.final = nn.Conv2d(4, 1, 1)

    def forward(self, images: torch.Tensor) -> SegmentationOutput:
        features = torch.tanh(self.encoder(images))
        return SegmentationOutput(
            logits=self.final(features), features=F.avg_pool2d(features, 4)
        )


class _FixedTeachers(nn.Module):
    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        b, _, h, w = images.shape
        return [
            torch.full((b, 1, h, w), value, device=images.device)
            for value in (0.2, 0.8, 0.6)
        ]


def _config(beta: float) -> dict:
    return {
        "training": {"total_epochs": 120, "phases": {
            "baseline_warmup": 25, "crisp_full": 65, "finetune": 30,
            "phase2_ramp_epochs": 10,
        }, "mixed_precision": False},
        "method": {"use_crisp": True, "use_projector": True, "use_teachers": True,
                   "target_mode": "boundary_posterior",
                   "use_boundary_weighted_task": True,
                   "use_identity_regularization": True,
                   "use_amortization_loss": True},
        "crisp": {
            "boundary": {"sigma_b": 6.0, "mode": "gaussian_soft_field"},
            "teacher": {"tau": 1.0, "gamma": 1.5, "strict": True},
            "projection": {"lambda": 1.0, "mu": 0.25, "beta": beta,
                           "eta_dice": 0.5, "alpha_min": 0.5, "alpha_max": 1.75,
                           "eps_target": 1e-3, "zeta": 0.1, "zmax": 8.0},
            "solver": {"newton_steps": 3, "bisection_steps": 12},
        },
    }


def test_matched_beta0_runtime_keeps_projection_and_only_removes_amortization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(27)
    model = _TinyStudent()
    projector = CRISPProjectorHead(4)
    teachers = _FixedTeachers()
    batch = make_toy_batch(batch_size=2, image_size=32)
    observed: dict[str, dict] = {"full": {}, "beta0": {}}
    current = "full"

    def record(name: str, *tensor_indices: int):
        original = getattr(trainer_module, name)

        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            observed[current][name] = {
                "inputs": [args[index].detach().clone() for index in tensor_indices],
                "result": result,
            }
            return result

        monkeypatch.setattr(trainer_module, name, wrapped)

    record("compute_boundary_posterior_target", 0, 1, 2)
    record("clip_posterior_target", 0)
    record("crisp_task_loss", 0, 1, 2, 3)
    record("solve_alpha_star", 0, 1, 2)
    record("crisp_amortization_loss", 0, 1, 2, 3)

    steps = {}
    for arm, beta in (("full", 0.35), ("beta0", 0.0)):
        current = arm
        trainer = Trainer(model, projector, teachers, _config(beta))
        assert trainer.use_amortization_loss
        assert trainer._crisp_schedule_state(90)["beta_factor"] == 1.0
        steps[arm] = trainer.train_one_step(batch, epoch=90, step=0)

    for name in observed["full"]:
        full = observed["full"][name]
        beta0 = observed["beta0"][name]
        for left, right in zip(full["inputs"], beta0["inputs"]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        left, right = full["result"], beta0["result"]
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        else:
            torch.testing.assert_close(left[0], right[0], rtol=0, atol=0)

    task = observed["beta0"]["crisp_task_loss"]["result"]["task_loss"]
    amort = observed["beta0"]["crisp_amortization_loss"]["result"]["amort_loss"]
    torch.testing.assert_close(steps["beta0"].loss, task)
    torch.testing.assert_close(steps["full"].loss - steps["beta0"].loss, 0.35 * amort)
    weight = projector.conv1.weight
    grad_task = torch.autograd.grad(task, weight, retain_graph=True)[0]
    grad_amort = torch.autograd.grad(amort, weight, retain_graph=True)[0]
    grad_beta0 = torch.autograd.grad(steps["beta0"].loss, weight)[0]
    grad_full = torch.autograd.grad(steps["full"].loss, weight)[0]
    torch.testing.assert_close(grad_beta0, grad_task)
    torch.testing.assert_close(grad_full, grad_task + 0.35 * grad_amort)
    assert grad_task.abs().sum() > 0
    assert grad_amort.abs().sum() > 0
    assert "solver/sat_lo" in steps["beta0"].logs
    assert "solver/sat_hi" in steps["beta0"].logs


def test_matched_beta0_schedule_preserves_phases_and_zero_effective_beta() -> None:
    model = _TinyStudent()
    projector = CRISPProjectorHead(4)
    teachers = _FixedTeachers()
    full = Trainer(model, projector, teachers, _config(0.35))
    matched = Trainer(model, projector, teachers, _config(0.0))
    for epoch in (0, 24, 25, 29, 34, 89, 90, 119):
        full_state = full._crisp_schedule_state(epoch)
        matched_state = matched._crisp_schedule_state(epoch)
        assert full_state == matched_state
        assert matched.beta_value * matched_state["beta_factor"] == 0.0
        assert full.lambda_value * full_state["lambda_factor"] == (
            matched.lambda_value * matched_state["lambda_factor"]
        )
