"""
Trainer-level smoke tests for CRISP discipline and reproducibility.

These tests cover the remaining end-to-end invariants not exercised by the
math-only unit tests:
- the trainer can execute one CRISP step with explicit detached alpha_star
  supervision,
- current-protocol configs can require source validation explicitly,
- checkpoints preserve enough state for reproducible resume/audit.
"""

from __future__ import annotations

from pathlib import Path
from itertools import permutations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from crisp.engine.checkpointing import load_checkpoint
from crisp.engine.trainer import BOUNDARY_F1_WINDOW, SELECTION_METRIC, Trainer
from crisp.models.base import SegmentationOutput
from crisp.models.projector_head import CRISPProjectorHead
from crisp.tests_support.toy_data import make_toy_batch


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder_channels = 4
        self.feature_conv = nn.Conv2d(3, self.decoder_channels, kernel_size=3, padding=1)
        self.logit_conv = nn.Conv2d(self.decoder_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> SegmentationOutput:
        feat_full = torch.relu(self.feature_conv(x))
        logits = self.logit_conv(feat_full)
        features = F.avg_pool2d(feat_full, kernel_size=4, stride=4)
        return SegmentationOutput(logits=logits, features=features)


class _ConstantTeacherEnsemble(nn.Module):
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        b, _, h, w = x.shape
        return [
            torch.full((b, 1, h, w), 0.2, device=x.device),
            torch.full((b, 1, h, w), 0.8, device=x.device),
            torch.full((b, 1, h, w), 0.6, device=x.device),
        ]


def _base_crisp_config(output_dir: Path) -> dict:
    return {
        "seed": 7,
        "output_dir": str(output_dir),
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "require_validation": False,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "lr_student": 1.0e-4,
            "lr_projector": 2.0e-4,
            "weight_decay": 1.0e-4,
            "mixed_precision": False,
            "gradient_clip_norm": 0.0,
        },
        "method": {
            "use_crisp": True,
            "use_projector": True,
            "use_teachers": True,
            "target_mode": "boundary_posterior",
            "use_boundary_weighted_task": True,
            "use_identity_regularization": True,
            "use_amortization_loss": True,
            "allow_self_ensemble_teacher": False,
        },
        "crisp": {
            "boundary": {"sigma_b": 6.0, "mode": "gaussian_soft_field"},
            "teacher": {"tau": 1.0, "gamma": 1.5, "strict": True},
            "projection": {
                "lambda": 1.0,
                "mu": 0.25,
                "beta": 0.35,
                "eta_dice": 0.5,
                "alpha_min": 0.5,
                "alpha_max": 1.75,
                "eps_target": 1.0e-3,
                "zeta": 0.10,
                "zmax": 8.0,
            },
            "solver": {"newton_steps": 3, "bisection_steps": 12},
            "warmup": {"enabled": False, "epochs": 15},
        },
    }


def test_trainer_one_step_crisp_smoke(tmp_path: Path) -> None:
    """Trainer should execute one CRISP step with finite task and amortization losses."""
    config = _base_crisp_config(tmp_path)
    model = _TinyModel()
    projector = CRISPProjectorHead(feature_channels=model.decoder_channels)
    teachers = _ConstantTeacherEnsemble()
    trainer = Trainer(model=model, projector=projector, teacher_ensemble=teachers, config=config)

    batch = make_toy_batch(batch_size=2, image_size=32)
    out = trainer.train_one_step(batch, epoch=0, step=0)

    assert torch.isfinite(out.loss)
    assert out.logs["task_loss"] >= 0.0
    assert out.logs["amort_loss"] >= 0.0
    assert "solver/sat_lo" in out.logs
    assert "solver/sat_hi" in out.logs


def test_paper_optimizer_groups_and_scheduler(tmp_path: Path) -> None:
    config = _base_crisp_config(tmp_path)
    config["training"].pop("lr_projector")
    config["training"]["total_epochs"] = 120
    model = _TinyModel()
    projector = CRISPProjectorHead(feature_channels=model.decoder_channels)
    trainer = Trainer(model, projector, _ConstantTeacherEnsemble(), config)

    optimizer = trainer.build_optimizer()
    scheduler = trainer.build_scheduler(optimizer)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(5e-4)
    assert all(group["weight_decay"] == pytest.approx(1e-4) for group in optimizer.param_groups)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert scheduler.T_max == 120
    assert scheduler.eta_min == pytest.approx(1e-6)


def test_explicit_projector_lr_override_is_preserved(tmp_path: Path) -> None:
    config = _base_crisp_config(tmp_path)
    config["training"]["lr_projector"] = 7e-4
    trainer = Trainer(_TinyModel(), nn.Conv2d(4, 1, 1), _ConstantTeacherEnsemble(), config)

    assert trainer.build_optimizer().param_groups[1]["lr"] == pytest.approx(7e-4)


def test_trainer_solver_fallback_matches_manuscript(tmp_path: Path) -> None:
    config = _base_crisp_config(tmp_path)
    config["crisp"].pop("solver")
    trainer = Trainer(_TinyModel(), None, _ConstantTeacherEnsemble(), config)

    assert trainer.newton_steps == 3
    assert trainer.bisection_steps == 12


def test_trainer_requires_validation_when_configured(tmp_path: Path) -> None:
    """Current-protocol configs should fail if source validation is missing."""
    config = _base_crisp_config(tmp_path)
    config["training"]["require_validation"] = True

    trainer = Trainer(
        model=_TinyModel(),
        projector=None,
        teacher_ensemble=None,
        config={
            **config,
            "method": {
                "use_crisp": False,
                "use_projector": False,
                "use_teachers": False,
            },
        },
    )

    train_loader = [make_toy_batch(batch_size=2, image_size=32)]
    try:
        trainer.fit(train_loader, val_loader=None)
    except ValueError as exc:
        assert "validation" in str(exc).lower()
    else:
        raise AssertionError("Trainer.fit should require a validation loader when configured.")


def test_checkpoint_payload_includes_reproducibility_state(tmp_path: Path) -> None:
    """Periodic checkpoints should preserve scheduler/scaler/config/seed metadata."""
    config = _base_crisp_config(tmp_path)
    config["method"] = {
        "use_crisp": False,
        "use_projector": False,
        "use_teachers": False,
    }
    trainer = Trainer(
        model=_TinyModel(),
        projector=None,
        teacher_ensemble=None,
        config=config,
    )

    batch = make_toy_batch(batch_size=2, image_size=32)
    train_loader = [batch]
    trainer.fit(train_loader, val_loader=None)

    checkpoint = load_checkpoint(tmp_path / "epoch_1.pt")
    assert checkpoint["seed"] == 7
    assert checkpoint["scheduler_state_dict"] is not None
    assert checkpoint["grad_scaler_state_dict"] is not None
    assert checkpoint["config"]["training"]["scheduler"] == "cosine"
    assert "train_metrics" in checkpoint
    assert "best_boundary_f1" in checkpoint
    assert "best_bece" in checkpoint
    assert "best_epoch" in checkpoint
    assert checkpoint["selection_metric"] == SELECTION_METRIC


def test_thesis_schedule_keeps_phase_i_baseline_then_ramps_crisp(tmp_path: Path) -> None:
    """The current schedule should use 25 baseline epochs then ramp lambda/beta."""
    config = _base_crisp_config(tmp_path)
    config["crisp"]["schedule"] = {
        "enabled": True,
        "phase_i_epochs": 25,
        "phase_ii_epochs": 65,
        "phase_iii_epochs": 30,
        "phase_ii_ramp_epochs": 10,
    }
    trainer = Trainer(
        model=_TinyModel(),
        projector=CRISPProjectorHead(feature_channels=4),
        teacher_ensemble=_ConstantTeacherEnsemble(),
        config=config,
    )

    phase_i = trainer._crisp_schedule_state(0)
    phase_ii_start = trainer._crisp_schedule_state(25)
    phase_ii_full = trainer._crisp_schedule_state(34)
    phase_iii = trainer._crisp_schedule_state(90)

    assert phase_i["phase"] == "phase_i_baseline"
    assert phase_i["phase_id"] == 1
    assert phase_i["crisp_active"] is False
    assert phase_ii_start["phase"] == "phase_ii_crisp"
    assert phase_ii_start["phase_id"] == 2
    assert phase_ii_start["crisp_active"] is True
    assert abs(phase_ii_start["lambda_factor"] - 0.1) < 1e-8
    assert phase_ii_start["mu_factor"] == 1.0
    assert phase_ii_full["lambda_factor"] == 1.0
    assert phase_ii_full["beta_factor"] == 1.0
    assert phase_iii["phase"] == "phase_iii_finetune"
    assert phase_iii["phase_id"] == 3


def _selected_epoch(rows: list[tuple[int, float, float, float]]) -> int:
    candidates = [
        Trainer._validation_candidate(
            epoch, {"boundary_f1": boundary_f1, "bece": bece, "dice": dice}
        )
        for epoch, boundary_f1, bece, dice in rows
    ]
    return Trainer._select_validation_candidate(candidates).epoch


def test_selection_clear_boundary_f1_winner_outside_window() -> None:
    assert _selected_epoch([(0, 0.800, 0.01, 0.90), (1, 0.803, 0.50, 0.80)]) == 1


def test_selection_lower_bece_wins_within_window() -> None:
    assert _selected_epoch([(0, 0.800, 0.01, 0.80), (1, 0.801, 0.10, 0.90)]) == 0


def test_selection_exact_window_boundary_is_eligible() -> None:
    b_max = 0.802
    boundary = b_max - BOUNDARY_F1_WINDOW
    assert _selected_epoch([(0, boundary, 0.01, 0.80), (1, b_max, 0.50, 0.90)]) == 0


def test_selection_just_outside_window_is_ineligible() -> None:
    b_max = 0.802
    below = b_max - BOUNDARY_F1_WINDOW - 1e-6
    assert _selected_epoch([(0, below, 0.01, 0.80), (1, b_max, 0.50, 0.90)]) == 1


def test_selection_higher_dice_breaks_equal_bece() -> None:
    assert _selected_epoch([(0, 0.800, 0.10, 0.80), (1, 0.801, 0.10, 0.90)]) == 1


def test_selection_earliest_epoch_breaks_equal_bece_and_dice() -> None:
    assert _selected_epoch([(2, 0.801, 0.10, 0.90), (0, 0.800, 0.10, 0.90)]) == 0


def test_selection_is_independent_of_candidate_iteration_order() -> None:
    rows = [(0, 0.800, 0.01, 0.80), (1, 0.801, 0.10, 0.90), (2, 0.798, 0.001, 0.95)]
    assert {_selected_epoch(list(order)) for order in permutations(rows)} == {0}


def test_selection_reconsiders_history_after_late_new_boundary_f1_maximum() -> None:
    first_two = [(0, 0.800, 0.01, 0.80), (1, 0.801, 0.10, 0.90)]
    assert _selected_epoch(first_two) == 0
    assert _selected_epoch(first_two + [(2, 0.803, 0.20, 0.95)]) == 1


@pytest.mark.parametrize("metric", ["boundary_f1", "bece", "dice"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_selection_rejects_non_finite_validation_metric(metric: str, nonfinite: float) -> None:
    metrics = {"boundary_f1": 0.8, "bece": 0.1, "dice": 0.9}
    metrics[metric] = nonfinite
    with pytest.raises(ValueError, match=f"Epoch 4 validation {metric} is non-finite"):
        Trainer._validation_candidate(4, metrics)


def test_checkpoint_selection_materializes_selected_older_epoch_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import crisp.engine.evaluator as evaluator_module

    class MarkerModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.marker = nn.Parameter(torch.tensor(0.0))

    model = MarkerModel()
    config = _base_crisp_config(tmp_path)
    config["method"] = {"use_crisp": False, "use_projector": False, "use_teachers": False}
    config["training"]["epochs"] = 3
    config["training"]["scheduler"] = "none"
    trainer = Trainer(model=model, projector=None, teacher_ensemble=None, config=config)

    def fake_train_one_epoch(_loader: object, epoch: int) -> dict[str, float]:
        with torch.no_grad():
            model.marker.fill_(epoch + 1)
        return {"loss": 0.0}

    class FakeEvaluator:
        def __init__(self, current_model: nn.Module, _projector: object, _config: dict) -> None:
            self.current_model = current_model

        def evaluate_dataset(self, _loader: object, _name: str, projector_on: bool) -> dict[str, float]:
            metrics = [
                {"boundary_f1": 0.800, "bece": 0.01, "dice": 0.80},
                {"boundary_f1": 0.801, "bece": 0.10, "dice": 0.90},
                {"boundary_f1": 0.803, "bece": 0.20, "dice": 0.95},
            ]
            return metrics[int(self.current_model.marker.item()) - 1]

    monkeypatch.setattr(trainer, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(evaluator_module, "Evaluator", FakeEvaluator)
    trainer.fit(train_loader=[], val_loader=[{}])

    best = load_checkpoint(tmp_path / "best.pt")
    latest = load_checkpoint(tmp_path / "epoch_3.pt")
    assert best["epoch"] == best["best_epoch"] == 1
    assert best["best_boundary_f1"] == pytest.approx(0.801)
    assert best["best_bece"] == pytest.approx(0.10)
    assert best["selection_metric"] == SELECTION_METRIC
    assert best["model_state_dict"]["marker"].item() == pytest.approx(2.0)
    assert latest["epoch"] == 2
    assert latest["best_epoch"] == 1
    assert latest["model_state_dict"]["marker"].item() == pytest.approx(3.0)


def test_training_total_epochs_and_phases_schema_controls_schedule(tmp_path: Path) -> None:
    config = _base_crisp_config(tmp_path)
    config["training"]["total_epochs"] = 120
    config["training"]["phases"] = {
        "baseline_warmup": 25,
        "crisp_full": 65,
        "finetune": 30,
        "phase2_ramp_epochs": 10,
    }
    trainer = Trainer(
        model=_TinyModel(),
        projector=CRISPProjectorHead(feature_channels=4),
        teacher_ensemble=_ConstantTeacherEnsemble(),
        config=config,
    )

    assert trainer.epochs == 120
    assert trainer.schedule_enabled is True
    assert trainer._crisp_schedule_state(0)["phase"] == "phase_i_baseline"
    assert trainer._crisp_schedule_state(25)["phase"] == "phase_ii_crisp"
    assert trainer._crisp_schedule_state(90)["phase"] == "phase_iii_finetune"
