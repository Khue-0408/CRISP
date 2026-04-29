"""
Trainer-level smoke tests for CRISP discipline and reproducibility.

These tests cover the remaining end-to-end invariants not exercised by the
math-only unit tests:
- the trainer can execute one CRISP step with explicit detached alpha_star
  supervision,
- paper-faithful configs can require source validation explicitly,
- checkpoints preserve enough state for reproducible resume/audit.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from crisp.engine.checkpointing import load_checkpoint
from crisp.engine.trainer import Trainer
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


def test_trainer_requires_validation_when_configured(tmp_path: Path) -> None:
    """Paper-faithful configs should fail if source validation is missing."""
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
    assert checkpoint["selection_metric"] == "validation_boundary_f1_then_bece_then_dice"


def test_thesis_schedule_keeps_phase_i_baseline_then_ramps_crisp(tmp_path: Path) -> None:
    """Updated thesis schedule should use 25 baseline epochs then ramp lambda/beta."""
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


def test_validation_score_prefers_boundary_f1_then_lower_bece() -> None:
    assert Trainer._validation_score({"boundary_f1": 0.8, "bece": 0.2}) > Trainer._validation_score(
        {"boundary_f1": 0.7, "bece": 0.01}
    )
    assert Trainer._validation_score({"boundary_f1": 0.8, "bece": 0.1}) > Trainer._validation_score(
        {"boundary_f1": 0.8, "bece": 0.2}
    )
    assert Trainer._validation_score({"boundary_f1": 0.8, "bece": 0.1, "dice": 0.9}) > Trainer._validation_score(
        {"boundary_f1": 0.8, "bece": 0.1, "dice": 0.8}
    )


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
