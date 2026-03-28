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
            "boundary": {"sigma_b": 3.0, "mode": "gaussian_soft_field"},
            "teacher": {"tau": 1.0, "gamma": 6.0, "strict": True},
            "projection": {
                "lambda": 0.8,
                "mu": 0.05,
                "beta": 0.2,
                "eta_dice": 0.5,
                "alpha_min": 0.5,
                "alpha_max": 1.8,
                "eps_target": 1.0e-4,
                "zeta": 1.0e-2,
                "zmax": 12.0,
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
