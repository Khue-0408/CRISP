"""Smoke tests for baseline-backed model adapters and real checkpoint keyspaces."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from crisp.engine.trainer import Trainer
from crisp.models.teacher_wrapper import FrozenTeacher
from crisp.models.projector_head import CRISPProjectorHead
from crisp.registry import build_model
from crisp.scripts.train import _maybe_build_teacher_ensemble
from crisp.tests_support.toy_data import make_toy_batch
from crisp.utils.model_loading import load_model_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]
POLYP_PVT_BACKBONE_CKPT = REPO_ROOT / "1_baseline/Polyp-PVT/pretrained_pth/pvt_v2_b2.pth"


def _skip_if_polyp_pvt_backbone_missing(model_cfg: dict) -> None:
    if (
        str(model_cfg.get("name", "")).lower() == "polyp_pvt"
        and not POLYP_PVT_BACKBONE_CKPT.exists()
    ):
        pytest.skip(f"Missing local Polyp-PVT backbone pretrain: {POLYP_PVT_BACKBONE_CKPT}")


@pytest.mark.parametrize(
    ("name", "model_cfg", "checkpoint"),
    [
        (
            "unet",
            {"name": "unet", "in_channels": 3, "num_classes": 1, "base_channels": 64},
            "1_baseline/unet/kvasir-seg-best-checkpoint/UNet_IoUBCELoss_augmented.pth",
        ),
        (
            "unetpp",
            {"name": "unetpp", "in_channels": 3, "num_classes": 1},
            "1_baseline/UNet++/output/models/model.pth",
        ),
        (
            "pranet",
            {"name": "pranet", "in_channels": 3, "num_classes": 1, "pretrained": False},
            "1_baseline/PraNet/PraNet-19.pth",
        ),
        (
            "polyp_pvt",
            {"name": "polyp_pvt", "in_channels": 3, "num_classes": 1},
            "1_baseline/Polyp-PVT/model_pth/PolypPVT.pth",
        ),
        (
            "uacanet",
            {"name": "uacanet", "in_channels": 3, "num_classes": 1, "pretrained": False},
            "1_baseline/UACANet/snapshots/UACANet-L/latest.pth",
        ),
    ],
)
def test_real_baseline_checkpoint_loads_strictly(
    name: str,
    model_cfg: dict,
    checkpoint: str,
) -> None:
    checkpoint_path = REPO_ROOT / checkpoint
    if not checkpoint_path.exists():
        pytest.skip(f"Missing local baseline checkpoint: {checkpoint_path}")
    _skip_if_polyp_pvt_backbone_missing(model_cfg)

    model = build_model({"model": model_cfg})
    diagnostics = load_model_checkpoint(model, checkpoint_path, strict=True)

    assert diagnostics["missing_keys"] == []
    assert diagnostics["unexpected_keys"] == []

    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))

    assert out.logits.shape == (1, 1, 64, 64), name
    assert out.features.ndim == 4, name
    assert out.features.shape[1] == model.decoder_channels, name


def test_real_teacher_ensemble_outputs_aligned_probability_maps() -> None:
    """The thesis teacher pool should load strictly and emit aligned probabilities."""
    uacanet_ckpt = REPO_ROOT / "1_baseline/UACANet/snapshots/UACANet-L/latest.pth"
    polyp_pvt_ckpt = REPO_ROOT / "1_baseline/Polyp-PVT/model_pth/PolypPVT.pth"
    missing = [path for path in (uacanet_ckpt, polyp_pvt_ckpt) if not path.exists()]
    if missing:
        pytest.skip(f"Missing local teacher checkpoint(s): {missing}")

    ensemble = _maybe_build_teacher_ensemble(
        {
            "crisp": {"teacher": {"strict": True}},
            "teacher_pool": {
                "teachers": [
                    {
                        "name": "uacanet_l",
                        "model": "uacanet_l",
                        "model_config": {
                            "name": "uacanet",
                            "in_channels": 3,
                            "num_classes": 1,
                            "pretrained": False,
                        },
                        "checkpoint": str(uacanet_ckpt),
                        "checkpoint_loading": {"strict": True},
                    },
                    {
                        "name": "polyp_pvt",
                        "model": "polyp_pvt",
                        "model_config": {
                            "name": "polyp_pvt",
                            "in_channels": 3,
                            "num_classes": 1,
                        },
                        "checkpoint": str(polyp_pvt_ckpt),
                        "checkpoint_loading": {"strict": True},
                    },
                ]
            },
        }
    )

    assert ensemble is not None
    with torch.no_grad():
        outputs = ensemble(torch.randn(1, 3, 64, 64))

    assert len(outputs) == 2
    for prob in outputs:
        assert prob.shape == (1, 1, 64, 64)
        assert torch.isfinite(prob).all()
        assert prob.min() >= 0.0
        assert prob.max() <= 1.0

    student = build_model(
        {"model": {"name": "unet", "in_channels": 3, "num_classes": 1, "base_channels": 8}}
    )
    projector = CRISPProjectorHead(feature_channels=student.decoder_channels)
    trainer = Trainer(
        model=student,
        projector=projector,
        teacher_ensemble=ensemble,
        config={
            "seed": 2026,
            "training": {
                "epochs": 1,
                "optimizer": "adamw",
                "scheduler": "none",
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
                "boundary": {"sigma_b": 6.0},
                "teacher": {"tau": 1.0, "gamma": 1.5, "strict": True},
                "projection": {
                    "lambda": 1.0,
                    "mu": 0.25,
                    "beta": 0.35,
                    "alpha_min": 0.5,
                    "alpha_max": 1.75,
                    "eps_target": 1.0e-3,
                    "zeta": 0.10,
                    "zmax": 8.0,
                },
                "warmup": {"enabled": False},
            },
        },
    )
    step = trainer.train_one_step(make_toy_batch(batch_size=1, image_size=64), epoch=0, step=0)
    step.loss.backward()

    assert torch.isfinite(step.loss)
    assert step.logs["task_loss"] >= 0.0
    assert step.logs["amort_loss"] >= 0.0
    assert any(param.grad is not None for param in student.parameters())
    assert any(param.grad is not None for param in projector.parameters())
    assert all(param.grad is None for param in ensemble.parameters())


@pytest.mark.parametrize(
    ("name", "model_cfg"),
    [
        ("unet", {"name": "unet", "in_channels": 3, "num_classes": 1, "base_channels": 64}),
        ("unetpp", {"name": "unetpp", "in_channels": 3, "num_classes": 1}),
        ("pranet", {"name": "pranet", "in_channels": 3, "num_classes": 1, "pretrained": False}),
    ],
)
def test_student_adapter_forward_352_exposes_crisp_contract(name: str, model_cfg: dict) -> None:
    _skip_if_polyp_pvt_backbone_missing(model_cfg)
    model = build_model({"model": model_cfg})
    model.eval()

    with torch.no_grad():
        out = model(torch.randn(1, 3, 352, 352))

    assert out.logits.shape == (1, 1, 352, 352), name
    assert out.features.ndim == 4, name
    assert out.features.shape[0] == 1, name
    assert out.features.shape[1] == model.decoder_channels, name


@pytest.mark.parametrize(
    ("name", "model_cfg"),
    [
        ("polyp_pvt", {"name": "polyp_pvt", "in_channels": 3, "num_classes": 1}),
        (
            "uacanet_l",
            {
                "name": "uacanet_l",
                "in_channels": 3,
                "num_classes": 1,
                "pretrained": False,
            },
        ),
    ],
)
def test_teacher_adapter_forward_352_outputs_detached_probability(name: str, model_cfg: dict) -> None:
    _skip_if_polyp_pvt_backbone_missing(model_cfg)
    teacher = FrozenTeacher(build_model({"model": model_cfg}), checkpoint_path="")
    teacher.eval()

    with torch.no_grad():
        prob = teacher(torch.randn(1, 3, 352, 352))

    assert prob.shape == (1, 1, 352, 352), name
    assert torch.isfinite(prob).all(), name
    assert prob.min() >= 0.0, name
    assert prob.max() <= 1.0, name
    assert prob.requires_grad is False, name
