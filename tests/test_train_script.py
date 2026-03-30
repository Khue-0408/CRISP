"""
Entry-point wiring tests for paper-faithful training discipline.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from crisp.models.unet import UNet
from crisp.scripts.train import _maybe_build_teacher_ensemble


def test_strict_teacher_builder_requires_checkpoint_paths(tmp_path: Path) -> None:
    """Strict CRISP configs must fail loudly when a configured teacher checkpoint is missing."""
    cfg = {
        "crisp": {"teacher": {"strict": True}},
        "teachers": [
            {
                "model": {"name": "unet", "in_channels": 3, "num_classes": 1},
                "checkpoint": "",
            }
        ],
    }

    with pytest.raises(ValueError, match="teacher pool"):
        _maybe_build_teacher_ensemble(cfg)


def test_strict_teacher_builder_loads_complete_pool(tmp_path: Path) -> None:
    """Strict teacher loading should succeed when every configured checkpoint is present."""
    ckpt_path = tmp_path / "teacher.pt"
    torch.save({"model_state_dict": UNet().state_dict()}, ckpt_path)

    cfg = {
        "crisp": {"teacher": {"strict": True}},
        "teachers": [
            {
                "model": {"name": "unet", "in_channels": 3, "num_classes": 1},
                "checkpoint": str(ckpt_path),
            }
        ],
    }

    ensemble = _maybe_build_teacher_ensemble(cfg)
    assert ensemble is not None
    assert len(ensemble.teachers) == 1


def test_teacher_pool_group_is_supported(tmp_path: Path) -> None:
    """U-Net teacher pool configs may provide teachers under `teacher_pool.teachers`."""
    ckpt_path = tmp_path / "teacher.pt"
    torch.save({"model_state_dict": UNet().state_dict()}, ckpt_path)

    cfg = {
        "crisp": {"teacher": {"strict": True}},
        "teacher_pool": {
            "teachers": [
                {
                    "model": {"name": "unet", "in_channels": 3, "num_classes": 1},
                    "checkpoint": str(ckpt_path),
                }
            ]
        },
    }

    ensemble = _maybe_build_teacher_ensemble(cfg)
    assert ensemble is not None
    assert len(ensemble.teachers) == 1
