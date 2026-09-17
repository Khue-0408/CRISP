"""Checkpoint integrity at the frozen-teacher construction boundary."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

import crisp.registry as registry
from crisp.models.teacher_wrapper import FrozenTeacher


def _tiny_model() -> nn.Conv2d:
    return nn.Conv2d(3, 1, kernel_size=1)


def test_valid_sam_like_class_with_empty_checkpoint_fails(monkeypatch) -> None:
    monkeypatch.setattr(registry, "_MODEL_REGISTRY", {"test_model": nn.Conv2d})
    model = registry.build_model({"model": {
        "class_path": "torch.nn.Conv2d", "in_channels": 3, "out_channels": 1, "kernel_size": 1,
    }})
    assert isinstance(model, nn.Conv2d)
    with pytest.raises(ValueError, match="sammamba.*nonempty pretrained checkpoint artifact"):
        FrozenTeacher(model, checkpoint_path="", teacher_name="sammamba")
    with pytest.raises(ValueError, match="sammamba.*nonempty pretrained checkpoint artifact"):
        FrozenTeacher(model, checkpoint_path="   ", teacher_name="sammamba")


def test_robustness_config_without_sam_environment_has_no_artifact(monkeypatch) -> None:
    monkeypatch.delenv("CRISP_SAMMAMBA_CLASS", raising=False)
    monkeypatch.delenv("CRISP_SAMMAMBA_CKPT", raising=False)
    config_path = Path(__file__).resolve().parents[1] / "configs/teacher_pool/thesis_robustness.yaml"
    teacher = yaml.safe_load(config_path.read_text(encoding="utf-8"))["teachers"][2]
    assert teacher["enabled"] is True and teacher["name"] == "sammamba"
    assert teacher["model_config"]["class_path"] == '${oc.env:CRISP_SAMMAMBA_CLASS,""}'
    assert teacher["checkpoint"] == '${oc.env:CRISP_SAMMAMBA_CKPT,""}'
    with pytest.raises(ValueError, match="sammamba.*nonempty pretrained checkpoint artifact"):
        FrozenTeacher(_tiny_model(), checkpoint_path="", teacher_name=teacher["name"])


def test_missing_teacher_checkpoint_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="sammamba"):
        FrozenTeacher(
            _tiny_model(), checkpoint_path=str(tmp_path / "absent.pt"), teacher_name="sammamba",
        )


def test_incompatible_teacher_checkpoint_fails_strictly(tmp_path: Path) -> None:
    path = tmp_path / "incompatible.pt"
    torch.save({"state_dict": {"weight": torch.ones(2, 3, 1, 1), "bias": torch.zeros(1)}}, path)
    with pytest.raises(RuntimeError, match="size mismatch"):
        FrozenTeacher(_tiny_model(), checkpoint_path=str(path), teacher_name="sammamba")


def test_nonstrict_teacher_checkpoint_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "teacher.pt"
    torch.save({"state_dict": _tiny_model().state_dict()}, path)
    with pytest.raises(ValueError, match="sammamba.*strict checkpoint loading"):
        FrozenTeacher(
            _tiny_model(), checkpoint_path=str(path), teacher_name="sammamba",
            checkpoint_loading={"strict": False},
        )


def test_matching_teacher_checkpoint_loads_and_stays_frozen(tmp_path: Path) -> None:
    source = _tiny_model()
    with torch.no_grad():
        source.weight.fill_(2.0)
        source.bias.fill_(0.5)
    path = tmp_path / "matching.pt"
    torch.save({"state_dict": source.state_dict()}, path)

    target = _tiny_model()
    with torch.no_grad():
        target.weight.zero_()
        target.bias.zero_()
    teacher = FrozenTeacher(target, checkpoint_path=str(path), teacher_name="sammamba")
    torch.testing.assert_close(teacher.model.weight, source.weight)
    torch.testing.assert_close(teacher.model.bias, source.bias)
    assert not torch.equal(teacher.model.weight, torch.zeros_like(source.weight))
    assert not teacher.training and not teacher.model.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    teacher.train()
    assert not teacher.training and not teacher.model.training
    images = torch.ones(1, 3, 2, 2, requires_grad=True)
    logits = teacher.forward_logits(images)
    probabilities = teacher(images)
    torch.testing.assert_close(logits, torch.full_like(logits, 6.5))
    torch.testing.assert_close(probabilities, torch.sigmoid(logits))
    assert not logits.requires_grad and not probabilities.requires_grad
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_explicit_synthetic_teacher_bypass_is_test_only() -> None:
    teacher = FrozenTeacher(
        _tiny_model(), checkpoint_path="", teacher_name="synthetic_test_teacher",
        allow_uninitialized_for_testing=True,
    )
    assert not teacher.training and not teacher.model.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
