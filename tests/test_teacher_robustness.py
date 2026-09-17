"""Controlled checks for the opt-in three-teacher robustness control."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import crisp.engine.trainer as trainer_module
from crisp.engine.trainer import Trainer
from crisp.models.base import SegmentationOutput
from crisp.models.projector_head import CRISPProjectorHead
from crisp.models.teacher_wrapper import FrozenTeacher, TeacherEnsemble
from crisp.modules.teacher_posterior import (
    aggregate_teacher_posterior,
    binary_entropy,
    corrupt_teacher_logits,
    stable_teacher_noise_seed,
    teacher_robustness_posterior,
)
from crisp.tests_support.toy_data import make_toy_batch


ROOT = Path(__file__).resolve().parents[1]
TEACHER_NAMES = ["uacanet_l", "polyp_pvt", "sammamba"]


def test_robustness_configs_keep_default_two_teacher_pool_unchanged() -> None:
    default = (ROOT / "configs/teacher_pool/thesis_default.yaml").read_text(encoding="utf-8")
    robust = (ROOT / "configs/teacher_pool/thesis_robustness.yaml").read_text(encoding="utf-8")
    assert [line.removeprefix("    name: ") for line in default.splitlines()
            if line.startswith("    name: ")] == TEACHER_NAMES[:2]
    assert [line.removeprefix("    name: ") for line in robust.splitlines()
            if line.startswith("    name: ")] == TEACHER_NAMES
    assert robust.count("  - enabled: true") == 3
    weighted = (ROOT / "configs/experiment/thesis_pranet_teacher_robustness_weighted.yaml").read_text(encoding="utf-8")
    equal = (ROOT / "configs/experiment/thesis_pranet_teacher_robustness_equal.yaml").read_text(encoding="utf-8")
    assert "/teacher_pool: thesis_robustness" in weighted
    assert "logit_noise_std: 1.0" in weighted
    assert "seed_keys: [seed, dataset_name, image_id]" in weighted
    assert "aggregation: weighted" in weighted
    assert equal.replace("thesis_pranet_teacher_robustness_equal", "thesis_pranet_teacher_robustness_weighted").replace(
        "aggregation: equal_average", "aggregation: weighted",
    ) == weighted


class _RawTeacher(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(offset))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images[:, :1] + self.offset


def _teachers() -> TeacherEnsemble:
    return TeacherEnsemble([
        FrozenTeacher(_RawTeacher(offset), checkpoint_path="", allow_uninitialized_for_testing=True)
        for offset in (-1.0, 0.0, 1.0)
    ])


def test_frozen_wrapper_exposes_raw_logits_and_corrupts_before_sigmoid() -> None:
    teachers = _teachers()
    images = torch.tensor([0.25, -0.5]).reshape(2, 1, 1, 1).repeat(1, 3, 1, 1)
    raw = teachers.forward_logits(images)
    probabilities = teachers(images)
    for logits, probs in zip(raw, probabilities):
        torch.testing.assert_close(probs, torch.sigmoid(logits), rtol=0, atol=0)
    corrupt, noise = corrupt_teacher_logits(raw[2], ["source/a", "source/b"], 2026)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_teacher_noise_seed(2026, "source/a"))
    expected_noise = torch.randn(raw[2].shape[1:], generator=generator, dtype=torch.float32)
    assert torch.equal(noise[0], expected_noise)
    torch.testing.assert_close(corrupt, raw[2] + noise, rtol=0, atol=0)
    torch.testing.assert_close(torch.sigmoid(corrupt), torch.sigmoid(raw[2] + noise), rtol=0, atol=0)
    assert not torch.allclose(torch.sigmoid(corrupt), torch.sigmoid(raw[2]) + noise)
    assert not corrupt.requires_grad and not noise.requires_grad
    with pytest.raises(ValueError, match="std=1.0"):
        corrupt_teacher_logits(raw[2], ["source/a", "source/b"], 2026, std=0.5)


def test_stable_seed_and_noise_use_seed_plus_image_id_not_batch_order() -> None:
    expected = int.from_bytes(hashlib.sha256(b"2026\0source/a").digest()[:8], "big") % (2**63)
    assert stable_teacher_noise_seed(2026, "source/a") == expected
    logits = torch.zeros(3, 1, 4, 5)
    ids = ["source/a", "source/b", "source/c"]
    corrupted, ordered = corrupt_teacher_logits(logits, ids, 2026)
    reordered_logits, reordered = corrupt_teacher_logits(logits, [ids[2], ids[0], ids[1]], 2026)
    for index, image_id in enumerate(ids):
        _, alone = corrupt_teacher_logits(logits[:1], [image_id], 2026)
        assert torch.equal(ordered[index], alone[0])
    assert torch.equal(ordered[0], reordered[1])
    assert torch.equal(ordered[1], reordered[2])
    assert torch.equal(ordered[2], reordered[0])
    assert torch.equal(corrupted[0], reordered_logits[1])
    assert torch.equal(torch.sigmoid(corrupted[1]), torch.sigmoid(reordered_logits[2]))
    assert torch.equal(ordered, corrupt_teacher_logits(logits, ids, 2026)[1])
    assert not torch.equal(ordered, corrupt_teacher_logits(logits, ids, 2027)[1])
    assert not torch.equal(ordered[0], corrupt_teacher_logits(logits[:1], ["source/other"], 2026)[1][0])


def test_weighted_eq3_and_equal_average_on_identical_corrupted_inputs() -> None:
    image_id = ["source/a"]
    raw_first = torch.full((1, 1, 2, 2), torch.logit(torch.tensor(0.05)).item())
    raw_second = raw_first.clone()
    _, noise = corrupt_teacher_logits(torch.zeros_like(raw_first), image_id, 2026)
    raw_third = -noise  # After logit corruption the third teacher has p=0.5.
    logits = [raw_first, raw_second, raw_third]
    weighted, diag_weighted = teacher_robustness_posterior(
        logits, image_id, 2026, mode="weighted", tau=1.0, gamma=1.5,
    )
    equal, diag_equal = teacher_robustness_posterior(
        logits, image_id, 2026, mode="equal_average", tau=1.0, gamma=1.5,
    )
    for before, after in zip(diag_weighted["teacher_probs_after"], diag_equal["teacher_probs_after"]):
        assert torch.equal(before, after)
    stacked = torch.stack(diag_weighted["teacher_probs_after"])
    mean = stacked.mean(dim=0)
    entropy = binary_entropy(stacked)
    manual_weights = torch.softmax(-entropy - 1.5 * (stacked - mean).square(), dim=0)
    torch.testing.assert_close(diag_weighted["weights"], manual_weights)
    torch.testing.assert_close(weighted, (manual_weights * stacked).sum(dim=0))
    torch.testing.assert_close(equal, mean)
    assert torch.equal(diag_equal["weights"], torch.full_like(stacked, 1 / 3))
    assert (diag_weighted["weights"][2] < 1 / 3).all()
    assert diag_weighted["degraded_mean_weight"] < 1 / 3
    assert not torch.equal(weighted, equal)
    assert torch.equal(diag_weighted["teacher_probs_before"][2], torch.sigmoid(raw_third))
    assert torch.equal(diag_weighted["teacher_probs_after"][2], torch.full_like(raw_third, 0.5))


class _TinyStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, 3, padding=1)
        self.final = nn.Conv2d(4, 1, 1)

    def forward(self, images: torch.Tensor) -> SegmentationOutput:
        features = torch.tanh(self.encoder(images))
        return SegmentationOutput(logits=self.final(features), features=F.avg_pool2d(features, 4))


def _trainer_config(mode: str) -> dict:
    return {
        "seed": 2026,
        "teacher_pool": {"teachers": [{"name": name} for name in TEACHER_NAMES]},
        "training": {"mixed_precision": False},
        "method": {"use_crisp": True, "use_projector": True, "use_teachers": True,
                   "target_mode": "boundary_posterior", "use_amortization_loss": True},
        "crisp": {
            "teacher": {"tau": 1.0, "gamma": 1.5, "strict": True,
                        "teacher_names": TEACHER_NAMES,
                        "robustness": {"enabled": True, "degraded_teacher": "sammamba",
                                       "distribution": "gaussian", "logit_noise_std": 1.0,
                                       "seed_keys": ["seed", "dataset_name", "image_id"],
                                       "aggregation": mode}},
            "projection": {"beta": 0.35},
            "solver": {"newton_steps": 3, "bisection_steps": 12},
        },
    }


def test_trainer_robustness_is_label_independent_and_teachers_stay_frozen(monkeypatch) -> None:
    torch.manual_seed(17)
    student = _TinyStudent()
    projector = CRISPProjectorHead(4)
    teachers = _teachers()
    teachers.train()
    assert all(not param.requires_grad for param in teachers.parameters())
    assert all(not teacher.training and not teacher.model.training for teacher in teachers.teachers)
    trainer = Trainer(student, projector, teachers, _trainer_config("weighted"))
    batch = make_toy_batch(batch_size=2, image_size=32)
    batch["meta"] = {"dataset_name": ["source", "source"], "image_id": ["a", "b"]}
    captured = []
    diagnostics = []
    original = trainer_module.compute_boundary_posterior_target
    original_robustness = trainer_module.teacher_robustness_posterior

    def record_robustness(*args, **kwargs):
        posterior, diagnostic = original_robustness(*args, **kwargs)
        diagnostics.append((diagnostic["degraded_noise"].clone(), diagnostic["weights"].clone()))
        return posterior, diagnostic

    def record_target(mask, boundary_weight, teacher_posterior, *args, **kwargs):
        captured.append(teacher_posterior.detach().clone())
        return original(mask, boundary_weight, teacher_posterior, *args, **kwargs)

    monkeypatch.setattr(trainer_module, "compute_boundary_posterior_target", record_target)
    monkeypatch.setattr(trainer_module, "teacher_robustness_posterior", record_robustness)
    first = trainer.train_one_step(batch, epoch=0, step=0)
    changed = {**batch, "mask": 1.0 - batch["mask"]}
    second = trainer.train_one_step(changed, epoch=0, step=0)
    assert torch.equal(captured[0], captured[1])
    assert torch.equal(diagnostics[0][0], diagnostics[1][0])
    assert torch.equal(diagnostics[0][1], diagnostics[1][1])
    assert first.logs["teacher/degraded_mean_weight"] == second.logs["teacher/degraded_mean_weight"]
    first.loss.backward()
    assert all(param.grad is None for param in teachers.parameters())
    assert projector.conv1.weight.grad is not None
    with pytest.raises(ValueError, match="stable dataset_name/image_id"):
        trainer.train_one_step({"image": batch["image"], "mask": batch["mask"]}, epoch=0, step=0)
