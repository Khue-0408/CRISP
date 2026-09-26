"""PraNet native side supervision in baseline and CRISP training."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import crisp.engine.trainer as trainer_module
from crisp.engine.trainer import Trainer
from crisp.models.base import SegmentationOutput
from crisp.models.pranet import PraNet
from crisp.models.projector_head import CRISPProjectorHead
from crisp.models.unet import UNet
from crisp.models.unetpp import UNetPP
from crisp.modules.losses import (
    baseline_bce_dice_loss,
    pranet_native_side_losses,
    pranet_native_structure_loss,
)
from crisp.modules.margin_label_smoothing import margin_label_smoothing_penalty


def _retained_structure_loss():
    """Load only the function body from the retained MyTrain.py, not its script imports."""
    path = Path(__file__).resolve().parents[1] / "1_baseline/PraNet/MyTrain.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "structure_loss"
    )
    scope = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    return scope["structure_loss"]


@pytest.mark.parametrize("mask_kind", ["empty", "region", "boundary"])
def test_native_loss_matches_retained_source(mask_kind: str) -> None:
    mask = torch.zeros(2, 1, 32, 32)
    if mask_kind == "region":
        mask[:, :, 8:24, 8:24] = 1
    elif mask_kind == "boundary":
        mask[:, :, 4:28:4, 4:28] = 1
    pred = torch.linspace(-2, 2, mask.numel()).reshape_as(mask)
    with pytest.warns(UserWarning, match="size_average and reduce"):
        retained = _retained_structure_loss()(pred, mask)
    torch.testing.assert_close(pranet_native_structure_loss(pred, mask), retained,
                               rtol=1e-6, atol=1e-7)


def _aux_maps(mask: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "lateral_map_5": torch.full_like(mask, -2.0),
        "lateral_map_4": torch.full_like(mask, 0.25),
        "lateral_map_3": torch.full_like(mask, 1.5),
        "lateral_map_2": torch.full_like(mask, 4.0),
    }


def test_native_side_weights_sum_and_exclude_final_map() -> None:
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 7:25, 9:23] = 1
    aux = _aux_maps(mask)
    result = pranet_native_side_losses(aux, mask)
    individual = [pranet_native_structure_loss(aux[key], mask) for key in
                  ("lateral_map_5", "lateral_map_4", "lateral_map_3")]
    torch.testing.assert_close(result["native_aux_loss"], sum(individual))
    assert not torch.isclose(result["native_aux_loss"], sum(individual) / 3)
    aux["lateral_map_2"] = torch.full_like(mask, -7.0)
    torch.testing.assert_close(pranet_native_side_losses(aux, mask)["native_aux_loss"],
                               result["native_aux_loss"], rtol=0, atol=0)


@pytest.mark.parametrize("change", ["missing", "unexpected"])
def test_native_side_structure_fails_loudly(change: str) -> None:
    mask = torch.zeros(1, 1, 32, 32)
    aux = _aux_maps(mask)
    if change == "missing":
        del aux["lateral_map_4"]
    else:
        aux["unknown"] = torch.zeros_like(mask)
    with pytest.raises(ValueError, match="exactly"):
        pranet_native_side_losses(aux, mask)


class _TinyPraNet(PraNet):
    """PraNet host identity with inexpensive controlled native side maps."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self._decoder_channels = 4
        self.encoder = nn.Conv2d(3, 4, 3, padding=1)
        self.final = nn.Conv2d(4, 1, 1)
        self.side5 = nn.Conv2d(4, 1, 1)
        self.side4 = nn.Conv2d(4, 1, 1)
        self.side3 = nn.Conv2d(4, 1, 1)
        self.side_shift = 0.0

    def forward(self, x: torch.Tensor) -> SegmentationOutput:
        full_feature = torch.tanh(self.encoder(x))
        logits = self.final(full_feature)
        return SegmentationOutput(
            logits=logits,
            features=F.avg_pool2d(full_feature, 4),
            aux={
                "lateral_map_5": self.side5(full_feature) + self.side_shift,
                "lateral_map_4": self.side4(full_feature),
                "lateral_map_3": self.side3(full_feature),
                "lateral_map_2": logits,
            },
        )


class _OtherHost(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core = _TinyPraNet()

    def forward(self, x: torch.Tensor) -> SegmentationOutput:
        return self.core(x)


def _config(use_crisp: bool) -> dict:
    return {
        "training": {"epochs": 2, "optimizer": "adamw", "scheduler": "none",
                     "mixed_precision": False},
        "method": {"use_crisp": use_crisp, "use_projector": use_crisp,
                   "use_teachers": False, "target_mode": "hard_label",
                   "use_amortization_loss": use_crisp},
        "crisp": {
            "schedule": {"enabled": True, "phase_i_epochs": 1, "phase_ii_epochs": 1,
                         "phase_iii_epochs": 0, "phase_ii_ramp_epochs": 1},
            "projection": {"lambda": 1.0, "mu": 0.25, "beta": 0.35,
                           "eta_dice": 0.5, "alpha_min": 0.5, "alpha_max": 1.75},
            "solver": {"newton_steps": 3, "bisection_steps": 12},
        },
    }


@pytest.fixture
def controlled_batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(21)
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 8:24, 9:23] = 1
    return {"image": torch.randn(1, 3, 32, 32), "mask": mask}


def test_pranet_baseline_and_phase_i_share_native_side_loss(controlled_batch) -> None:
    model = _TinyPraNet()
    baseline = Trainer(model, None, None, _config(False))
    phase_i = Trainer(model, CRISPProjectorHead(4), None, _config(True))
    with torch.no_grad():
        output = model(controlled_batch["image"])
        final = baseline_bce_dice_loss(output.logits, controlled_batch["mask"])["loss"]
        side = pranet_native_side_losses(output.aux, controlled_batch["mask"])["native_aux_loss"]

    baseline_step = baseline.train_one_step(controlled_batch, epoch=0, step=0)
    phase_i_step = phase_i.train_one_step(controlled_batch, epoch=0, step=0)
    torch.testing.assert_close(baseline_step.loss, final + side)
    torch.testing.assert_close(phase_i_step.loss, final + side)
    assert baseline_step.logs["native_aux_loss"] == pytest.approx(phase_i_step.logs["native_aux_loss"])
    assert baseline_step.logs["final_loss"] == pytest.approx(phase_i_step.logs["final_loss"])
    baseline_step.loss.backward()
    assert model.side5.weight.grad is not None and model.side5.weight.grad.abs().sum() > 0
    assert model.side4.weight.grad is not None and model.side4.weight.grad.abs().sum() > 0
    assert model.side3.weight.grad is not None and model.side3.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    phase_i_step.loss.backward()
    assert model.side5.weight.grad is not None and model.side5.weight.grad.abs().sum() > 0


def test_pranet_margin_control_preserves_native_aux_and_penalizes_final_only(
    controlled_batch,
) -> None:
    model = _TinyPraNet()
    baseline = Trainer(model, None, None, _config(False))
    control_config = _config(False)
    control_config["method"]["name"] = "margin_label_smoothing"
    control_config["calibration_control"] = {
        "name": "margin_label_smoothing",
        "margin": 10.0,
        "weight": 0.1,
    }
    control = Trainer(model, None, None, control_config)

    with torch.no_grad():
        output = model(controlled_batch["image"])
        final_penalty = margin_label_smoothing_penalty(output.logits)
    baseline_step = baseline.train_one_step(controlled_batch, epoch=0, step=0)
    control_step = control.train_one_step(controlled_batch, epoch=0, step=0)

    torch.testing.assert_close(
        control_step.loss,
        baseline_step.loss + 0.1 * final_penalty,
    )
    assert control_step.logs["native_aux_loss"] == pytest.approx(
        baseline_step.logs["native_aux_loss"]
    )
    for key in ("lateral_map_5", "lateral_map_4", "lateral_map_3"):
        assert control_step.logs[f"native_aux/{key}"] == pytest.approx(
            baseline_step.logs[f"native_aux/{key}"]
        )
    assert control_step.logs["margin_penalty"] == pytest.approx(final_penalty.item())

    model.side_shift = 1.0
    shifted = control.train_one_step(controlled_batch, epoch=0, step=0)
    assert shifted.logs["margin_penalty"] == pytest.approx(
        control_step.logs["margin_penalty"]
    )
    assert shifted.logs["native_aux_loss"] != pytest.approx(
        control_step.logs["native_aux_loss"]
    )


def test_active_crisp_adds_same_side_loss_without_changing_final_projection(
    controlled_batch, monkeypatch
) -> None:
    model = _TinyPraNet()
    projector = CRISPProjectorHead(4)
    trainer = Trainer(model, projector, None, _config(True))
    baseline = Trainer(model, None, None, _config(False))
    baseline_step = baseline.train_one_step(controlled_batch, epoch=0, step=0)
    solver_inputs = []
    projector_logits = []
    original_solver = trainer_module.solve_alpha_star

    def record_solver(logits, target, *args, **kwargs):
        solver_inputs.append((logits.detach().clone(), target.detach().clone()))
        return original_solver(logits, target, *args, **kwargs)

    monkeypatch.setattr(trainer_module, "solve_alpha_star", record_solver)
    hook = projector.register_forward_pre_hook(
        lambda _module, inputs: projector_logits.append(inputs[1].detach().clone())
    )
    try:
        first = trainer.train_one_step(controlled_batch, epoch=1, step=0)
        model.side_shift = 1.0
        second = trainer.train_one_step(controlled_batch, epoch=1, step=0)
    finally:
        hook.remove()

    assert len(solver_inputs) == len(projector_logits) == 2
    for before, after in zip(solver_inputs[0], solver_inputs[1]):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    torch.testing.assert_close(projector_logits[0], projector_logits[1], rtol=0, atol=0)
    assert first.logs["task_loss"] == pytest.approx(second.logs["task_loss"])
    assert first.logs["native_aux_loss"] == pytest.approx(baseline_step.logs["native_aux_loss"])
    assert first.logs["amort_loss"] == pytest.approx(second.logs["amort_loss"])
    assert first.logs["final_loss"] == pytest.approx(second.logs["final_loss"])
    assert first.logs["native_aux_loss"] != pytest.approx(second.logs["native_aux_loss"])
    assert first.logs["loss"] == pytest.approx(first.logs["final_loss"] + first.logs["native_aux_loss"])
    assert second.logs["loss"] == pytest.approx(second.logs["final_loss"] + second.logs["native_aux_loss"])
    second.loss.backward()
    assert model.side5.weight.grad is not None and model.side5.weight.grad.abs().sum() > 0
    assert projector.conv1.weight.grad is not None


def test_pranet_matched_beta0_preserves_native_side_loss(controlled_batch) -> None:
    torch.manual_seed(31)
    model = _TinyPraNet()
    projector = CRISPProjectorHead(4)
    full_config = _config(True)
    beta0_config = _config(True)
    beta0_config["crisp"]["projection"]["beta"] = 0.0
    full = Trainer(model, projector, None, full_config)
    matched = Trainer(model, projector, None, beta0_config)

    full_step = full.train_one_step(controlled_batch, epoch=1, step=0)
    beta0_step = matched.train_one_step(controlled_batch, epoch=1, step=0)

    assert full.use_amortization_loss and matched.use_amortization_loss
    assert full_step.logs["native_aux_loss"] == pytest.approx(beta0_step.logs["native_aux_loss"])
    assert full_step.logs["task_loss"] == pytest.approx(beta0_step.logs["task_loss"])
    assert full_step.logs["amort_loss"] == pytest.approx(beta0_step.logs["amort_loss"])
    torch.testing.assert_close(
        full_step.loss - beta0_step.loss,
        0.35 * torch.as_tensor(beta0_step.logs["amort_loss"]),
    )
    assert beta0_step.logs["loss"] == pytest.approx(
        beta0_step.logs["final_loss"] + beta0_step.logs["native_aux_loss"]
    )
    beta0_step.loss.backward()
    assert projector.conv1.weight.grad is not None
    assert projector.conv1.weight.grad.abs().sum() > 0
    assert model.side5.weight.grad is not None
    assert model.side5.weight.grad.abs().sum() > 0


def test_non_pranet_host_does_not_inherit_native_side_loss(controlled_batch) -> None:
    model = _OtherHost()
    trainer = Trainer(model, None, None, _config(False))
    with torch.no_grad():
        final = baseline_bce_dice_loss(model(controlled_batch["image"]).logits,
                                       controlled_batch["mask"])["loss"]
    result = trainer.train_one_step(controlled_batch, epoch=0, step=0)
    torch.testing.assert_close(result.loss, final)
    assert "native_aux_loss" not in result.logs


@pytest.mark.parametrize(
    ("model_class", "kwargs"), [(UNet, {"base_channels": 8}), (UNetPP, {})]
)
def test_unet_hosts_keep_their_final_only_baseline_loss(
    controlled_batch, model_class, kwargs
) -> None:
    model = model_class(**kwargs).eval()
    trainer = Trainer(model, None, None, _config(False))
    with torch.no_grad():
        final = baseline_bce_dice_loss(
            model(controlled_batch["image"]).logits, controlled_batch["mask"]
        )["loss"]
    result = trainer.train_one_step(controlled_batch, epoch=0, step=0)
    torch.testing.assert_close(result.loss, final)
    assert "native_aux_loss" not in result.logs
