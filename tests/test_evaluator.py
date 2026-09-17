"""
Unit tests for evaluation-time CRISP invariants.

These tests protect against method drift in the inference protocol:
- inference must use p̃ = sigmoid(alpha_hat * z),
- projector-off must set alpha_hat = 1 at test time only,
- no teachers/solver are invoked during inference.
"""

import math
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn as nn

import crisp.engine.evaluator as evaluator_module
from crisp.engine.evaluator import Evaluator
from crisp.metrics.calibration import boundary_support_mask
from crisp.models.base import SegmentationOutput


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder_channels = 4

    def forward(self, x: torch.Tensor) -> SegmentationOutput:
        # Deterministic logits = 2.0 everywhere, features arbitrary.
        B, _, H, W = x.shape
        logits = torch.full((B, 1, H, W), 2.0, device=x.device)
        features = torch.zeros((B, self.decoder_channels, H // 4, W // 4), device=x.device)
        return SegmentationOutput(logits=logits, features=features)


class _DummyProjector(nn.Module):
    def forward(self, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        # Deterministic alpha_hat = 0.5 everywhere (bounded).
        return torch.full_like(logits, 0.5)


class _ConstantLogitModel(nn.Module):
    def __init__(self, logit_value: float) -> None:
        super().__init__()
        self.logit_value = logit_value
        self.decoder_channels = 4

    def forward(self, x: torch.Tensor) -> SegmentationOutput:
        B, _, H, W = x.shape
        logits = torch.full((B, 1, H, W), self.logit_value, device=x.device)
        features = torch.zeros((B, self.decoder_channels, H // 4, W // 4), device=x.device)
        return SegmentationOutput(logits=logits, features=features)


class _StatefulModel(nn.Module):
    decoder_channels = 1

    def __init__(self, weight: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(weight))

    def forward(self, images: torch.Tensor) -> SegmentationOutput:
        logits = self.weight * images[:, :1]
        return SegmentationOutput(logits=logits, features=logits)


class _StatefulProjector(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(logits) * self.alpha


def _run_evaluation_script(
    monkeypatch: pytest.MonkeyPatch, config: dict, model: nn.Module, projector: nn.Module | None
) -> None:
    # Replace unavailable CLI and factory dependencies; checkpoint loading stays real.
    hydra = ModuleType("hydra")
    hydra.main = lambda **_kwargs: lambda function: function
    omegaconf = ModuleType("omegaconf")
    omegaconf.DictConfig = dict

    class OmegaConf:
        @staticmethod
        def to_container(cfg: dict, **_kwargs: object) -> dict:
            return cfg

    omegaconf.OmegaConf = OmegaConf
    registry = ModuleType("crisp.registry")
    registry.build_model = lambda _config: model
    registry.build_projector = lambda _config, in_channels: projector
    registry.get_model_decoder_channels = lambda current_model: current_model.decoder_channels
    registry.build_dataset = lambda _config, split: None
    monkeypatch.setitem(sys.modules, "hydra", hydra)
    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf)
    monkeypatch.setitem(sys.modules, "crisp.registry", registry)
    script = runpy.run_module("crisp.scripts.evaluate", run_name="crisp.scripts.evaluate_test")
    globals_ = script["main"].__globals__
    globals_["setup_logger"] = lambda _output_dir: None

    image = torch.zeros(3, 16, 16)
    image[0, :, :8] = -1.0
    image[0, :, 8:] = 1.0
    mask = (image[:1] >= 0).float()
    globals_["build_dataset"] = lambda _config, split: [{"image": image, "mask": mask}]
    globals_["main"](config)


def _script_config(tmp_path: Path, checkpoint_path: Path, use_projector: bool = True) -> dict:
    return {
        "seed": 7,
        "checkpoint": str(checkpoint_path),
        "eval_output_dir": str(tmp_path / "eval"),
        "method": {"use_crisp": use_projector, "use_projector": use_projector},
        "eval_datasets": ["toy"],
        "eval_data": {"toy": {"num_workers": 0, "pin_memory": False}},
        "eval": {"batch_size": 1},
    }


def _save_test_checkpoint(path: Path, projector_state: object = None, include_projector: bool = True) -> None:
    state = {"model_state_dict": _StatefulModel(2.0).state_dict()}
    if include_projector:
        state["projector_state_dict"] = projector_state
    torch.save(state, path)


def test_projector_checkpoint_valid_state_loads_and_same_checkpoint_masks_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "valid.pt"
    _save_test_checkpoint(checkpoint_path, _StatefulProjector(0.75).state_dict())
    model = _StatefulModel(-3.0)
    projector = _StatefulProjector(1.5)
    _run_evaluation_script(monkeypatch, _script_config(tmp_path, checkpoint_path), model, projector)

    assert model.weight.item() == pytest.approx(2.0)
    assert projector.alpha.item() == pytest.approx(0.75)
    assert (tmp_path / "eval" / "toy" / "projector_on.json").exists()
    assert (tmp_path / "eval" / "toy" / "projector_off.json").exists()

    image = torch.tensor([-1.0, 0.0, 1.0]).reshape(1, 1, 1, 3).repeat(1, 3, 1, 1)
    evaluator = Evaluator(model, projector, {})
    on = evaluator.predict_batch({"image": image}, projector_on=True)
    off = evaluator.predict_batch({"image": image}, projector_on=False)
    assert torch.allclose(on["alpha_hat"], torch.full_like(on["alpha_hat"], 0.75))
    assert torch.equal(off["alpha_hat"], torch.ones_like(off["alpha_hat"]))
    assert torch.equal(on["preds"], off["preds"])
    assert not torch.allclose(on["probs"], off["probs"])


@pytest.mark.parametrize(
    "case, projector_state, include_projector, error",
    [
        ("missing", None, False, "key is missing"),
        ("null", None, True, "projector_state_dict is None"),
        ("missing_parameter", {}, True, "Missing key"),
        ("unexpected_parameter", {"alpha": torch.tensor(0.75), "extra": torch.tensor(1.0)}, True, "Unexpected key"),
        ("wrong_shape", {"alpha": torch.ones(2)}, True, "size mismatch"),
    ],
)
def test_projector_checkpoint_invalid_state_fails_before_result_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    projector_state: object,
    include_projector: bool,
    error: str,
) -> None:
    checkpoint_path = tmp_path / f"{case}.pt"
    _save_test_checkpoint(checkpoint_path, projector_state, include_projector)
    with pytest.raises(ValueError, match=error) as exc:
        _run_evaluation_script(
            monkeypatch, _script_config(tmp_path, checkpoint_path), _StatefulModel(-3.0), _StatefulProjector(1.5)
        )
    assert "Projector-on CRISP evaluation" in str(exc.value)
    assert str(checkpoint_path) in str(exc.value)
    assert not (tmp_path / "eval" / "toy" / "projector_on.json").exists()
    assert not (tmp_path / "eval" / "summary.json").exists()


def test_projector_checkpoint_requires_module_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "valid.pt"
    _save_test_checkpoint(checkpoint_path, _StatefulProjector(0.75).state_dict())
    with pytest.raises(ValueError, match="requires a projector module"):
        _run_evaluation_script(monkeypatch, _script_config(tmp_path, checkpoint_path), _StatefulModel(-3.0), None)
    assert not (tmp_path / "eval" / "toy" / "projector_on.json").exists()


def test_baseline_checkpoint_without_projector_remains_evaluable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "baseline.pt"
    _save_test_checkpoint(checkpoint_path, include_projector=False)
    _run_evaluation_script(
        monkeypatch, _script_config(tmp_path, checkpoint_path, use_projector=False), _StatefulModel(-3.0), None
    )
    assert (tmp_path / "eval" / "toy" / "projector_off.json").exists()
    assert not (tmp_path / "eval" / "toy" / "projector_on.json").exists()


def test_projector_off_uses_valid_crisp_checkpoint_without_applying_alpha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "valid.pt"
    _save_test_checkpoint(checkpoint_path, _StatefulProjector(0.75).state_dict())
    config = _script_config(tmp_path, checkpoint_path)
    config["projector_off_only"] = True
    model = _StatefulModel(-3.0)
    projector = _StatefulProjector(1.5)
    _run_evaluation_script(monkeypatch, config, model, projector)
    assert projector.alpha.item() == pytest.approx(0.75)
    assert (tmp_path / "eval" / "toy" / "projector_off.json").exists()
    assert not (tmp_path / "eval" / "toy" / "projector_on.json").exists()


def test_projector_off_does_not_accept_incomplete_crisp_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "incomplete.pt"
    _save_test_checkpoint(checkpoint_path, include_projector=False)
    config = _script_config(tmp_path, checkpoint_path)
    config["projector_off_only"] = True
    with pytest.raises(ValueError, match="projector_state_dict.*missing"):
        _run_evaluation_script(monkeypatch, config, _StatefulModel(-3.0), _StatefulProjector(1.5))
    assert not (tmp_path / "eval" / "toy" / "projector_off.json").exists()


def test_projector_off_sets_alpha_one_and_preserves_logits() -> None:
    model = _DummyModel()
    projector = _DummyProjector()
    config = {"crisp": {"boundary": {"sigma_b": 6.0}, "projection": {"alpha_min": 0.5, "alpha_max": 1.75}}}
    ev = Evaluator(model=model, projector=projector, config=config)

    batch = {"image": torch.randn(2, 3, 32, 32), "mask": torch.zeros(2, 1, 32, 32)}
    out_on = ev.predict_batch(batch, projector_on=True)
    out_off = ev.predict_batch(batch, projector_on=False)

    # Logits must match backbone output (raw z) in both modes.
    assert torch.allclose(out_on["logits"], out_off["logits"])
    assert torch.allclose(out_on["logits"], torch.full_like(out_on["logits"], 2.0))

    # Projector-off forces alpha_hat = 1.
    assert torch.allclose(out_off["alpha_hat"], torch.ones_like(out_off["alpha_hat"]))

    # Projector-on uses projector alpha_hat.
    assert torch.allclose(out_on["alpha_hat"], torch.full_like(out_on["alpha_hat"], 0.5))

    # Probabilities must be sigmoid(alpha_hat * z).
    p_on_expected = torch.sigmoid(out_on["alpha_hat"] * out_on["logits"])
    p_off_expected = torch.sigmoid(out_off["alpha_hat"] * out_off["logits"])
    assert torch.allclose(out_on["probs"], p_on_expected)
    assert torch.allclose(out_off["probs"], p_off_expected)


def test_boundary_metrics_are_aggregated_globally_over_support() -> None:
    """Boundary calibration should aggregate selected pixels globally across images."""
    model = _ConstantLogitModel(logit_value=-0.8472978603872037)  # sigmoid -> 0.3
    config = {
        "crisp": {"boundary": {"sigma_b": 6.0}, "projection": {"alpha_min": 0.5, "alpha_max": 1.75}},
        "eval": {"boundary_support": {"top_percent": 20.0}, "ece": {"bins": 15}, "tace": {"threshold": 1.0e-3}},
    }
    ev = Evaluator(model=model, projector=None, config=config)

    batch = {
        "image": torch.randn(2, 3, 16, 16),
        "mask": torch.stack(
            [
                torch.zeros(1, 16, 16),
                torch.ones(1, 16, 16),
            ],
            dim=0,
        ),
    }
    metrics = ev.evaluate_dataset([batch], "toy", projector_on=False)
    assert abs(metrics["bece"] - 0.2) < 1e-6


def test_metric_export_contains_thesis_aliases() -> None:
    model = _ConstantLogitModel(logit_value=3.0)
    config = {
        "crisp": {"boundary": {"sigma_b": 6.0}, "projection": {"alpha_min": 0.5, "alpha_max": 1.75}},
        "eval": {"boundary_support": {"top_percent": 20.0}, "ece": {"bins": 15}, "tace": {"threshold": 1.0e-3}},
    }
    ev = Evaluator(model=model, projector=None, config=config)
    batch = {
        "image": torch.randn(1, 3, 16, 16),
        "mask": torch.ones(1, 1, 16, 16),
    }

    metrics = ev.evaluate_dataset([batch], "toy", projector_on=False)

    for key in ["mDice", "mIoU", "B-F1", "HD95", "bECE", "off-bECE"]:
        assert key in metrics
    assert metrics["mDice"] == metrics["dice"]
    assert metrics["mIoU"] == metrics["iou"]
    assert metrics["B-F1"] == metrics["boundary_f1"]
    assert metrics["HD95"] == metrics["hd95"]
    assert metrics["bECE"] == metrics["bece"]
    assert metrics["off-bECE"] == metrics["off_bece"]
    assert "boundary_nll" in metrics
    assert "boundary_brier" in metrics


def test_boundary_proper_scores_are_invariant_to_dataloader_batch_splitting() -> None:
    class ImageLogits(nn.Module):
        def forward(self, images: torch.Tensor) -> SegmentationOutput:
            logits = images[:, :1]
            return SegmentationOutput(logits=logits, features=logits)

    logits = torch.linspace(-2.0, 2.0, 3 * 16 * 16).reshape(3, 1, 16, 16)
    images = logits.repeat(1, 3, 1, 1)
    masks = torch.zeros_like(logits)
    masks[0, :, 3:10, 4:11] = 1
    masks[1, :, 6:15, 2:12] = 1
    masks[2, :, 1:8, 8:15] = 1
    evaluator = Evaluator(ImageLogits(), None, {})

    one_batch = evaluator.evaluate_dataset(
        [{"image": images, "mask": masks}], "toy", projector_on=False,
    )
    split_batches = evaluator.evaluate_dataset(
        [
            {"image": images[:1], "mask": masks[:1]},
            {"image": images[1:], "mask": masks[1:]},
        ],
        "toy",
        projector_on=False,
    )
    for key in ("boundary_nll", "boundary_brier", "nll", "brier", "ece", "bece"):
        assert one_batch[key] == pytest.approx(split_batches[key], abs=1e-7)


def test_bece_support_sensitivity_does_not_change_boundary_proper_scores(monkeypatch) -> None:
    class ImageLogits(nn.Module):
        def forward(self, images: torch.Tensor) -> SegmentationOutput:
            logits = images[:, :1]
            return SegmentationOutput(logits=logits, features=logits)

    probs = torch.full((1, 1, 2, 5), 0.1)
    probs.reshape(-1)[0] = 0.9
    probs.reshape(-1)[-1] = 0.9
    labels = torch.zeros_like(probs)
    labels.reshape(-1)[0] = 1.0
    wb = torch.arange(10, dtype=torch.float32).reshape_as(probs)
    assert boundary_support_mask(wb).reshape(-1).nonzero().reshape(-1).tolist() == [8, 9]
    monkeypatch.setattr(evaluator_module, "compute_boundary_weight", lambda *_args, **_kwargs: wb)
    batch = {"image": torch.logit(probs).repeat(1, 3, 1, 1), "mask": labels}

    def evaluate(top_percent: float) -> dict[str, float]:
        config = {"eval": {"boundary_support": {"top_percent": top_percent}}}
        return Evaluator(ImageLogits(), None, config).evaluate_dataset(
            [batch], "toy", projector_on=False,
        )

    at_10, at_20, at_30 = (evaluate(value) for value in (10.0, 20.0, 30.0))
    assert at_10["bece"] != pytest.approx(at_30["bece"])
    for result in (at_10, at_20, at_30):
        assert result["boundary_nll"] == pytest.approx((-math.log(0.9) - math.log(0.1)) / 2)
        assert result["boundary_brier"] == pytest.approx(0.41)
    for key in ("boundary_nll", "boundary_brier", "ece", "nll", "brier"):
        assert at_10[key] == pytest.approx(at_30[key])
