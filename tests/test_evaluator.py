"""
Unit tests for evaluation-time CRISP invariants.

These tests protect against method drift in the inference protocol:
- inference must use p̃ = sigmoid(alpha_hat * z),
- projector-off must set alpha_hat = 1 at test time only,
- no teachers/solver are invoked during inference.
"""

import torch
import torch.nn as nn

from crisp.engine.evaluator import Evaluator
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
