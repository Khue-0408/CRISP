"""
Unit tests for the CRISP projector head.
"""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from crisp.models.projector_head import CRISPProjectorHead
from crisp.modules.calibration import calibrate_logits_with_alpha
from crisp.registry import build_projector


def test_canonical_projector_structure() -> None:
    projector = CRISPProjectorHead(feature_channels=4)
    convs = [module for module in projector.modules() if isinstance(module, nn.Conv2d)]
    norms = [module for module in projector.modules() if isinstance(module, nn.GroupNorm)]
    activations = [module for module in projector.modules() if isinstance(module, nn.GELU)]

    assert [(layer.in_channels, layer.out_channels, layer.kernel_size) for layer in convs] == [
        (5, 64, (3, 3)), (64, 64, (3, 3)), (64, 1, (1, 1)),
    ]
    assert [(norm.num_groups, norm.num_channels) for norm in norms] == [(8, 64), (8, 64)]
    assert len(activations) == 2
    assert list(projector.children()) == [
        projector.conv1, projector.norm1, projector.act1,
        projector.conv2, projector.norm2, projector.act2, projector.head,
    ]
    assert sum(parameter.numel() for parameter in projector.parameters()) == 40065


def test_first_convolution_receives_features_and_resized_raw_logit() -> None:
    projector = CRISPProjectorHead(feature_channels=4)
    features = torch.randn(1, 4, 8, 8)
    logits = torch.linspace(-4.0, 4.0, 32 * 32).reshape(1, 1, 32, 32)
    captured = []
    hook = projector.conv1.register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0]))
    try:
        alpha = projector(features, logits)
    finally:
        hook.remove()

    assert captured[0].shape == (1, 5, 8, 8)
    torch.testing.assert_close(captured[0][:, :4], features)
    torch.testing.assert_close(
        captured[0][:, 4:],
        F.interpolate(logits, size=(8, 8), mode="bilinear", align_corners=False),
    )
    assert alpha.shape == (1, 1, 32, 32)
    assert torch.isfinite(alpha).all()
    assert (alpha >= 0.5).all() and (alpha <= 1.75).all() and (alpha > 0).all()


def test_alpha_is_mapped_on_feature_grid_before_bilinear_upsampling() -> None:
    projector = CRISPProjectorHead(feature_channels=4)
    captured = []
    hook = projector.head.register_forward_hook(lambda _module, _inputs, output: captured.append(output))
    try:
        alpha = projector(torch.randn(1, 4, 8, 8), torch.randn(1, 1, 32, 32))
    finally:
        hook.remove()

    assert captured[0].shape == (1, 1, 8, 8)
    alpha_low = 0.5 + 1.25 * torch.sigmoid(captured[0])
    torch.testing.assert_close(
        alpha, F.interpolate(alpha_low, size=(32, 32), mode="bilinear", align_corners=False)
    )


@pytest.mark.parametrize("score", [-12.0, 0.0, 12.0])
def test_alpha_score_mapping(score: float) -> None:
    projector = CRISPProjectorHead(feature_channels=4)
    with torch.no_grad():
        projector.head.weight.zero_()
        projector.head.bias.fill_(score)
    alpha = projector(torch.zeros(1, 4, 8, 8), torch.zeros(1, 1, 32, 32))
    expected = 0.5 + 1.25 * torch.sigmoid(torch.tensor(score))
    torch.testing.assert_close(alpha, torch.full_like(alpha, expected.item()))
    assert (alpha >= 0.5).all() and (alpha <= 1.75).all()
    if score == 0.0:
        assert expected.item() == (0.5 + 1.75) / 2


def test_projector_and_raw_logit_gradient_path() -> None:
    torch.manual_seed(7)
    projector = CRISPProjectorHead(feature_channels=4)
    features = torch.randn(1, 4, 8, 8, requires_grad=True)
    logits = torch.randn(1, 1, 32, 32, requires_grad=True)
    alpha = projector(features, logits)
    alpha.mean().backward()
    assert features.grad is not None and features.grad.abs().sum() > 0
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert all(parameter.grad is not None for parameter in projector.parameters())


def test_calibration_uses_original_full_resolution_raw_logit() -> None:
    projector = CRISPProjectorHead(feature_channels=4)
    features = torch.randn(1, 4, 8, 8)
    logits = torch.arange(32 * 32, dtype=torch.float32).reshape(1, 1, 32, 32) / 128 - 4
    alpha = projector(features, logits)
    probabilities = calibrate_logits_with_alpha(logits, alpha)
    torch.testing.assert_close(probabilities, torch.sigmoid(alpha * logits))
    assert torch.equal(probabilities >= 0.5, logits >= 0)


def test_default_config_builds_manuscript_projector() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs/crisp/default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    projector = build_projector(config, in_channels=4)
    assert isinstance(projector, CRISPProjectorHead)
    assert projector.conv1.out_channels == projector.conv2.out_channels == 64
    assert projector.norm1.num_groups == projector.norm2.num_groups == 8
    assert (projector.alpha_min, projector.alpha_max) == (0.5, 1.75)


def test_head_width_and_group_config_are_consumed() -> None:
    projector = build_projector(
        {"projector_head": {"hidden_channels": 128, "num_groups": 8}}, in_channels=4
    )
    assert projector.conv1.out_channels == projector.conv2.out_channels == 128
    assert projector.norm1.num_groups == projector.norm2.num_groups == 8
    with pytest.raises(ValueError, match="divisible"):
        build_projector(
            {"projector_head": {"hidden_channels": 65, "num_groups": 8}}, in_channels=4
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [("num_conv_layers", 3), ("activation", "relu"), ("upsample_mode", "nearest"),
     ("norm", "batchnorm")],
)
def test_unsupported_head_config_fails_loudly(key: str, value: object) -> None:
    with pytest.raises(ValueError, match=key if key != "norm" else "GroupNorm"):
        build_projector({"projector_head": {key: value}}, in_channels=4)


def test_projector_output_in_range() -> None:
    """Ensure alpha_hat lies within [alpha_min, alpha_max]."""
    alpha_min, alpha_max = 0.5, 1.8
    projector = CRISPProjectorHead(
        feature_channels=32,
        hidden_channels=16,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )

    features = torch.randn(2, 32, 22, 22)
    logits = torch.randn(2, 1, 88, 88)

    alpha_hat = projector(features, logits)
    assert alpha_hat.shape == (2, 1, 88, 88), f"Shape mismatch: {alpha_hat.shape}"
    assert alpha_hat.min() >= alpha_min - 1e-5, f"Below alpha_min: {alpha_hat.min()}"
    assert alpha_hat.max() <= alpha_max + 1e-5, f"Above alpha_max: {alpha_hat.max()}"


@pytest.mark.parametrize(
    ("batch", "feature_hw", "logit_hw"),
    [(1, (8, 8), (32, 32)), (1, (11, 11), (44, 44)), (2, (9, 13), (36, 52))],
)
def test_projector_output_matches_raw_logit_grid(
    batch: int, feature_hw: tuple[int, int], logit_hw: tuple[int, int]
) -> None:
    projector = CRISPProjectorHead(feature_channels=16, hidden_channels=8)
    features = torch.randn(batch, 16, *feature_hw)
    logits = torch.randn(batch, 1, *logit_hw)

    alpha = projector(features, logits)
    probabilities = calibrate_logits_with_alpha(logits, alpha)
    assert alpha.shape == logits.shape == probabilities.shape
    assert (alpha >= projector.alpha_min).all() and (alpha <= projector.alpha_max).all()
    torch.testing.assert_close(probabilities, torch.sigmoid(alpha * logits))


def test_projector_has_no_independent_output_size_override() -> None:
    projector = CRISPProjectorHead(feature_channels=16, hidden_channels=8)
    features = torch.randn(1, 16, 11, 11)
    logits = torch.randn(1, 1, 44, 44)

    with pytest.raises(TypeError, match="output_size"):
        projector(features, logits, output_size=(64, 64))
    with pytest.raises(TypeError):
        projector(features, logits, (64, 64))


@pytest.mark.parametrize(
    ("feature_shape", "logit_shape", "error"),
    [
        ((1, 16, 11), (1, 1, 44, 44), "4D"),
        ((1, 16, 11, 11), (1, 1, 44), "4D"),
        ((1, 16, 11, 11), (1, 2, 44, 44), "one foreground channel"),
        ((2, 16, 11, 11), (1, 1, 44, 44), "same batch size"),
    ],
)
def test_projector_rejects_incompatible_input_shapes(
    feature_shape: tuple[int, ...], logit_shape: tuple[int, ...], error: str
) -> None:
    projector = CRISPProjectorHead(feature_channels=16, hidden_channels=8)
    features = torch.randn(*feature_shape)
    logits = torch.randn(*logit_shape)

    with pytest.raises(ValueError, match=error):
        projector(features, logits)


def test_projector_gradients_flow() -> None:
    """Alpha_hat must keep gradients for backpropagation."""
    projector = CRISPProjectorHead(feature_channels=16, hidden_channels=8)
    features = torch.randn(1, 16, 11, 11, requires_grad=True)
    logits = torch.randn(1, 1, 44, 44, requires_grad=True)

    alpha = projector(features, logits)
    loss = alpha.mean()
    loss.backward()
    assert features.grad is not None
    assert logits.grad is not None
