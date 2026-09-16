"""PraNet's parameter-free CRISP feature-grid adapter contract."""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from crisp.models.pranet import PraNet
from crisp.models.unet import UNet
from crisp.models.unetpp import UNetPP
from crisp.modules.calibration import calibrate_logits_with_alpha
from crisp.registry import build_projector, get_model_decoder_channels


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def canonical_pranet_pass():
    model = PraNet().eval()
    image = torch.randn(1, 3, 352, 352)
    native_features = []
    hook = model.model.rfb2_1.register_forward_hook(
        lambda _module, _inputs, output: native_features.append(output)
    )
    try:
        with torch.no_grad():
            native_outputs = model.model(image)
            output = model(image)
    finally:
        hook.remove()
    return model, image, native_features[-1], native_outputs, output


def test_pranet_quarter_grid_is_bilinear_resized_native_feature(canonical_pranet_pass) -> None:
    model, image, native_feature, _, output = canonical_pranet_pass
    assert image.shape == (1, 3, 352, 352)
    assert native_feature.shape == (1, 32, 44, 44)
    assert output.logits.shape == (1, 1, 352, 352)
    assert output.features.shape == (1, 32, 88, 88)
    assert get_model_decoder_channels(model) == output.features.shape[1] == 32
    torch.testing.assert_close(
        output.features,
        F.interpolate(native_feature, size=(88, 88), mode="bilinear", align_corners=False),
        rtol=0,
        atol=0,
    )


def test_pranet_prediction_and_aux_match_native_backbone(canonical_pranet_pass) -> None:
    model, _, _, native_outputs, output = canonical_pranet_pass
    assert len(native_outputs) == 4
    assert list(output.aux) == [
        "lateral_map_5", "lateral_map_4", "lateral_map_3", "lateral_map_2",
    ]
    for key, native in zip(output.aux, native_outputs):
        assert output.aux[key].shape == (1, 1, 352, 352)
        torch.testing.assert_close(output.aux[key], native, rtol=0, atol=0)
    torch.testing.assert_close(output.logits, native_outputs[-1], rtol=0, atol=0)

    wrapper_shapes = {key: value.shape for key, value in model.state_dict().items()}
    native_shapes = {key: value.shape for key, value in model.model.state_dict().items()}
    assert wrapper_shapes == native_shapes
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in model.model.parameters()
    )


def test_pranet_projector_runs_on_quarter_grid(canonical_pranet_pass) -> None:
    model, _, _, _, output = canonical_pranet_pass
    config_path = Path(__file__).resolve().parents[1] / "configs/crisp/default.yaml"
    projector = build_projector(
        yaml.safe_load(config_path.read_text(encoding="utf-8")),
        in_channels=get_model_decoder_channels(model),
    ).eval()
    first_conv_inputs = []
    hook = projector.conv1.register_forward_pre_hook(
        lambda _module, inputs: first_conv_inputs.append(inputs[0])
    )
    try:
        with torch.no_grad():
            alpha = projector(output.features, output.logits)
            probability = calibrate_logits_with_alpha(output.logits, alpha)
    finally:
        hook.remove()
    assert first_conv_inputs[0].shape == (1, 33, 88, 88)
    assert alpha.shape == probability.shape == output.logits.shape == (1, 1, 352, 352)
    torch.testing.assert_close(probability, torch.sigmoid(alpha * output.logits))


def test_pranet_feature_resize_preserves_gradient_to_native_tap() -> None:
    model = PraNet().eval()
    native_features = []
    hook = model.model.rfb2_1.register_forward_hook(
        lambda _module, _inputs, output: native_features.append(output)
    )
    try:
        output = model(torch.randn(1, 3, 64, 64))
    finally:
        hook.remove()
    native_features[0].retain_grad()
    output.features.square().mean().backward()
    assert native_features[0].grad is not None
    assert native_features[0].grad.abs().sum() > 0
    assert model.model.rfb2_1.branch0[0].conv.weight.grad is not None


def test_pranet_quarter_grid_is_dynamic_for_divisible_input() -> None:
    model = PraNet().eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 320, 384))
    assert output.logits.shape == (1, 1, 320, 384)
    assert output.features.shape == (1, 32, 80, 96)


@pytest.mark.parametrize(
    ("model_class", "kwargs"), [(UNet, {"base_channels": 8}), (UNetPP, {})]
)
def test_other_hosts_still_expose_quarter_grid(model_class, kwargs) -> None:
    model = model_class(**kwargs).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 64, 64))
    assert output.logits.shape == (1, 1, 64, 64)
    assert output.features.shape[-2:] == (16, 16)
    assert output.features.shape[1] == model.decoder_channels
