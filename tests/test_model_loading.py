"""
Unit tests for robust model checkpoint loading.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from crisp.utils.model_loading import load_model_checkpoint


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=1)


def test_load_model_checkpoint_handles_nested_state_dict_and_prefixes(tmp_path: Path) -> None:
    model = _TinyNet()
    reference = _TinyNet()
    checkpoint_path = tmp_path / "nested_prefixed.pt"
    prefixed = {
        f"module.model.{key}": value.clone()
        for key, value in reference.state_dict().items()
    }
    torch.save({"state_dict": prefixed}, checkpoint_path)

    load_model_checkpoint(model, checkpoint_path)

    for key, value in reference.state_dict().items():
        assert torch.allclose(model.state_dict()[key], value)


def test_load_model_checkpoint_raises_when_no_matching_keys_exist(tmp_path: Path) -> None:
    model = _TinyNet()
    checkpoint_path = tmp_path / "bad.pt"
    torch.save({"state_dict": {"totally.wrong.weight": torch.randn(1)}}, checkpoint_path)

    try:
        load_model_checkpoint(model, checkpoint_path, strict=False)
    except RuntimeError as exc:
        assert "does not contain any parameter keys matching" in str(exc)
    else:
        raise AssertionError("Expected a clear error when checkpoint keys do not match the model.")


def test_load_model_checkpoint_can_auto_download_from_configured_url(tmp_path: Path) -> None:
    reference = _TinyNet()
    target = _TinyNet()
    for param in target.parameters():
        param.data.zero_()

    source_checkpoint = tmp_path / "source.pt"
    destination_checkpoint = tmp_path / "downloads" / "auto.pt"
    torch.save({"model_state_dict": reference.state_dict()}, source_checkpoint)

    load_model_checkpoint(
        target,
        destination_checkpoint,
        auto_download=True,
        download_url=source_checkpoint.as_uri(),
    )

    assert destination_checkpoint.exists()
    for key, value in reference.state_dict().items():
        assert torch.allclose(target.state_dict()[key], value)


class _TinyLegacyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inc = nn.Module()
        self.inc.block = nn.Sequential(nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False))
        self.down1 = nn.Module()
        self.down1.pool_conv = nn.Sequential(
            nn.Identity(),
            nn.Module(),
        )
        self.down1.pool_conv[1].block = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False)
        )
        self.out_conv = nn.Conv2d(4, 1, kernel_size=1)


def test_load_model_checkpoint_remaps_legacy_unet_keys(tmp_path: Path) -> None:
    model = _TinyLegacyUNet()
    reference = _TinyLegacyUNet()
    checkpoint_path = tmp_path / "legacy_unet.pt"
    torch.save(
        {
            "state_dict": {
                "inc.double_conv.0.weight": reference.inc.block[0].weight.clone(),
                "down1.maxpool_conv.1.double_conv.0.weight": (
                    reference.down1.pool_conv[1].block[0].weight.clone()
                ),
                "outc.conv.weight": reference.out_conv.weight.clone(),
                "outc.conv.bias": reference.out_conv.bias.clone(),
            }
        },
        checkpoint_path,
    )

    load_model_checkpoint(model, checkpoint_path, strict=False)

    assert torch.allclose(model.inc.block[0].weight, reference.inc.block[0].weight)
    assert torch.allclose(
        model.down1.pool_conv[1].block[0].weight,
        reference.down1.pool_conv[1].block[0].weight,
    )
    assert torch.allclose(model.out_conv.weight, reference.out_conv.weight)
    assert torch.allclose(model.out_conv.bias, reference.out_conv.bias)
