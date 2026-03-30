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
