"""Baseline-backed UACANet adapter."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from crisp.models.base import BaseSegmentationModel, SegmentationOutput
from crisp.models.baseline_imports import BASELINE_ROOT, isolated_baseline_import


class UACANet(BaseSegmentationModel):
    """
    UACANet adapter backed by ``1_baseline/UACANet/lib/UACANet.py``.

    UACANet is thesis-default teacher-side code. The adapter accepts tensors and
    internally builds the sample dict expected by the baseline implementation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        channels: int = 256,
        output_stride: int = 16,
        pretrained: bool = False,
        baseline_root: str | Path | None = None,
    ) -> None:
        super().__init__()
        if in_channels != 3 or num_classes != 1:
            raise ValueError("Baseline UACANet supports RGB binary segmentation only.")
        self._baseline_root = Path(baseline_root) if baseline_root else BASELINE_ROOT / "UACANet"
        self.model = self._build_baseline_model(
            channels=channels,
            output_stride=output_stride,
            pretrained=pretrained,
        )
        self._decoder_channels = 1

    def _build_baseline_model(
        self,
        channels: int,
        output_stride: int,
        pretrained: bool,
    ) -> nn.Module:
        with isolated_baseline_import(self._baseline_root):
            module = import_module("lib.UACANet")
            return module.UACANet(
                channels=channels,
                output_stride=output_stride,
                pretrained=pretrained,
            )

    @property
    def decoder_channels(self) -> int:
        return self._decoder_channels

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def forward(self, x: torch.Tensor) -> SegmentationOutput:
        out = self.model({"image": x})
        logits = out["pred"]
        logits = F.interpolate(
            logits,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        features = F.interpolate(
            logits,
            size=(max(1, x.shape[2] // 4), max(1, x.shape[3] // 4)),
            mode="bilinear",
            align_corners=False,
        )
        return SegmentationOutput(logits=logits, features=features, aux=out)
