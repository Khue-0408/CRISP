"""Baseline-backed Polyp-PVT adapter."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from crisp.models.base import BaseSegmentationModel, SegmentationOutput
from crisp.models.baseline_imports import BASELINE_ROOT, isolated_baseline_import


class PolypPVT(BaseSegmentationModel):
    """
    Polyp-PVT adapter backed by ``1_baseline/Polyp-PVT/lib/pvt.py``.

    Baseline inference uses ``prediction1 + prediction2`` as logits before
    sigmoid normalization; the adapter preserves that behavior.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        pretrained: bool = False,
        channel: int = 32,
        decoder_channels: int | None = None,
        baseline_root: str | Path | None = None,
    ) -> None:
        super().__init__()
        if in_channels != 3 or num_classes != 1:
            raise ValueError("Baseline Polyp-PVT supports RGB binary segmentation only.")
        self._baseline_root = Path(baseline_root) if baseline_root else BASELINE_ROOT / "Polyp-PVT"
        self.model = self._build_baseline_model(channel=channel)
        self._decoder_channels = int(decoder_channels or channel)

    def _build_baseline_model(self, channel: int) -> nn.Module:
        with isolated_baseline_import(self._baseline_root, chdir=True):
            module = import_module("lib.pvt")
            return module.PolypPVT(channel=channel)

    @property
    def decoder_channels(self) -> int:
        return self._decoder_channels

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def forward(self, x: torch.Tensor) -> SegmentationOutput:
        input_size = x.shape[2:]
        pvt = self.model.backbone(x)
        x1, x2, x3, x4 = pvt[0], pvt[1], pvt[2], pvt[3]

        x1 = self.model.ca(x1) * x1
        cim_feature = self.model.sa(x1) * x1

        x2_t = self.model.Translayer2_1(x2)
        x3_t = self.model.Translayer3_1(x3)
        x4_t = self.model.Translayer4_1(x4)
        cfm_feature = self.model.CFM(x4_t, x3_t, x2_t)

        t2 = self.model.Translayer2_0(cim_feature)
        t2 = self.model.down05(t2)
        sam_feature = self.model.SAM(cfm_feature, t2)

        pred1 = self.model.out_CFM(cfm_feature)
        pred2 = self.model.out_SAM(sam_feature)
        logits = F.interpolate(
            pred1 + pred2,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        features = F.interpolate(
            cfm_feature,
            size=(max(1, input_size[0] // 4), max(1, input_size[1] // 4)),
            mode="bilinear",
            align_corners=False,
        )
        return SegmentationOutput(
            logits=logits,
            features=features,
            aux={"prediction1": pred1, "prediction2": pred2},
        )
