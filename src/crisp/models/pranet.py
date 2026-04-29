"""Baseline-backed PraNet adapter."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from crisp.models.base import BaseSegmentationModel, SegmentationOutput
from crisp.models.baseline_imports import BASELINE_ROOT, isolated_baseline_import


class PraNet(BaseSegmentationModel):
    """
    PraNet adapter backed by ``1_baseline/PraNet/lib/PraNet_Res2Net.py``.

    The baseline implementation hard-codes ImageNet pretraining from an absolute
    path. The adapter disables that initialization because the full PraNet
    checkpoint is loaded explicitly by the CRISP training/evaluation wiring.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        pretrained: bool = False,
        channel: int = 32,
        baseline_root: str | Path | None = None,
    ) -> None:
        super().__init__()
        if in_channels != 3 or num_classes != 1:
            raise ValueError("Baseline PraNet supports RGB binary segmentation only.")
        self._baseline_root = Path(baseline_root) if baseline_root else BASELINE_ROOT / "PraNet"
        self.model = self._build_baseline_model(channel=channel)
        self._decoder_channels = int(channel)

    def _build_baseline_model(self, channel: int) -> nn.Module:
        with isolated_baseline_import(self._baseline_root):
            module = import_module("lib.PraNet_Res2Net")
            res2net_module = import_module("lib.Res2Net_v1b")
            original_res2net50 = res2net_module.res2net50_v1b_26w_4s

            def _no_external_pretrain(pretrained: bool = False, **kwargs: Any) -> nn.Module:
                return original_res2net50(pretrained=False, **kwargs)

            module.res2net50_v1b_26w_4s = _no_external_pretrain
            return module.PraNet(channel=channel)

    @property
    def decoder_channels(self) -> int:
        return self._decoder_channels

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def forward(self, x: torch.Tensor) -> SegmentationOutput:
        input_size = x.shape[2:]
        z = self.model.resnet.conv1(x)
        z = self.model.resnet.bn1(z)
        z = self.model.resnet.relu(z)
        z = self.model.resnet.maxpool(z)
        x1 = self.model.resnet.layer1(z)
        x2 = self.model.resnet.layer2(x1)
        x3 = self.model.resnet.layer3(x2)
        x4 = self.model.resnet.layer4(x3)

        x2_rfb = self.model.rfb2_1(x2)
        x3_rfb = self.model.rfb3_1(x3)
        x4_rfb = self.model.rfb4_1(x4)

        ra5_feat = self.model.agg1(x4_rfb, x3_rfb, x2_rfb)
        lateral_map_5 = F.interpolate(ra5_feat, size=input_size, mode="bilinear", align_corners=False)

        crop_4 = F.interpolate(ra5_feat, size=x4.shape[2:], mode="bilinear", align_corners=False)
        z = -1 * torch.sigmoid(crop_4) + 1
        z = z.expand(-1, 2048, -1, -1).mul(x4)
        z = self.model.ra4_conv1(z)
        z = F.relu(self.model.ra4_conv2(z))
        z = F.relu(self.model.ra4_conv3(z))
        z = F.relu(self.model.ra4_conv4(z))
        ra4_feat = self.model.ra4_conv5(z)
        z = ra4_feat + crop_4
        lateral_map_4 = F.interpolate(z, size=input_size, mode="bilinear", align_corners=False)

        crop_3 = F.interpolate(z, size=x3.shape[2:], mode="bilinear", align_corners=False)
        z = -1 * torch.sigmoid(crop_3) + 1
        z = z.expand(-1, 1024, -1, -1).mul(x3)
        z = self.model.ra3_conv1(z)
        z = F.relu(self.model.ra3_conv2(z))
        z = F.relu(self.model.ra3_conv3(z))
        ra3_feat = self.model.ra3_conv4(z)
        z = ra3_feat + crop_3
        lateral_map_3 = F.interpolate(z, size=input_size, mode="bilinear", align_corners=False)

        crop_2 = F.interpolate(z, size=x2.shape[2:], mode="bilinear", align_corners=False)
        z = -1 * torch.sigmoid(crop_2) + 1
        z = z.expand(-1, 512, -1, -1).mul(x2)
        z = self.model.ra2_conv1(z)
        z = F.relu(self.model.ra2_conv2(z))
        z = F.relu(self.model.ra2_conv3(z))
        ra2_feat = self.model.ra2_conv4(z)
        z = ra2_feat + crop_2
        lateral_map_2 = F.interpolate(z, size=input_size, mode="bilinear", align_corners=False)

        logits = F.interpolate(
            lateral_map_2,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        return SegmentationOutput(
            logits=logits,
            features=x2_rfb,
            aux={
                "lateral_map_5": lateral_map_5,
                "lateral_map_4": lateral_map_4,
                "lateral_map_3": lateral_map_3,
                "lateral_map_2": lateral_map_2,
            },
        )
