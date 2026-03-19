"""
RaBiT backbone wrapper.

This is an interface for the RaBiT architecture used in the paper's
cross-domain evaluation suite. Uses a ResNet-based encoder with a
reverse-attention boundary decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from crisp.models.base import BaseSegmentationModel, SegmentationOutput


class _BoundaryRefinement(nn.Module):
    """Boundary-aware refinement module for RaBiT-style decoding."""

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(mid_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.conv2(self.conv1(x)))


class RaBiT(BaseSegmentationModel):
    """
    Wrapper class for a RaBiT-style backbone.

    This scaffold reserves a stable interface so the rest of the codebase
    does not depend on architecture-specific internals.

    Uses a ResNet-34 encoder with a simple boundary-aware decoder.
    For a full RaBiT reproduction, replace with the official architecture.

    Parameters
    ----------
    in_channels:
        Number of input channels (default 3).
    num_classes:
        Number of output classes (default 1).
    decoder_ch:
        Internal decoder channel width (default 64).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        decoder_ch: int = 64,
    ) -> None:
        super().__init__()
        resnet = models.resnet34(weights=None)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1   # stride 4, 64 ch
        self.layer2 = resnet.layer2   # stride 8, 128 ch
        self.layer3 = resnet.layer3   # stride 16, 256 ch
        self.layer4 = resnet.layer4   # stride 32, 512 ch

        # Reduce encoder channels
        self.reduce4 = nn.Conv2d(512, decoder_ch, 1)
        self.reduce3 = nn.Conv2d(256, decoder_ch, 1)
        self.reduce2 = nn.Conv2d(128, decoder_ch, 1)

        # Boundary refinement stages
        self.refine3 = _BoundaryRefinement(decoder_ch, decoder_ch, decoder_ch)
        self.refine2 = _BoundaryRefinement(decoder_ch, decoder_ch, decoder_ch)

        self.head = nn.Conv2d(decoder_ch, num_classes, 1)
        self._decoder_channels = decoder_ch

    @property
    def decoder_channels(self) -> int:
        return self._decoder_channels

    def forward(self, x: torch.Tensor) -> SegmentationOutput:
        input_size = x.shape[2:]

        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # Top-down decoding
        d4 = self.reduce4(x4)  # stride 32
        d3 = self.reduce3(x3) + F.interpolate(d4, size=x3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.refine3(d3)  # stride 16
        d2 = self.reduce2(x2) + F.interpolate(d3, size=x2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.refine2(d2)  # stride 8 — decoder features

        logits = self.head(d2)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

        return SegmentationOutput(logits=logits, features=d2)
