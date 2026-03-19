"""
PraNet backbone wrapper.

This file contains a clean, isolated implementation of the PraNet architecture
used as one of the main hosts in the paper.

Responsibilities
----------------
- expose a standard forward API returning raw logits and decoder features,
- hide architecture-specific internal details behind a stable interface,
- make it easy to swap PraNet for another host backbone.

The implementation follows the PraNet paper (Fan et al., MICCAI 2020)
with a ResNet-based encoder + RFB modules + parallel partial decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from crisp.models.base import BaseSegmentationModel, SegmentationOutput


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _RFBBlock(nn.Module):
    """Receptive Field Block (simplified) for multi-scale feature extraction."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=3, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=5, dilation=5),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(4 * out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv_res = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = torch.cat([x0, x1, x2, x3], dim=1)
        x_cat = self.conv_cat(x_cat)
        return F.relu(x_cat + self.conv_res(x), inplace=True)


class _AggregationModule(nn.Module):
    """Partial Decoder Aggregation for combining multi-level features."""

    def __init__(self, channel: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv_upsample1 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )
        self.conv_upsample2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )
        self.conv_concat = nn.Sequential(
            nn.Conv2d(3 * channel, 3 * channel, 3, padding=1),
            nn.BatchNorm2d(3 * channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(3 * channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )
        self.conv_out = nn.Conv2d(channel, 1, 1)

    def forward(
        self, f3: torch.Tensor, f4: torch.Tensor, f5: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits : [B, 1, H_f3, W_f3]
        features : [B, channel, H_f3, W_f3]  — decoder features for projector.
        """
        target_size = f3.shape[2:]
        f4_up = self.conv_upsample1(
            F.interpolate(f4, size=target_size, mode="bilinear", align_corners=False)
        )
        f5_up = self.conv_upsample2(
            F.interpolate(f5, size=target_size, mode="bilinear", align_corners=False)
        )
        concat = torch.cat([f3, f4_up, f5_up], dim=1)
        features = self.conv_concat(concat)
        logits = self.conv_out(features)
        return logits, features


# ---------------------------------------------------------------------------
# PraNet wrapper
# ---------------------------------------------------------------------------

class PraNet(BaseSegmentationModel):
    """
    Wrapper class for the PraNet segmentation backbone.

    Parameters
    ----------
    in_channels:
        Number of input channels, usually 3 for RGB images.
    num_classes:
        Number of output segmentation classes, 1 for binary segmentation.
    pretrained:
        Whether to initialize the encoder with pretrained weights.
    channel:
        Internal feature channel width for RFB and aggregation modules.

    Notes
    -----
    The encoder uses a standard ResNet-34 backbone. For a closer match to the
    original PraNet (Res2Net-50), swap the encoder when Res2Net weights are available.
    The wrapper returns logits at the input spatial resolution (upsampled) and
    decoder-aligned features at quarter resolution for the CRISP projector.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        pretrained: bool = False,
        channel: int = 32,
    ) -> None:
        super().__init__()
        # Encoder: use ResNet-34 for simplicity (Res2Net can be swapped in).
        resnet = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT if pretrained else None
        )
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # stride 4,  64 ch
        self.layer2 = resnet.layer2  # stride 8, 128 ch
        self.layer3 = resnet.layer3  # stride 16, 256 ch
        self.layer4 = resnet.layer4  # stride 32, 512 ch

        # RFB modules to reduce channel width for each encoder stage.
        self.rfb2 = _RFBBlock(128, channel)
        self.rfb3 = _RFBBlock(256, channel)
        self.rfb4 = _RFBBlock(512, channel)

        # Aggregation / partial decoder.
        self.agg = _AggregationModule(channel)

        self._decoder_channels = channel

    @property
    def decoder_channels(self) -> int:
        """Number of channels in the decoder feature map passed to the projector."""
        return self._decoder_channels

    def forward(self, x: torch.Tensor) -> SegmentationOutput:
        """
        Forward pass through PraNet.

        Parameters
        ----------
        x:
            Input batch of images [B, 3, H, W].

        Returns
        -------
        SegmentationOutput
            Raw logits [B, 1, H, W] upsampled to input size and
            decoder features at ~quarter resolution.
        """
        input_size = x.shape[2:]

        # Encoder
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # RFB refinement
        f2 = self.rfb2(x2)
        f3 = self.rfb3(x3)
        f4 = self.rfb4(x4)

        # Aggregation
        logits_dec, features = self.agg(f2, f3, f4)

        # Upsample logits to input resolution.
        logits = F.interpolate(
            logits_dec, size=input_size, mode="bilinear", align_corners=False
        )

        return SegmentationOutput(logits=logits, features=features)
