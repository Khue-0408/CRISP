"""
U-Net backbone wrapper.

This module provides a standardized U-Net interface so U-Net can be used
as both a student host and, potentially, a teacher model in the CRISP pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from crisp.models.base import BaseSegmentationModel, SegmentationOutput


class _DoubleConv(nn.Module):
    """Two 3×3 convolutions with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Down(nn.Module):
    """Max‑pool then double conv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class _Up(nn.Module):
    """Bilinear upsample + skip‑concat + double conv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(BaseSegmentationModel):
    """
    Standard U-Net wrapper with CRISP-compatible outputs.

    Parameters
    ----------
    in_channels:
        Number of input image channels (default 3).
    num_classes:
        Number of output classes (default 1 for binary).
    base_channels:
        Base channel width; each level doubles this (default 64).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        c = base_channels

        self.inc = _DoubleConv(in_channels, c)
        self.down1 = _Down(c, 2 * c)
        self.down2 = _Down(2 * c, 4 * c)
        self.down3 = _Down(4 * c, 8 * c)
        self.down4 = _Down(8 * c, 16 * c)

        self.up1 = _Up(16 * c + 8 * c, 8 * c)
        self.up2 = _Up(8 * c + 4 * c, 4 * c)
        self.up3 = _Up(4 * c + 2 * c, 2 * c)
        self.up4 = _Up(2 * c + c, c)

        self.out_conv = nn.Conv2d(c, num_classes, 1)

        # Decoder features exposed to the projector come from
        # the deepest decoder stage at quarter resolution.
        self._decoder_channels = 4 * c

    @property
    def decoder_channels(self) -> int:
        """Number of channels in the decoder feature map for the projector."""
        return self._decoder_channels

    def forward(self, x: torch.Tensor) -> SegmentationOutput:
        """
        Run the U-Net forward pass and return logits plus decoder features.

        Returns
        -------
        SegmentationOutput
            logits at input resolution, features at ~quarter resolution.
        """
        x1 = self.inc(x)       # [B, c,   H,   W]
        x2 = self.down1(x1)    # [B, 2c,  H/2, W/2]
        x3 = self.down2(x2)    # [B, 4c,  H/4, W/4]
        x4 = self.down3(x3)    # [B, 8c,  H/8, W/8]
        x5 = self.down4(x4)    # [B, 16c, H/16, W/16]

        d4 = self.up1(x5, x4)  # [B, 8c,  H/8, W/8]
        d3 = self.up2(d4, x3)  # [B, 4c,  H/4, W/4]  ← decoder features
        d2 = self.up3(d3, x2)  # [B, 2c,  H/2, W/2]
        d1 = self.up4(d2, x1)  # [B, c,   H,   W]

        logits = self.out_conv(d1)  # [B, 1, H, W]

        # Expose quarter-resolution decoder features for the CRISP projector.
        return SegmentationOutput(logits=logits, features=d3)
