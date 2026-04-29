"""Baseline-compatible U-Net wrapper."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from crisp.models.base import BaseSegmentationModel, SegmentationOutput


def _pad_to_skip(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    diff_y = skip.shape[2] - x.shape[2]
    diff_x = skip.shape[3] - x.shape[3]
    return F.pad(
        x,
        [
            diff_x // 2,
            diff_x - diff_x // 2,
            diff_y // 2,
            diff_y - diff_y // 2,
        ],
    )


class UNet(BaseSegmentationModel):
    """
    U-Net host using the canonical baseline keyspace from ``1_baseline/unet``.

    The module names intentionally match ``1_baseline/unet/models/unet.py`` so
    the real checkpoint under ``1_baseline`` can load directly. The forward
    return is adapted to CRISP by exposing quarter-resolution decoder features.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
        channel_in: int | None = None,
        channel_out: int | None = None,
    ) -> None:
        super().__init__()
        channel_in = int(channel_in if channel_in is not None else in_channels)
        channel_out = int(channel_out if channel_out is not None else num_classes)
        c = int(base_channels)

        self.initial = self._conv_block(channel_in, c)
        self.down0 = self._conv_block(c, 2 * c)
        self.down1 = self._conv_block(2 * c, 4 * c)
        self.down2 = self._conv_block(4 * c, 8 * c)
        self.down3 = self._conv_block(8 * c, 16 * c)

        self.up0_0 = nn.ConvTranspose2d(16 * c, 8 * c, kernel_size=2, stride=2)
        self.up0_1 = self._conv_block(16 * c, 8 * c)
        self.up1_0 = nn.ConvTranspose2d(8 * c, 4 * c, kernel_size=2, stride=2)
        self.up1_1 = self._conv_block(8 * c, 4 * c)
        self.up2_0 = nn.ConvTranspose2d(4 * c, 2 * c, kernel_size=2, stride=2)
        self.up2_1 = self._conv_block(4 * c, 2 * c)
        self.up3_0 = nn.ConvTranspose2d(2 * c, c, kernel_size=2, stride=2)
        self.up3_1 = self._conv_block(2 * c, c)

        self.final = nn.Conv2d(c, channel_out, kernel_size=1)
        self._decoder_channels = 4 * c

    @staticmethod
    def _conv_block(channel_in: int, channel_out: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_out, channel_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True),
        )

    @property
    def decoder_channels(self) -> int:
        return self._decoder_channels

    def forward(self, x: torch.Tensor) -> SegmentationOutput:
        x_in = self.initial(x)
        enc0 = self.down0(F.max_pool2d(x_in, 2))
        enc1 = self.down1(F.max_pool2d(enc0, 2))
        enc2 = self.down2(F.max_pool2d(enc1, 2))
        enc3 = self.down3(F.max_pool2d(enc2, 2))

        dec0 = _pad_to_skip(self.up0_0(enc3), enc2)
        dec0 = self.up0_1(torch.cat((enc2, dec0), dim=1))

        dec1 = _pad_to_skip(self.up1_0(dec0), enc1)
        dec1 = self.up1_1(torch.cat((enc1, dec1), dim=1))

        dec2 = _pad_to_skip(self.up2_0(dec1), enc0)
        dec2 = self.up2_1(torch.cat((enc0, dec2), dim=1))

        dec3 = _pad_to_skip(self.up3_0(dec2), x_in)
        dec3 = self.up3_1(torch.cat((x_in, dec3), dim=1))

        logits = self.final(dec3)
        return SegmentationOutput(logits=logits, features=dec1)
