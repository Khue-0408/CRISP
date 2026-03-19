"""
Polyp-PVT backbone wrapper.

This file mirrors the role of other backbone wrappers and exists to make
Table 1-style multi-backbone reproduction clean and modular.

Uses a lightweight PVT-v2 style encoder with a simple FPN-like decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from crisp.models.base import BaseSegmentationModel, SegmentationOutput


class _SimpleFPNDecoder(nn.Module):
    """Lightweight FPN-like decoder for multi-scale feature fusion."""

    def __init__(self, in_channels_list: list[int], out_channels: int = 64) -> None:
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels_list
        ])
        self.smooth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(out_channels, 1, 1)

    def forward(self, features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # Top-down pathway.
        laterals = [lat(f) for lat, f in zip(self.lateral_convs, features)]
        # Fuse from deepest to shallowest.
        for i in range(len(laterals) - 1, 0, -1):
            target_size = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=target_size, mode="bilinear", align_corners=False
            )
        # Use shallowest fused feature.
        dec_feat = self.smooth(laterals[0])
        logits = self.out_conv(dec_feat)
        return logits, dec_feat


class _MiniPVTBlock(nn.Module):
    """Simplified PVT-like block: depthwise conv + pointwise + SE attention."""

    def __init__(self, dim: int, sr_ratio: int = 1) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.pwconv(self.norm(self.dwconv(x))))


class _PVTStage(nn.Module):
    """One encoder stage: optional down-sample → N transformer-like blocks."""

    def __init__(self, in_ch: int, out_ch: int, depth: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, stride, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if stride > 1 else nn.Conv2d(in_ch, out_ch, 1)
        self.blocks = nn.Sequential(*[_MiniPVTBlock(out_ch) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.downsample(x))


class PolypPVT(BaseSegmentationModel):
    """
    Wrapper class for a Polyp-PVT segmentation model.

    Uses a simplified PVT-v2 style encoder with an FPN decoder. For a full
    reproduction with official PVT-v2 weights, the encoder can be replaced
    with the pretrained PVTv2-B2 from timm.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        embed_dims: tuple[int, ...] = (64, 128, 256, 512),
        depths: tuple[int, ...] = (2, 2, 2, 2),
        decoder_channels: int = 64,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(inplace=True),
        )
        self.stages = nn.ModuleList()
        for i in range(len(embed_dims)):
            in_ch = embed_dims[0] if i == 0 else embed_dims[i - 1]
            stride = 1 if i == 0 else 2
            self.stages.append(_PVTStage(in_ch, embed_dims[i], depths[i], stride))

        self.decoder = _SimpleFPNDecoder(list(embed_dims), decoder_channels)
        self._decoder_channels = decoder_channels

    @property
    def decoder_channels(self) -> int:
        return self._decoder_channels

    def forward(self, x: torch.Tensor) -> SegmentationOutput:
        input_size = x.shape[2:]
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        logits_dec, dec_feat = self.decoder(features)
        logits = F.interpolate(logits_dec, size=input_size, mode="bilinear", align_corners=False)
        return SegmentationOutput(logits=logits, features=dec_feat)
