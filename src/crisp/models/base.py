"""
Base model interfaces for segmentation backbones.

The main requirement imposed by CRISP is that a backbone should expose:
1. raw foreground logits z(x),
2. decoder-aligned features F(x) suitable for the projector head.

This file defines lightweight interfaces and common return conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class SegmentationOutput:
    """
    Structured container returned by student backbones.

    Attributes
    ----------
    logits:
        Raw foreground logits of shape [B, 1, H, W].
    features:
        Decoder-aligned feature map used by the projector head.
    aux:
        Optional auxiliary outputs for backbone-specific training or debugging.
    """

    logits: torch.Tensor
    features: torch.Tensor
    aux: Optional[dict] = None


class BaseSegmentationModel(nn.Module):
    """
    Abstract base class for segmentation backbones.

    Any concrete model should implement `forward` so that the training loop
    can consume logits and decoder features in a uniform way.
    """

    def forward(self, x: torch.Tensor) -> SegmentationOutput:
        """
        Run a forward pass and return logits and decoder features.

        Parameters
        ----------
        x:
            Input tensor of shape [B, 3, H, W].

        Returns
        -------
        SegmentationOutput
            Structured model outputs.
        """
        raise NotImplementedError
