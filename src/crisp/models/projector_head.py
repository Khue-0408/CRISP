"""
CRISP amortized projector head.

This module implements the lightweight head that predicts the spatial inverse-temperature
field α̂_φ(u) from decoder features and student logits. In the paper,
the head is a two-layer 3x3 convolutional block with 64 hidden channels, GroupNorm,
GELU, and a final 1x1 convolution, operating at quarter resolution and upsampled
to logit resolution. [file:1]

CRISP reference: instruct.md §10.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CRISPProjectorHead(nn.Module):
    """
    Lightweight amortized projector for CRISP.

    Responsibilities
    ----------------
    - consume decoder features F_theta(x) and raw student logits z(x),
    - predict a bounded inverse-temperature field alpha_hat,
    - preserve a clean separation between the learned projector and the detached
      local projection solver used during training.

    Parameters
    ----------
    feature_channels:
        Number of channels in the decoder feature tensor.
    hidden_channels:
        Hidden channel width of the projector head (default 64).
    alpha_min:
        Lower bound of the inverse-temperature interval (default 0.50).
    alpha_max:
        Upper bound of the inverse-temperature interval (default 1.80).
    norm:
        Normalization type used inside the projector head.

    CRISP reference
    ---------------
    instruct.md §10:
      α̂(u) = α_min + (α_max - α_min) · σ(a_φ(F_θ(x)(u), z(u)))
      Two-layer 3×3 conv block, 64 hidden channels, GroupNorm, GELU, 1×1 conv.
      Predicts at quarter resolution and is bilinearly upsampled.
    """

    def __init__(
        self,
        feature_channels: int,
        hidden_channels: int = 64,
        alpha_min: float = 0.50,
        alpha_max: float = 1.80,
        norm: str = "groupnorm",
    ) -> None:
        super().__init__()
        if alpha_min >= alpha_max:
            raise ValueError(
                f"Expected alpha_min < alpha_max, got {alpha_min} >= {alpha_max}."
            )
        if norm.lower() != "groupnorm":
            raise ValueError(
                "CRISPProjectorHead currently supports only GroupNorm, matching instruct.md §10."
            )
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        # Input: concatenation of decoder features + downsampled logits (1 ch).
        in_channels = feature_channels + 1

        # Two 3×3 conv layers with GroupNorm and GELU.
        num_groups = min(32, hidden_channels)  # GroupNorm requires groups ≤ channels
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, hidden_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups, hidden_channels)
        self.act2 = nn.GELU()

        # Final 1×1 conv to single-channel unconstrained score.
        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        output_size: Tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """
        Predict the bounded inverse-temperature field alpha_hat.

        Parameters
        ----------
        features:
            Decoder-aligned features of shape [B, C, Hf, Wf].
        logits:
            Raw foreground logits of shape [B, 1, H, W].
        output_size:
            Optional target spatial size (H, W) for upsampling alpha_hat.
            If None, uses the spatial size of *logits*.

        Returns
        -------
        torch.Tensor
            Predicted inverse-temperature field of shape [B, 1, H, W]
            with values constrained to [alpha_min, alpha_max].
        """
        if output_size is None:
            output_size = (logits.shape[2], logits.shape[3])

        # Step 1: Align logits to feature resolution (quarter resolution).
        feat_size = (features.shape[2], features.shape[3])
        logits_down = F.interpolate(
            logits, size=feat_size, mode="bilinear", align_corners=False
        )

        # Step 2: Concatenate features and downsampled logits.
        x = torch.cat([features, logits_down], dim=1)  # [B, C+1, Hf, Wf]

        # Step 3: Two-layer conv block.
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))

        # Step 4: Predict unconstrained score map.
        score = self.head(x)  # [B, 1, Hf, Wf]

        # Step 5: Upsample to output resolution.
        score = F.interpolate(
            score, size=output_size, mode="bilinear", align_corners=False
        )  # [B, 1, H, W]

        # Step 6: Sigmoid-affine remap to enforce [alpha_min, alpha_max].
        alpha_hat = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(score)
        if (alpha_hat < self.alpha_min - 1e-6).any() or (alpha_hat > self.alpha_max + 1e-6).any():
            raise RuntimeError("Projector produced alpha_hat outside configured bounds.")

        return alpha_hat
