"""
Transform builders for segmentation training and evaluation.

The transform pipeline must preserve image-mask alignment while allowing
augmentation parity across baselines.

Typical responsibilities include:
- resize to task-specific input resolution,
- random flips or photometric augmentation during training,
- tensor conversion and normalization,
- deterministic preprocessing during validation and testing.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


class _JointTransform:
    """Apply a sequence of co-ordinated image-mask transforms."""

    def __init__(
        self,
        resize: Tuple[int, int],
        random_hflip: bool = False,
        random_vflip: bool = False,
        normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        self.resize = resize
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def __call__(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        image : np.ndarray of shape [H, W, 3], uint8
        mask  : np.ndarray of shape [H, W], uint8 in {0, 1}

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            image tensor [3, H', W'] normalized, mask tensor [1, H', W'] float.
        """
        # Convert to PIL for torchvision functional ops.
        img_pil = Image.fromarray(image)
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L")

        # Resize
        img_pil = TF.resize(img_pil, list(self.resize), interpolation=TF.InterpolationMode.BILINEAR)
        mask_pil = TF.resize(mask_pil, list(self.resize), interpolation=TF.InterpolationMode.NEAREST)

        # Random horizontal flip (synchronised)
        if self.random_hflip and torch.rand(1).item() > 0.5:
            img_pil = TF.hflip(img_pil)
            mask_pil = TF.hflip(mask_pil)

        # Random vertical flip (synchronised)
        if self.random_vflip and torch.rand(1).item() > 0.5:
            img_pil = TF.vflip(img_pil)
            mask_pil = TF.vflip(mask_pil)

        # To tensor
        img_t = TF.to_tensor(img_pil)  # [3, H, W] float in [0,1]
        mask_t = TF.to_tensor(mask_pil)  # [1, H, W] float in [0,1]

        # Normalize image
        img_t = TF.normalize(img_t, mean=self.normalize_mean, std=self.normalize_std)

        # Binarize mask (handle potential interpolation artifacts)
        mask_t = (mask_t >= 0.5).float()

        return img_t, mask_t


def build_train_transforms(config: Dict[str, Any]) -> _JointTransform:
    """
    Build the training-time transform pipeline.

    Parameters
    ----------
    config:
        Dataset or augmentation configuration block.  Expected keys:
        ``image_size`` (int or [H, W]), ``random_hflip`` (bool),
        ``random_vflip`` (bool).

    Returns
    -------
    _JointTransform
        Transform callable that accepts (image_np, mask_np) pairs.
    """
    size = config.get("image_size", config.get("resize", 352))
    if isinstance(size, int):
        size = (size, size)
    return _JointTransform(
        resize=tuple(size),
        random_hflip=config.get("random_hflip", True),
        random_vflip=config.get("random_vflip", True),
    )


def build_eval_transforms(config: Dict[str, Any]) -> _JointTransform:
    """
    Build the deterministic evaluation-time transform pipeline.

    Parameters
    ----------
    config:
        Dataset or preprocessing configuration block.

    Returns
    -------
    _JointTransform
        Deterministic transform (no random augmentations).
    """
    size = config.get("image_size", config.get("resize", 352))
    if isinstance(size, int):
        size = (size, size)
    return _JointTransform(
        resize=tuple(size),
        random_hflip=False,
        random_vflip=False,
    )
