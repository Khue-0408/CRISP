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

from collections.abc import Sequence
from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _as_pair(value: Any, default: Tuple[float, float]) -> Tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        scalar = float(value)
        return (scalar, scalar)
    if isinstance(value, Sequence) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    raise ValueError(f"Expected scalar or length-2 sequence, got {value!r}.")


def _as_tuple(value: Any, default: Tuple[float, ...]) -> Tuple[float, ...]:
    if value is None:
        return default
    if isinstance(value, Sequence):
        return tuple(float(v) for v in value)
    raise ValueError(f"Expected sequence, got {value!r}.")


def _sample_uniform(low: float, high: float) -> float:
    return torch.empty(1).uniform_(low, high).item()


class _JointTransform:
    """Apply a sequence of co-ordinated image-mask transforms."""

    def __init__(
        self,
        resize: Tuple[int, int],
        random_hflip: bool = False,
        random_vflip: bool = False,
        random_rotate_degrees: float = 0.0,
        random_scale_range: Tuple[float, float] = (1.0, 1.0),
        color_jitter: Dict[str, float] | None = None,
        random_gaussian_blur: Dict[str, Any] | None = None,
        normalize_mean: Tuple[float, ...] = _IMAGENET_MEAN,
        normalize_std: Tuple[float, ...] = _IMAGENET_STD,
    ) -> None:
        self.resize = resize
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip
        self.random_rotate_degrees = max(0.0, float(random_rotate_degrees))
        self.random_scale_range = random_scale_range
        self.color_jitter = color_jitter or {}
        self.random_gaussian_blur = random_gaussian_blur or {}
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

        scale_min, scale_max = self.random_scale_range
        use_affine = self.random_rotate_degrees > 0.0 or (scale_min, scale_max) != (1.0, 1.0)
        if use_affine:
            angle = _sample_uniform(-self.random_rotate_degrees, self.random_rotate_degrees)
            scale = _sample_uniform(scale_min, scale_max)
            img_pil = TF.affine(
                img_pil,
                angle=angle,
                translate=[0, 0],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=0,
            )
            mask_pil = TF.affine(
                mask_pil,
                angle=angle,
                translate=[0, 0],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=TF.InterpolationMode.NEAREST,
                fill=0,
            )

        # Photometric augmentation applies to the image only.
        img_pil = self._apply_color_jitter(img_pil)
        img_pil = self._apply_gaussian_blur(img_pil)

        # To tensor
        img_t = TF.to_tensor(img_pil)  # [3, H, W] float in [0,1]
        mask_t = TF.to_tensor(mask_pil)  # [1, H, W] float in [0,1]

        # Normalize image
        img_t = TF.normalize(img_t, mean=self.normalize_mean, std=self.normalize_std)

        # Binarize mask (handle potential interpolation artifacts)
        mask_t = (mask_t >= 0.5).float()

        return img_t, mask_t

    def _apply_color_jitter(self, img_pil: Image.Image) -> Image.Image:
        brightness = float(self.color_jitter.get("brightness", 0.0) or 0.0)
        contrast = float(self.color_jitter.get("contrast", 0.0) or 0.0)
        saturation = float(self.color_jitter.get("saturation", 0.0) or 0.0)
        hue = float(self.color_jitter.get("hue", 0.0) or 0.0)

        if brightness > 0.0:
            img_pil = TF.adjust_brightness(
                img_pil,
                max(0.0, _sample_uniform(1.0 - brightness, 1.0 + brightness)),
            )
        if contrast > 0.0:
            img_pil = TF.adjust_contrast(
                img_pil,
                max(0.0, _sample_uniform(1.0 - contrast, 1.0 + contrast)),
            )
        if saturation > 0.0:
            img_pil = TF.adjust_saturation(
                img_pil,
                max(0.0, _sample_uniform(1.0 - saturation, 1.0 + saturation)),
            )
        if hue > 0.0:
            hue_factor = _sample_uniform(-min(hue, 0.5), min(hue, 0.5))
            img_pil = TF.adjust_hue(img_pil, hue_factor)
        return img_pil

    def _apply_gaussian_blur(self, img_pil: Image.Image) -> Image.Image:
        probability = float(
            self.random_gaussian_blur.get(
                "probability",
                self.random_gaussian_blur.get("p", 0.0),
            )
            or 0.0
        )
        if probability <= 0.0 or torch.rand(1).item() >= probability:
            return img_pil

        sigma = _as_pair(self.random_gaussian_blur.get("sigma"), (0.1, 1.0))
        radius = _sample_uniform(sigma[0], sigma[1])
        return img_pil.filter(ImageFilter.GaussianBlur(radius=radius))


def build_train_transforms(config: Dict[str, Any]) -> _JointTransform:
    """
    Build the training-time transform pipeline.

    Parameters
    ----------
    config:
        Dataset or augmentation configuration block.  Expected keys:
        ``image_size`` (int or [H, W]), ``random_hflip`` (bool),
        ``random_vflip`` (bool), and optional Chapter 4 augmentations:
        ``random_rotate_degrees``, ``random_scale_range``, ``color_jitter``,
        and ``random_gaussian_blur``.

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
        random_rotate_degrees=float(config.get("random_rotate_degrees", 0.0) or 0.0),
        random_scale_range=_as_pair(config.get("random_scale_range"), (1.0, 1.0)),
        color_jitter=dict(config.get("color_jitter", {}) or {}),
        random_gaussian_blur=dict(config.get("random_gaussian_blur", {}) or {}),
        normalize_mean=_as_tuple(config.get("normalize_mean"), _IMAGENET_MEAN),
        normalize_std=_as_tuple(config.get("normalize_std"), _IMAGENET_STD),
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
        random_rotate_degrees=0.0,
        random_scale_range=(1.0, 1.0),
        color_jitter=None,
        random_gaussian_blur=None,
        normalize_mean=_as_tuple(config.get("normalize_mean"), _IMAGENET_MEAN),
        normalize_std=_as_tuple(config.get("normalize_std"), _IMAGENET_STD),
    )
