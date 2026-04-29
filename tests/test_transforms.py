"""Tests for segmentation transform alignment and thesis augmentations."""

from __future__ import annotations

import numpy as np
import torch

from crisp.data.transforms import build_eval_transforms, build_train_transforms


def _square_pair(size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 1
    image[mask == 1] = (255, 255, 255)
    return image, mask


def test_train_transform_supports_chapter4_augmentation_and_binary_mask() -> None:
    image, mask = _square_pair()
    transform = build_train_transforms(
        {
            "image_size": 48,
            "random_hflip": True,
            "random_vflip": True,
            "random_rotate_degrees": 15,
            "random_scale_range": [0.75, 1.25],
            "color_jitter": {
                "brightness": 0.10,
                "contrast": 0.10,
                "saturation": 0.10,
                "hue": 0.02,
            },
            "random_gaussian_blur": {"probability": 1.0, "sigma": [0.1, 1.0]},
        }
    )

    torch.manual_seed(2026)
    image_t, mask_t = transform(image, mask)

    assert image_t.shape == (3, 48, 48)
    assert mask_t.shape == (1, 48, 48)
    assert set(mask_t.unique().tolist()).issubset({0.0, 1.0})


def test_train_transform_keeps_image_mask_geometry_synchronized() -> None:
    image, mask = _square_pair()
    transform = build_train_transforms(
        {
            "image_size": 64,
            "random_hflip": True,
            "random_vflip": True,
            "random_rotate_degrees": 15,
            "random_scale_range": [0.75, 1.25],
            "color_jitter": {
                "brightness": 0.0,
                "contrast": 0.0,
                "saturation": 0.0,
                "hue": 0.0,
            },
            "random_gaussian_blur": {"probability": 0.0},
            "normalize_mean": [0.0, 0.0, 0.0],
            "normalize_std": [1.0, 1.0, 1.0],
        }
    )

    torch.manual_seed(17)
    image_t, mask_t = transform(image, mask)

    image_fg = image_t[:1] > 0.5
    mask_fg = mask_t > 0.5
    intersection = (image_fg & mask_fg).sum().float()
    union = (image_fg | mask_fg).sum().float()
    assert intersection / union > 0.85


def test_eval_transform_is_deterministic() -> None:
    image, mask = _square_pair()
    transform = build_eval_transforms(
        {
            "image_size": 40,
            "normalize_mean": [0.0, 0.0, 0.0],
            "normalize_std": [1.0, 1.0, 1.0],
        }
    )

    first_image, first_mask = transform(image, mask)
    second_image, second_mask = transform(image, mask)

    assert torch.allclose(first_image, second_image)
    assert torch.equal(first_mask, second_mask)


def test_normalization_override_is_honored() -> None:
    image = np.full((8, 8, 3), 255, dtype=np.uint8)
    mask = np.ones((8, 8), dtype=np.uint8)
    transform = build_eval_transforms(
        {
            "image_size": 8,
            "normalize_mean": [0.0, 0.0, 0.0],
            "normalize_std": [1.0, 1.0, 1.0],
        }
    )

    image_t, mask_t = transform(image, mask)

    assert torch.allclose(image_t, torch.ones_like(image_t))
    assert torch.equal(mask_t, torch.ones_like(mask_t))
