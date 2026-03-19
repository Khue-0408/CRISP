"""
Low-level data I/O helpers.

This module centralizes tiny but important utilities such as:
- reading RGB images,
- reading binary masks,
- converting numpy arrays to tensors,
- validating path existence.

Keeping these helpers here avoids duplicated boilerplate inside dataset classes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image


def read_rgb_image(path: Path) -> np.ndarray:
    """
    Read an RGB image from disk as a HWC numpy array.

    Parameters
    ----------
    path:
        Image path.

    Returns
    -------
    np.ndarray
        RGB image array of shape [H, W, 3] with dtype uint8.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def read_binary_mask(path: Path) -> np.ndarray:
    """
    Read a binary segmentation mask from disk.

    Parameters
    ----------
    path:
        Mask path.

    Returns
    -------
    np.ndarray
        Binary mask array of shape [H, W] with values in {0, 1}, dtype uint8.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mask file not found: {path}")
    mask = Image.open(path).convert("L")
    arr = np.asarray(mask, dtype=np.uint8)
    # Binarize: any pixel > 127 is foreground.
    return (arr > 127).astype(np.uint8)


def numpy_image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """
    Convert a HWC image array into a CHW float tensor normalized to [0, 1].

    Parameters
    ----------
    image:
        Numpy array of shape [H, W, C] with dtype uint8.

    Returns
    -------
    torch.Tensor
        Float32 tensor of shape [C, H, W] in [0, 1].
    """
    # HWC → CHW
    t = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
    t = t / 255.0
    return t


def numpy_mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    """
    Convert a HW binary mask array into a [1, H, W] float tensor.

    Parameters
    ----------
    mask:
        Numpy array of shape [H, W] with values in {0, 1}.

    Returns
    -------
    torch.Tensor
        Float32 tensor of shape [1, H, W].
    """
    if mask.ndim == 2:
        mask = mask[np.newaxis, :, :]  # [1, H, W]
    return torch.from_numpy(mask.copy()).float()
