"""
Dataset definitions for source-domain training and target-domain evaluation.

The main design goal is to unify all segmentation datasets under a common API:
- image tensor,
- binary mask tensor,
- metadata dictionary.

Metadata is important for:
- result export,
- per-dataset evaluation,
- debugging failed examples,
- future support for center-wise or patient-wise splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from crisp.data.io_utils import read_rgb_image, read_binary_mask


@dataclass
class SampleRecord:
    """
    Lightweight record describing one dataset sample.

    Attributes
    ----------
    image_path:
        Absolute or repository-relative path to the RGB image.
    mask_path:
        Path to the binary segmentation mask.
    image_id:
        Stable identifier used for logging and exported predictions.
    dataset_name:
        Name of the dataset the sample belongs to.
    split:
        Split label such as train/val/test.
    """

    image_path: Path
    mask_path: Path
    image_id: str
    dataset_name: str
    split: str


class BinarySegmentationDataset(Dataset):
    """
    Generic binary segmentation dataset.

    This dataset class is intentionally generic so the same code path
    can support Kvasir-SEG, ColonDB, ETIS, and PolypGen with minimal duplication.

    Responsibilities
    ----------------
    - parse sample records from disk,
    - load RGB images and binary masks,
    - apply coordinated image-mask transforms,
    - return tensors and metadata in a consistent format.

    Expected output dictionary
    --------------------------
    {
        "image": FloatTensor [3, H, W],
        "mask": FloatTensor [1, H, W],
        "meta": {
            "image_id": str,
            "dataset_name": str,
            "split": str,
            ...
        }
    }
    """

    def __init__(
        self,
        samples: List[SampleRecord],
        transforms: Optional[Any] = None,
    ) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        samples:
            List of dataset sample records.
        transforms:
            Optional transform pipeline applied jointly to image and mask.
        """
        self.samples = samples
        self.transforms = transforms

    def __len__(self) -> int:
        """
        Return the number of available samples.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Load a single sample.

        Parameters
        ----------
        index:
            Sample index in the internal sample list.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing image tensor, mask tensor, and metadata.
        """
        rec = self.samples[index]

        image_np = read_rgb_image(rec.image_path)
        mask_np = read_binary_mask(rec.mask_path)

        if self.transforms is not None:
            image_t, mask_t = self.transforms(image_np, mask_np)
        else:
            # Fallback: simple conversion without augmentation.
            from crisp.data.io_utils import numpy_image_to_tensor, numpy_mask_to_tensor
            image_t = numpy_image_to_tensor(image_np)
            mask_t = numpy_mask_to_tensor(mask_np)

        meta: Dict[str, Any] = {
            "image_id": rec.image_id,
            "dataset_name": rec.dataset_name,
            "split": rec.split,
            "image_path": str(rec.image_path),
            "mask_path": str(rec.mask_path),
        }

        return {"image": image_t, "mask": mask_t, "meta": meta}


def build_dataset_samples(
    root: Path,
    image_dir: str,
    mask_dir: str,
    split: str,
    dataset_name: str,
    split_file: Optional[str] = None,
) -> List[SampleRecord]:
    """
    Scan a dataset directory and construct sample records.

    Parameters
    ----------
    root:
        Dataset root directory.
    image_dir:
        Relative path containing RGB images.
    mask_dir:
        Relative path containing binary masks.
    split:
        Split name.
    dataset_name:
        Human-readable dataset name.

    Returns
    -------
    List[SampleRecord]
        Parsed sample records sorted by filename stem for determinism.
    """
    root = Path(root)
    img_dir = root / image_dir
    msk_dir = root / mask_dir

    if not img_dir.is_dir():
        candidate = root / split / image_dir
        if candidate.is_dir():
            img_dir = candidate
    if not msk_dir.is_dir():
        candidate = root / split / mask_dir
        if candidate.is_dir():
            msk_dir = candidate

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not msk_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {msk_dir}")

    # Supported image extensions.
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    # Build stem → image_path mapping.
    image_files = {
        p.stem: p
        for p in sorted(img_dir.iterdir())
        if p.suffix.lower() in extensions
    }

    # Build stem → mask_path mapping.
    mask_files = {
        p.stem: p
        for p in sorted(msk_dir.iterdir())
        if p.suffix.lower() in extensions
    }

    # Only keep stems that have both image and mask.
    common_stems = sorted(set(image_files.keys()) & set(mask_files.keys()))

    allowed_ids: Optional[Set[str]] = None
    if split_file:
        split_path = Path(split_file)
        if not split_path.is_absolute():
            split_path = root / split_path
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        allowed_ids = {
            line.strip()
            for line in split_path.read_text().splitlines()
            if line.strip()
        }
        common_stems = [stem for stem in common_stems if stem in allowed_ids]

    records: List[SampleRecord] = []
    for stem in common_stems:
        records.append(
            SampleRecord(
                image_path=image_files[stem],
                mask_path=mask_files[stem],
                image_id=stem,
                dataset_name=dataset_name,
                split=split,
            )
        )
    return records


def build_binary_segmentation_dataset(
    root: str,
    image_dir: str,
    mask_dir: str,
    split: str,
    dataset_name: str,
    transforms: Optional[Any] = None,
    split_file: Optional[str] = None,
) -> BinarySegmentationDataset:
    """
    Build a dataset instance from path configuration.

    This function serves as the canonical dataset factory used by the registry
    and experiment scripts.
    """
    samples = build_dataset_samples(
        root=Path(root),
        image_dir=image_dir,
        mask_dir=mask_dir,
        split=split,
        dataset_name=dataset_name,
        split_file=split_file,
    )
    return BinarySegmentationDataset(samples=samples, transforms=transforms)
