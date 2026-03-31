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
import random
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from torch.utils.data import Dataset

from crisp.data.io_utils import (
    build_stem_to_path_map,
    candidate_dir_names,
    read_binary_mask,
    read_rgb_image,
)
from crisp.utils.paths import ensure_dir, resolve_local_data_root, resolve_path


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
    image_dir_candidates: Optional[Iterable[str]] = None,
    mask_dir_candidates: Optional[Iterable[str]] = None,
    strict_pairing: bool = False,
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
    img_dir, msk_dir = resolve_dataset_dirs(
        root=root,
        split=split,
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_dir_candidates=image_dir_candidates,
        mask_dir_candidates=mask_dir_candidates,
    )

    image_files = build_stem_to_path_map(img_dir)
    mask_files = build_stem_to_path_map(msk_dir)

    missing_masks = sorted(set(image_files.keys()) - set(mask_files.keys()))
    missing_images = sorted(set(mask_files.keys()) - set(image_files.keys()))
    if strict_pairing and (missing_masks or missing_images):
        raise ValueError(
            "Image/mask pairing mismatch detected. "
            f"Missing masks for stems: {missing_masks[:5]}. "
            f"Missing images for stems: {missing_images[:5]}."
        )

    # Only keep stems that have both image and mask.
    common_stems = sorted(set(image_files.keys()) & set(mask_files.keys()))

    allowed_ids: Optional[Set[str]] = None
    if split_file:
        split_path = resolve_path(split_file)
        if not split_path.exists():
            split_path = root / split_path
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        allowed_ids = {
            line.strip()
            for line in split_path.read_text().splitlines()
            if line.strip()
        }
        if strict_pairing:
            missing_from_dataset = sorted(allowed_ids - set(common_stems))
            if missing_from_dataset:
                raise ValueError(
                    "Split file references ids not present in matched image/mask pairs: "
                    f"{missing_from_dataset[:5]}"
                )
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


def resolve_dataset_dirs(
    root: Path,
    split: str,
    image_dir: str,
    mask_dir: str,
    image_dir_candidates: Optional[Iterable[str]] = None,
    mask_dir_candidates: Optional[Iterable[str]] = None,
) -> Tuple[Path, Path]:
    """
    Resolve image/mask directories with singular/plural fallback support.

    This keeps the original paper path working while allowing local mode to
    accept both ``image|mask`` and ``images|masks`` directory names.
    """
    root = Path(root)
    image_names = candidate_dir_names(image_dir, image_dir_candidates)
    mask_names = candidate_dir_names(mask_dir, mask_dir_candidates)

    image_search_paths = [root / name for name in image_names] + [
        root / split / name for name in image_names
    ]
    mask_search_paths = [root / name for name in mask_names] + [
        root / split / name for name in mask_names
    ]

    img_dir = next((path for path in image_search_paths if path.is_dir()), None)
    msk_dir = next((path for path in mask_search_paths if path.is_dir()), None)

    if img_dir is None:
        raise FileNotFoundError(
            f"Could not find an image directory under {root}. Tried: "
            f"{[str(path) for path in image_search_paths]}"
        )
    if msk_dir is None:
        raise FileNotFoundError(
            f"Could not find a mask directory under {root}. Tried: "
            f"{[str(path) for path in mask_search_paths]}"
        )

    return img_dir, msk_dir


def deterministic_train_val_split_ids(
    image_ids: List[str],
    seed: int,
    val_fraction: float,
) -> Tuple[List[str], List[str]]:
    """
    Build a deterministic train/val split from a list of sample ids.
    """
    if not image_ids:
        raise ValueError("Cannot build a deterministic split from an empty dataset.")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(
            f"val_fraction must lie in (0, 1), got {val_fraction}."
        )

    ordered_ids = sorted(image_ids)
    rng = random.Random(seed)
    rng.shuffle(ordered_ids)

    if len(ordered_ids) == 1:
        raise ValueError(
            "At least two samples are required to create a train/val split."
        )

    val_count = int(round(len(ordered_ids) * val_fraction))
    val_count = max(1, min(val_count, len(ordered_ids) - 1))
    val_ids = sorted(ordered_ids[:val_count])
    train_ids = sorted(ordered_ids[val_count:])
    return train_ids, val_ids


def materialize_deterministic_split_files(
    image_ids: List[str],
    split_root: str | Path,
    dataset_name: str,
    seed: int,
    val_fraction: float,
) -> Dict[str, Path]:
    """
    Save deterministic train/val split files under a metadata folder.
    """
    split_dir = ensure_dir(
        resolve_path(split_root)
        / f"{dataset_name}_seed_{seed}_val_{int(round(val_fraction * 1000))}"
    )
    train_ids, val_ids = deterministic_train_val_split_ids(
        image_ids=image_ids,
        seed=seed,
        val_fraction=val_fraction,
    )

    train_file = split_dir / "train.txt"
    val_file = split_dir / "val.txt"
    train_file.write_text("\n".join(train_ids) + "\n")
    val_file.write_text("\n".join(val_ids) + "\n")
    return {"train": train_file, "val": val_file}


def build_local_train_val_dataset(
    data_cfg: Dict[str, Any],
    split: str,
    transforms: Optional[Any],
    seed: int,
) -> BinarySegmentationDataset:
    """
    Build the local TrainDataset-based train/val dataset.

    Local mode uses the entire TrainDataset as the source dataset and creates a
    deterministic train/val split saved under ``metadata/splits``.
    """
    if split not in {"train", "val"}:
        raise ValueError(f"Local train/val builder does not support split='{split}'.")

    local_root = resolve_local_data_root(
        data_cfg.get("root", "."),
        train_dir=data_cfg.get("train_dir", "TrainDataset"),
        test_dir=data_cfg.get("test_dir", "TestDataset"),
    )
    train_root = local_root / data_cfg.get("train_dir", "TrainDataset")
    image_dir = data_cfg.get("train_image_dir", data_cfg.get("image_dir", "image"))
    mask_dir = data_cfg.get("train_mask_dir", data_cfg.get("mask_dir", "mask"))
    image_dir_candidates = data_cfg.get(
        "train_image_dir_candidates",
        data_cfg.get("image_dir_candidates", ["image", "images"]),
    )
    mask_dir_candidates = data_cfg.get(
        "train_mask_dir_candidates",
        data_cfg.get("mask_dir_candidates", ["mask", "masks"]),
    )
    strict_pairing = bool(data_cfg.get("strict_pairing", True))

    all_samples = build_dataset_samples(
        root=train_root,
        image_dir=image_dir,
        mask_dir=mask_dir,
        split="train",
        dataset_name=str(data_cfg.get("name", "local_train")),
        image_dir_candidates=image_dir_candidates,
        mask_dir_candidates=mask_dir_candidates,
        strict_pairing=strict_pairing,
    )
    split_cfg = dict(data_cfg.get("local_split", {}))
    split_seed = int(split_cfg.get("seed", seed))
    val_fraction = float(split_cfg.get("val_fraction", 0.1))
    split_root = split_cfg.get("cache_dir", "metadata/splits")
    split_files = materialize_deterministic_split_files(
        image_ids=[sample.image_id for sample in all_samples],
        split_root=split_root,
        dataset_name=str(data_cfg.get("name", "local_train")),
        seed=split_seed,
        val_fraction=val_fraction,
    )

    return build_binary_segmentation_dataset(
        root=str(train_root),
        image_dir=image_dir,
        mask_dir=mask_dir,
        split=split,
        dataset_name=str(data_cfg.get("name", "local_train")),
        transforms=transforms,
        split_file=str(split_files[split]),
        image_dir_candidates=image_dir_candidates,
        mask_dir_candidates=mask_dir_candidates,
        strict_pairing=strict_pairing,
    )


def discover_local_test_datasets(data_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Discover valid immediate test datasets under ``TestDataset/*``.

    A valid dataset is any child directory that contains both an image folder and
    a mask folder, allowing singular/plural naming variants.
    """
    local_root = resolve_local_data_root(
        data_cfg.get("root", "."),
        train_dir=data_cfg.get("train_dir", "TrainDataset"),
        test_dir=data_cfg.get("test_dir", "TestDataset"),
    )
    test_root = local_root / data_cfg.get("test_dir", "TestDataset")
    if not test_root.is_dir():
        raise FileNotFoundError(f"Local test dataset root not found: {test_root}")

    discovered: Dict[str, Dict[str, Any]] = {}
    image_dir_candidates = data_cfg.get(
        "test_image_dir_candidates",
        data_cfg.get("image_dir_candidates", ["image", "images"]),
    )
    mask_dir_candidates = data_cfg.get(
        "test_mask_dir_candidates",
        data_cfg.get("mask_dir_candidates", ["mask", "masks"]),
    )

    for child in sorted(test_root.iterdir()):
        if not child.is_dir():
            continue
        try:
            img_dir, msk_dir = resolve_dataset_dirs(
                root=child,
                split="test",
                image_dir=str(next(iter(candidate_dir_names(None, image_dir_candidates)))),
                mask_dir=str(next(iter(candidate_dir_names(None, mask_dir_candidates)))),
                image_dir_candidates=image_dir_candidates,
                mask_dir_candidates=mask_dir_candidates,
            )
            samples = build_dataset_samples(
                root=child,
                image_dir=img_dir.name,
                mask_dir=msk_dir.name,
                split="test",
                dataset_name=child.name,
                image_dir_candidates=image_dir_candidates,
                mask_dir_candidates=mask_dir_candidates,
                strict_pairing=bool(data_cfg.get("strict_pairing", True)),
            )
        except (FileNotFoundError, ValueError, StopIteration):
            continue

        if not samples:
            continue

        discovered[child.name] = {
            "name": child.name,
            "root": str(child),
            "image_dir": img_dir.name,
            "mask_dir": msk_dir.name,
            "image_dir_candidates": image_dir_candidates,
            "mask_dir_candidates": mask_dir_candidates,
            "image_size": data_cfg.get("image_size", 352),
            "num_workers": data_cfg.get("num_workers", 4),
            "pin_memory": data_cfg.get("pin_memory", True),
            "strict_pairing": bool(data_cfg.get("strict_pairing", True)),
        }

    if not discovered:
        raise FileNotFoundError(
            f"No valid test datasets were discovered under {test_root}."
        )

    return discovered


def build_binary_segmentation_dataset(
    root: str,
    image_dir: str,
    mask_dir: str,
    split: str,
    dataset_name: str,
    transforms: Optional[Any] = None,
    split_file: Optional[str] = None,
    image_dir_candidates: Optional[Iterable[str]] = None,
    mask_dir_candidates: Optional[Iterable[str]] = None,
    strict_pairing: bool = False,
) -> BinarySegmentationDataset:
    """
    Build a dataset instance from path configuration.

    This function serves as the canonical dataset factory used by the registry
    and experiment scripts.
    """
    samples = build_dataset_samples(
        root=resolve_path(root),
        image_dir=image_dir,
        mask_dir=mask_dir,
        split=split,
        dataset_name=dataset_name,
        split_file=split_file,
        image_dir_candidates=image_dir_candidates,
        mask_dir_candidates=mask_dir_candidates,
        strict_pairing=strict_pairing,
    )
    return BinarySegmentationDataset(samples=samples, transforms=transforms)
