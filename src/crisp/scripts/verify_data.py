"""Lightweight verifier for the TrainDataset/TestDataset folder protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from crisp.data.datasets import build_dataset_samples, resolve_dataset_dirs
from crisp.data.io_utils import candidate_dir_names
from crisp.utils.paths import resolve_local_data_root


def _count_pairs(root: Path, dataset_name: str, split: str) -> int:
    img_dir, mask_dir = resolve_dataset_dirs(
        root=root,
        split=split,
        image_dir="image",
        mask_dir="mask",
        image_dir_candidates=["image", "images"],
        mask_dir_candidates=["mask", "masks"],
    )
    samples = build_dataset_samples(
        root=root,
        image_dir=img_dir.name,
        mask_dir=mask_dir.name,
        split=split,
        dataset_name=dataset_name,
        image_dir_candidates=["image", "images"],
        mask_dir_candidates=["mask", "masks"],
        strict_pairing=True,
    )
    return len(samples)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify CRISP local data layout without enforcing thesis benchmark counts."
        )
    )
    parser.add_argument("--root", default="./data", help="Data root containing TrainDataset/TestDataset.")
    parser.add_argument("--train-dir", default="TrainDataset")
    parser.add_argument("--test-dir", default="TestDataset")
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Do not require a fixed set of benchmark dataset names or counts.",
    )
    args = parser.parse_args()

    data_root = resolve_local_data_root(
        args.root,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
    )
    train_root = data_root / args.train_dir
    test_root = data_root / args.test_dir

    train_count = _count_pairs(train_root, dataset_name=args.train_dir, split="train")
    print(f"{args.train_dir}: {train_count} matched image/mask pairs")

    if not test_root.is_dir():
        raise FileNotFoundError(f"Test dataset root not found: {test_root}")

    valid_children: list[tuple[str, int]] = []
    skipped_children: list[str] = []
    for child in sorted(test_root.iterdir()):
        if not child.is_dir():
            continue
        try:
            count = _count_pairs(child, dataset_name=child.name, split="test")
        except (FileNotFoundError, ValueError, StopIteration) as exc:
            names = ", ".join(candidate_dir_names(None, ["image", "images"]))
            masks = ", ".join(candidate_dir_names(None, ["mask", "masks"]))
            skipped_children.append(
                f"{child.name}: {exc} (expected image dirs: {names}; mask dirs: {masks})"
            )
            continue
        valid_children.append((child.name, count))

    if not valid_children:
        detail = "\n".join(skipped_children)
        raise FileNotFoundError(
            f"No valid test datasets were found under {test_root}."
            + (f"\nSkipped:\n{detail}" if detail else "")
        )

    for name, count in valid_children:
        print(f"{args.test_dir}/{name}: {count} matched image/mask pairs")
    if skipped_children:
        print("Skipped non-dataset folders:")
        for item in skipped_children:
            print(f"- {item}")

    if args.non_strict:
        print("Non-strict mode: no fixed benchmark dataset names or counts were enforced.")


if __name__ == "__main__":
    main()
