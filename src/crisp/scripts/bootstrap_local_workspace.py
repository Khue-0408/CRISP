"""
Bootstrap the local-mode workspace layout.

This script creates the practical directory tree needed for the local U-Net
baseline/CRISP workflow:
- checkpoint destinations for students and teachers,
- output/metrics/prediction folders,
- metadata split cache,
- local TrainDataset/TestDataset folders when absent.

It does not download any weights by itself; downloading remains an explicit,
config-driven action during checkpoint loading.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from crisp.utils.paths import ensure_dir, ensure_local_workspace, resolve_local_data_root, resolve_path


def bootstrap_local_workspace(root: str | Path = ".") -> dict[str, Path]:
    """
    Create the local workspace directories and return the resulting layout.
    """
    workspace = ensure_local_workspace(root)

    data_root = resolve_local_data_root(root)
    if not data_root.exists():
        data_root = ensure_dir(resolve_path(root))

    train_root = ensure_dir(data_root / "TrainDataset")
    ensure_dir(train_root / "image")
    ensure_dir(train_root / "mask")
    ensure_dir(data_root / "TestDataset")

    return workspace


def main() -> None:
    parser = argparse.ArgumentParser(description="Create local CRISP workspace folders.")
    parser.add_argument(
        "--root",
        default=".",
        help="Workspace root. Relative paths are anchored to the repository root.",
    )
    args = parser.parse_args()

    layout = bootstrap_local_workspace(args.root)
    for name, path in sorted(layout.items()):
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
