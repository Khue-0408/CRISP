"""
Path utilities for outputs, checkpoints, predictions, and metrics.

Keeping path creation centralized prevents scattered assumptions across scripts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


def ensure_dir(path: Path) -> Path:
    """
    Create a directory if it does not exist and return the path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_repo_root() -> Path:
    """
    Return the repository root as seen from runtime entry points.

    Hydra changes the process working directory during CLI execution. We anchor
    all relative paths to ``HYDRA_ORIG_CWD`` when available so dataset roots,
    checkpoint paths, and outputs remain stable across train/eval runs.
    """
    hydra_orig_cwd = os.environ.get("HYDRA_ORIG_CWD")
    if hydra_orig_cwd:
        return Path(hydra_orig_cwd).expanduser().resolve()
    return Path.cwd().expanduser().resolve()


def resolve_path(path: str | Path) -> Path:
    """
    Resolve a repository-relative or absolute path into an absolute path.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (get_repo_root() / candidate).resolve()


def resolve_local_data_root(
    root: str | Path,
    train_dir: str = "TrainDataset",
    test_dir: str = "TestDataset",
) -> Path:
    """
    Resolve the local-data workspace root for TrainDataset/TestDataset mode.

    Local users may place datasets either directly under the repository root:

    ``repo_root/TrainDataset`` and ``repo_root/TestDataset``

    or inside a dedicated data workspace:

    ``repo_root/data/TrainDataset`` and ``repo_root/data/TestDataset``.

    This helper accepts either style without changing the CRISP method path.
    """
    base = resolve_path(root)
    if (base / train_dir).is_dir() or (base / test_dir).is_dir():
        return base

    nested_data_root = base / "data"
    if (nested_data_root / train_dir).is_dir() or (nested_data_root / test_dir).is_dir():
        return nested_data_root

    return base


def build_experiment_dir(root: Path, experiment_name: str, seed: int) -> Path:
    """
    Build the canonical output directory for one experiment run.

    Structure: ``<root>/<experiment_name>/seed_<seed>/``
    """
    out = Path(root) / experiment_name / f"seed_{seed}"
    return ensure_dir(out)


def local_workspace_layout(root: str | Path) -> Dict[str, Path]:
    """
    Build the canonical local-mode workspace layout relative to ``root``.
    """
    workspace_root = resolve_path(root)
    return {
        "root": workspace_root,
        "checkpoints_root": workspace_root / "checkpoints",
        "student_checkpoints_root": workspace_root / "checkpoints" / "students",
        "teacher_checkpoints_root": workspace_root / "checkpoints" / "teachers",
        "student_unet_init_root": workspace_root / "checkpoints" / "students" / "unet" / "init",
        "student_unet_baseline_root": workspace_root / "checkpoints" / "students" / "unet" / "baseline",
        "student_unet_crisp_root": workspace_root / "checkpoints" / "students" / "unet" / "crisp",
        "teacher_pranet_root": workspace_root / "checkpoints" / "teachers" / "pranet",
        "teacher_polyp_pvt_root": workspace_root / "checkpoints" / "teachers" / "polyp_pvt",
        "teacher_uacanet_root": workspace_root / "checkpoints" / "teachers" / "uacanet",
        "outputs_root": workspace_root / "outputs",
        "runs_root": workspace_root / "outputs" / "runs",
        "metrics_root": workspace_root / "outputs" / "metrics",
        "predictions_root": workspace_root / "outputs" / "predictions",
        "posthoc_root": workspace_root / "outputs" / "posthoc",
        "tables_root": workspace_root / "outputs" / "tables",
        "metadata_root": workspace_root / "metadata",
        "metadata_splits_root": workspace_root / "metadata" / "splits",
        "logs_root": workspace_root / "logs",
        "cache_root": workspace_root / "cache",
    }


def ensure_local_workspace(root: str | Path) -> Dict[str, Path]:
    """
    Create the canonical local-mode workspace directories and return them.
    """
    layout = local_workspace_layout(root)
    for path in layout.values():
        ensure_dir(path)
    return layout
