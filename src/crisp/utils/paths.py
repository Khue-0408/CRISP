"""
Path utilities for outputs, checkpoints, predictions, and metrics.

Keeping path creation centralized prevents scattered assumptions across scripts.
"""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """
    Create a directory if it does not exist and return the path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_experiment_dir(root: Path, experiment_name: str, seed: int) -> Path:
    """
    Build the canonical output directory for one experiment run.

    Structure: ``<root>/<experiment_name>/seed_<seed>/``
    """
    out = Path(root) / experiment_name / f"seed_{seed}"
    return ensure_dir(out)
