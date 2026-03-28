"""
Serialization helpers for JSON, CSV, and YAML artifacts.

The replication package should export:
- per-run metrics,
- aggregated tables,
- config snapshots,
- optional qualitative summaries.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


def save_json(path: Path, data: Dict[str, Any]) -> None:
    """
    Save a dictionary as a JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """
    Save a list of row dictionaries as a CSV file.

    All rows must share the same set of keys.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    """
    Save a dictionary as a YAML file.

    This is used for resolved config snapshots so training/evaluation runs keep
    a human-readable record of the exact composed experiment configuration.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
