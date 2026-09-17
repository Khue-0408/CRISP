"""Explicit, dataset-qualified source membership validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Iterable, Mapping

from crisp.utils.paths import resolve_path


_SOURCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")

CURRENT_SOURCE_COUNT_PROFILE = {
    "pool": {"Kvasir-SEG": 900, "CVC-ClinicDB": 550},
    "train": {"Kvasir-SEG": 810, "CVC-ClinicDB": 495},
    "val": {"Kvasir-SEG": 90, "CVC-ClinicDB": 55},
}


@dataclass(frozen=True)
class SourceManifest:
    path: Path
    ids: tuple[str, ...]
    sha256: str

    @property
    def count(self) -> int:
        return len(self.ids)


def source_dataset_name(sample_id: str) -> str:
    """Require an exact ``<dataset>/<image-and-mask-stem>`` identifier."""
    if not _SOURCE_ID.fullmatch(sample_id):
        raise ValueError(f"Malformed source sample ID: {sample_id!r}")
    return sample_id.split("/", 1)[0]


def read_source_manifest(path: str | Path) -> SourceManifest:
    """Preserve manifest order and hash normalized, path-independent content."""
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Source split manifest not found: {resolved}")
    ids = tuple(line.strip() for line in resolved.read_text(encoding="utf-8").splitlines() if line.strip())
    if not ids:
        raise ValueError(f"Source split manifest is empty: {resolved}")
    for sample_id in ids:
        source_dataset_name(sample_id)
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate source sample ID in manifest: {resolved}")
    normalized = "\n".join(ids) + "\n"
    return SourceManifest(resolved, ids, hashlib.sha256(normalized.encode("utf-8")).hexdigest())


def validate_source_membership(
    available_ids: Iterable[str],
    train_ids: Iterable[str],
    val_ids: Iterable[str],
    supported_datasets: Iterable[str],
    expected_counts: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, dict[str, int]]:
    """Validate explicit membership, its full complement, and optional counts."""
    available = tuple(available_ids)
    train = tuple(train_ids)
    val = tuple(val_ids)
    supported = set(supported_datasets)
    if not supported:
        raise ValueError("Source split requires at least one supported dataset.")
    for group, ids in (("pool", available), ("train", train), ("val", val)):
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate source sample ID in {group}.")
        for sample_id in ids:
            if source_dataset_name(sample_id) not in supported:
                raise ValueError(f"Unsupported source dataset in {group}: {sample_id}")
    pool_set, train_set, val_set = set(available), set(train), set(val)
    if train_set & val_set:
        raise ValueError(f"Train/validation overlap: {sorted(train_set & val_set)[:5]}")
    unknown = (train_set | val_set) - pool_set
    if unknown:
        raise ValueError(f"Unknown source sample ID: {sorted(unknown)[:5]}")
    unassigned = pool_set - (train_set | val_set)
    if unassigned:
        raise ValueError(f"Unassigned source sample ID: {sorted(unassigned)[:5]}")

    counts = {
        group: dict(Counter(source_dataset_name(sample_id) for sample_id in ids))
        for group, ids in (("pool", available), ("train", train), ("val", val))
    }
    if expected_counts is not None:
        for group in ("pool", "train", "val"):
            if group not in expected_counts:
                raise ValueError(f"Expected source counts missing group: {group}")
            expected = dict(expected_counts[group])
            actual = {name: counts[group].get(name, 0) for name in supported}
            if expected != actual:
                raise ValueError(f"Wrong source {group} counts: expected {expected}, got {actual}")
    return counts


def validate_current_source_count_profile(
    available_ids: Iterable[str], train_ids: Iterable[str], val_ids: Iterable[str]
) -> dict[str, dict[str, int]]:
    """Check current manuscript counts without selecting any sample IDs."""
    return validate_source_membership(
        available_ids, train_ids, val_ids,
        supported_datasets=CURRENT_SOURCE_COUNT_PROFILE["pool"],
        expected_counts=CURRENT_SOURCE_COUNT_PROFILE,
    )
