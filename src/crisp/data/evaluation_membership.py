"""Exact, order-independent sample membership for CRISP evaluation datasets."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

from crisp.data.source_manifest import source_dataset_name
from crisp.utils.paths import resolve_path
from crisp.utils.provenance import SCHEMA_VERSION


CURRENT_EVALUATION_COUNTS = {
    "Kvasir-SEG": 100,
    "CVC-ClinicDB": 62,
    "CVC-300": 60,
    "CVC-ColonDB": 380,
    "ETIS": 196,
}

EVALUATION_STORAGE_ALIASES = {
    "Kvasir": "Kvasir-SEG",
    "ETIS-LaribPolypDB": "ETIS",
}


def canonical_evaluation_dataset_name(name: str) -> str:
    """Resolve only known storage aliases to manuscript dataset identities."""
    if not isinstance(name, str) or not name:
        raise ValueError("Evaluation dataset name must be a nonempty string.")
    return EVALUATION_STORAGE_ALIASES.get(name, name)


def resolve_evaluation_dataset_identity(
    storage_name: str, declared_identity: str | None = None,
) -> str:
    """Keep storage lookup separate from an explicit scientific identity."""
    resolved_storage = canonical_evaluation_dataset_name(storage_name)
    if declared_identity is None:
        return resolved_storage
    if declared_identity in EVALUATION_STORAGE_ALIASES:
        raise ValueError(
            f"Explicit evaluation identity must be canonical, not storage alias {declared_identity!r}."
        )
    if resolved_storage != declared_identity:
        raise ValueError(
            "Evaluation storage/config name conflicts with explicit scientific identity: "
            f"{storage_name!r} -> {resolved_storage!r}, declared {declared_identity!r}."
        )
    return declared_identity


def _normalized_sha256(ids: Iterable[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def read_evaluation_manifest(path: str | Path, dataset: str) -> tuple[tuple[str, ...], Path, str]:
    """Read dataset-qualified IDs without selecting or generating any membership."""
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Evaluation manifest not found: {resolved}")
    ids = tuple(line.strip() for line in resolved.read_text(encoding="utf-8").splitlines() if line.strip())
    if not ids:
        raise ValueError(f"Evaluation manifest is empty: {resolved}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate evaluation sample ID in manifest: {resolved}")
    for sample_id in ids:
        if source_dataset_name(sample_id) != dataset:
            raise ValueError(f"Evaluation manifest ID has wrong dataset: {sample_id}")
    return ids, resolved, _normalized_sha256(ids)


def evaluation_membership_record(
    dataset: str, sample_ids: Iterable[str], *, mode: str,
    manifest_path: str | Path | None = None, manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Hash the exact paired-sample set, independently of discovery order."""
    if mode not in {"explicit_manifest", "discovered_full_dataset", "legacy_split_file"}:
        raise ValueError(f"Unknown evaluation membership mode: {mode}")
    ids = tuple(sample_ids)
    if not ids:
        raise ValueError("Evaluation membership cannot be empty.")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evaluation sample identity.")
    for sample_id in ids:
        if source_dataset_name(sample_id) != dataset:
            raise ValueError(f"Evaluation sample ID has wrong dataset: {sample_id}")
    if mode == "explicit_manifest" and (manifest_path is None or manifest_sha256 is None):
        raise ValueError("Explicit evaluation manifest requires path and content SHA-256.")
    if mode != "explicit_manifest" and (manifest_path is not None or manifest_sha256 is not None):
        raise ValueError("Only explicit-manifest membership may carry manifest metadata.")
    sorted_ids = sorted(ids)
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "evaluation_dataset_membership",
        "dataset": dataset,
        "split": "test",
        "sample_count": len(sorted_ids),
        "sample_ids": sorted_ids,
        "membership_sha256": _normalized_sha256(sorted_ids),
        "mode": mode,
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "manifest_sha256": manifest_sha256,
    }


def validate_evaluation_membership(record: dict[str, Any]) -> None:
    """Fail if record metadata and the normalized sample set disagree."""
    if record.get("schema_version") != SCHEMA_VERSION or record.get("record_type") != "evaluation_dataset_membership":
        raise ValueError("Invalid evaluation membership schema.")
    rebuilt = evaluation_membership_record(
        record["dataset"], record["sample_ids"], mode=record["mode"],
        manifest_path=record.get("manifest_path"), manifest_sha256=record.get("manifest_sha256"),
    )
    for key in ("split", "sample_count", "sample_ids", "membership_sha256"):
        if record.get(key) != rebuilt[key]:
            raise ValueError(f"Evaluation membership {key} conflicts with its sample IDs.")


def validate_current_evaluation_count(record: dict[str, Any]) -> None:
    """Validate manuscript counts only; never choose sample IDs."""
    validate_evaluation_membership(record)
    dataset = record["dataset"]
    if dataset not in CURRENT_EVALUATION_COUNTS:
        raise ValueError(f"No current evaluation count profile for {dataset!r}.")
    expected = CURRENT_EVALUATION_COUNTS[dataset]
    if record["sample_count"] != expected:
        raise ValueError(f"Wrong evaluation count for {dataset}: expected {expected}, got {record['sample_count']}.")
