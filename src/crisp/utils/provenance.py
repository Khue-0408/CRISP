"""Versioned, byte-verified identities for CRISP run artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import uuid
from pathlib import Path
from typing import Any, Mapping

from crisp.utils.paths import get_repo_root, resolve_path


SCHEMA_VERSION = "crisp_provenance_v1"


def canonical_json(value: Any) -> str:
    """Serialize resolved JSON-compatible data independently of mapping order."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_provenance(path: str | Path, record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Provenance record has an unsupported schema version.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Never replace an existing record, even for an identical scientific identity.
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_provenance(path: str | Path, record_type: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION or data.get("record_type") != record_type:
        raise ValueError(f"Invalid {record_type} provenance record: {path}")
    return data


def git_state(repo: Path | None = None) -> dict[str, Any]:
    root = repo or get_repo_root()

    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=root, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    sha = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current") if sha else None
    status = git("status", "--porcelain") if sha else None
    return {"sha": sha, "branch": branch, "dirty": bool(status) if status is not None else None}


def source_split_provenance(config: Mapping[str, Any], train_dataset: Any = None) -> dict[str, Any]:
    data = config.get("source_data", {})
    split = data.get("source_split", {})
    mode = split.get("mode", "fraction" if data.get("mode") == "local_train_test" else None)
    actual = getattr(train_dataset, "source_split_provenance", None)
    if mode == "manifest":
        if not isinstance(actual, dict) or actual.get("mode") != "manifest":
            raise ValueError("Manifest-backed run requires validated dataset split provenance.")
        return dict(actual)
    if actual is not None:
        raise ValueError("Dataset split provenance conflicts with resolved source split mode.")
    if mode == "fraction" or data.get("mode") == "local_train_test":
        local = data.get("local_split", {})
        return {
            "mode": "fraction",
            "seed": int(local.get("seed", config.get("seed", 0))),
            "val_fraction": float(local.get("val_fraction", 0.1)),
            "manifest_paths": None,
            "manifest_hashes": None,
            "counts": None,
        }
    return {"mode": mode, "manifest_paths": None, "manifest_hashes": None, "counts": None}


def teacher_provenance(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = config.get("teachers") or config.get("teacher_pool", {}).get("teachers") or config.get("crisp", {}).get("teacher", {}).get("teachers") or []
    teachers = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("enabled", True):
            continue
        model = entry.get("model_config") or entry.get("model")
        checkpoint = entry.get("checkpoint")
        path = resolve_path(str(checkpoint)) if checkpoint and str(checkpoint).strip() else None
        exists = bool(path and path.is_file())
        loading = entry.get("checkpoint_loading") or {}
        teachers.append({
            "name": entry.get("name"),
            "model": model,
            "model_class": model.get("class_path", model.get("name")) if isinstance(model, dict) else model,
            "checkpoint_path": str(path) if path else None,
            "checkpoint_loading": {
                "strict": bool(loading.get("strict", True)),
                "state_dict_keys": loading.get("state_dict_keys"),
                "prefixes_to_strip": loading.get("prefixes_to_strip"),
                "auto_download": bool(entry.get("download", {}).get("enabled", False)),
            },
            "artifact_status": "available" if exists else "missing",
            "checkpoint_sha256": file_sha256(path) if exists else None,
            "origin": None,
            "training_exposure": None,
        })
    return teachers


def student_initialization_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    """Identify the optional external checkpoint by its actual file bytes."""
    init = config.get("student_init") or {}
    checkpoint = init.get("checkpoint")
    configured = str(checkpoint).strip() if checkpoint is not None else None
    path = resolve_path(configured) if configured else None
    exists = bool(path and path.is_file())
    return {
        "configured_checkpoint": checkpoint,
        "resolved_path": str(path) if path else None,
        "artifact_status": "available" if exists else "missing" if configured else "none",
        "checkpoint_sha256": file_sha256(path) if exists else None,
        "strict": bool(init.get("strict", True)),
        "state_dict_keys": init.get("state_dict_keys"),
        "prefixes_to_strip": init.get("prefixes_to_strip"),
        "download": init.get("download"),
    }


def external_artifact_identity(
    student_init: Mapping[str, Any], teachers: list[dict[str, Any]], split: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep byte-level dependencies separate from paths already in the config hash."""
    return {
        "student_initialization_sha256": student_init["checkpoint_sha256"],
        "teacher_checkpoints": [
            {"name": teacher["name"], "sha256": teacher["checkpoint_sha256"]}
            for teacher in teachers
        ],
        "source_manifests": (
            {"train_sha256": split["train_sha256"], "val_sha256": split["val_sha256"]}
            if split.get("mode") == "manifest" else None
        ),
    }


def run_provenance(
    config: Mapping[str, Any], *, train_dataset: Any = None,
    git: Mapping[str, Any] | None = None, run_id: str | None = None,
) -> dict[str, Any]:
    import torch

    git_info = dict(git) if git is not None else git_state()
    resolved = dict(config)
    config_hash = content_sha256(resolved)
    experiment = resolved.get("experiment_name")
    seed = int(resolved.get("seed", 0))
    split = source_split_provenance(resolved, train_dataset)
    teachers = teacher_provenance(resolved)
    student_init = student_initialization_provenance(resolved)
    artifacts = external_artifact_identity(student_init, teachers, split)
    identity = {
        "experiment": experiment, "seed": seed, "config_sha256": config_hash,
        "git_sha": git_info.get("sha"), "external_artifacts": artifacts,
    }
    # A unique execution suffix prevents a repeated scientific configuration from sharing a run ID.
    identity_hash = content_sha256(identity)[:16]
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "run",
        "run_id": run_id or f"{identity_hash}-{uuid.uuid4().hex}",
        "scientific_identity_sha256": content_sha256(identity),
        "experiment": experiment,
        "host": resolved.get("model"),
        "method": resolved.get("method"),
        "seed": seed,
        "git": git_info,
        "resolved_config": resolved,
        "config_sha256": config_hash,
        "source_data_role": resolved.get("source_data", {}).get("role"),
        "source_split": split,
        "teachers": teachers,
        "student_initialization": student_init,
        "external_artifact_identity": artifacts,
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
    }


def checkpoint_provenance(run: Mapping[str, Any], epoch: int, selection: Mapping[str, Any]) -> dict[str, Any]:
    if run.get("schema_version") != SCHEMA_VERSION or run.get("record_type") != "run":
        raise ValueError("Checkpoint requires a versioned run provenance record.")
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "checkpoint",
        "run_id": run["run_id"],
        "experiment": run["experiment"],
        "config_sha256": run["config_sha256"],
        "git_sha": run["git"].get("sha"),
        "seed": run["seed"],
        "epoch": epoch,
        "selection": dict(selection),
        "source_split": run["source_split"],
        "teachers": run["teachers"],
        "student_initialization": run["student_initialization"],
    }


def evaluation_provenance(
    config: Mapping[str, Any], checkpoint_path: str | Path, checkpoint: Mapping[str, Any],
    dataset: str, mode: str, metric_path: str | Path, *, git: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    parent = checkpoint.get("provenance")
    experiment = config.get("experiment_name")
    seed = int(config.get("seed", 0))
    if parent is not None:
        if parent.get("record_type") != "checkpoint" or parent.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Invalid checkpoint provenance.")
        for key, expected in (("experiment", experiment), ("seed", seed)):
            if parent.get(key) != expected:
                raise ValueError(f"Checkpoint {key} conflicts with evaluation configuration.")
        if parent.get("config_sha256") != content_sha256(checkpoint["config"]):
            raise ValueError("Checkpoint config hash conflicts with embedded resolved config.")
        for key in ("seed", "epoch"):
            if key in checkpoint and parent.get(key) != checkpoint[key]:
                raise ValueError(f"Checkpoint {key} conflicts with embedded provenance.")
    if mode not in {"projector_on", "projector_off"}:
        raise ValueError(f"Unsupported evaluation mode: {mode}")
    path = Path(checkpoint_path)
    checkpoint_hash = file_sha256(path)
    if parent is not None and parent.get("checkpoint_sha256") not in (None, checkpoint_hash):
        raise ValueError("Checkpoint byte SHA-256 conflicts with checkpoint provenance.")
    eval_hash = content_sha256(dict(config))
    git_info = dict(git) if git is not None else git_state()
    identity = {
        "run_id": parent.get("run_id") if parent else None,
        "checkpoint_sha256": checkpoint_hash,
        "evaluation_config_sha256": eval_hash,
        "dataset": dataset,
        "mode": mode,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "evaluation",
        "evaluation_id": content_sha256(identity),
        "run_id": parent.get("run_id") if parent else None,
        "experiment": experiment,
        "seed": seed,
        "dataset": dataset,
        "mode": mode,
        "checkpoint_path": str(path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "config_sha256": parent.get("config_sha256") if parent else None,
        "evaluation_config_sha256": eval_hash,
        "metric_config": config.get("eval"),
        "source_file": str(metric_path),
        "git_sha": git_info.get("sha"),
    }
