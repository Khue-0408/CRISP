"""Fail-loud execution contracts for controlled current CRISP experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from crisp.data.evaluation_membership import (
    CURRENT_EVALUATION_COUNTS,
    canonical_evaluation_dataset_name,
)
from crisp.utils.paths import resolve_path


CURRENT_PROTOCOL_PROFILE = "current_crisp"
CURRENT_EVALUATION_SUITE = frozenset(CURRENT_EVALUATION_COUNTS)
CURRENT_RETAINED_STUDENTS = frozenset({"unet", "unetpp", "pranet"})


def _uses_current_protocol(config: Mapping[str, Any]) -> bool:
    profile = config.get("protocol_profile")
    if profile is None:
        return False
    if profile != CURRENT_PROTOCOL_PROFILE:
        raise ValueError(
            f"Unknown protocol_profile {profile!r}; expected "
            f"{CURRENT_PROTOCOL_PROFILE!r} or no marker."
        )
    return True


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Current CRISP protocol requires {name} to be a mapping.")
    return value


def _require_manifest_file(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Current CRISP protocol requires an explicit {name} path.")
    path = resolve_path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Current CRISP protocol {name} not found: {path}")
    return path


def validate_current_training_protocol(config: Mapping[str, Any]) -> None:
    """Reject implicit or fraction-derived source membership for current runs."""
    if not _uses_current_protocol(config):
        return

    model = config.get("model")
    if isinstance(model, Mapping) and model.get("name") is not None:
        model_name = str(model["name"]).lower()
        if model_name not in CURRENT_RETAINED_STUDENTS:
            raise ValueError(
                "Current CRISP protocol execution is limited to the retained U-Net, "
                f"U-Net++, and PraNet students; received model.name={model_name!r}. "
                "External stronger-host adapters require separate validation."
            )

    source_data = _require_mapping(config.get("source_data"), "source_data")
    source_split = _require_mapping(source_data.get("source_split"), "source_data.source_split")
    mode = source_split.get("mode")
    if mode != "manifest":
        raise ValueError(
            "Current CRISP protocol requires source_data.source_split.mode='manifest'; "
            f"received {mode!r}. Fraction or implicit source membership is not valid "
            "current-protocol evidence."
        )
    if source_split.get("count_profile") != CURRENT_PROTOCOL_PROFILE:
        raise ValueError(
            "Current CRISP protocol requires "
            "source_data.source_split.count_profile='current_crisp'."
        )

    legacy_split = source_data.get("local_split")
    if isinstance(legacy_split, Mapping):
        forbidden = sorted(set(legacy_split) & {"val_fraction", "seed", "cache_dir"})
        if forbidden:
            raise ValueError(
                "Current CRISP protocol forbids legacy local_split membership controls: "
                f"{forbidden}."
            )

    _require_manifest_file(source_split.get("train_manifest"), "source train manifest")
    _require_manifest_file(source_split.get("val_manifest"), "source validation manifest")


def validate_current_evaluation_protocol(config: Mapping[str, Any]) -> None:
    """Require the exact controlled suite and explicit membership before evaluation."""
    if not _uses_current_protocol(config):
        return

    requested = config.get("eval_datasets")
    if not isinstance(requested, (list, tuple)):
        raise ValueError("Current CRISP protocol requires an explicit eval_datasets list.")
    if any(not isinstance(name, str) or not name for name in requested):
        raise ValueError(
            "Current CRISP protocol evaluation dataset names must be nonempty strings."
        )

    canonical = [canonical_evaluation_dataset_name(name) for name in requested]
    if len(canonical) != len(set(canonical)):
        raise ValueError(
            "Current CRISP protocol evaluation aliases resolve to duplicate scientific identities."
        )
    noncanonical = [name for name, identity in zip(requested, canonical) if name != identity]
    if noncanonical:
        raise ValueError(
            "Current CRISP protocol requires canonical evaluation dataset identities; "
            f"replace storage aliases {noncanonical}."
        )
    if set(canonical) != CURRENT_EVALUATION_SUITE:
        missing = sorted(CURRENT_EVALUATION_SUITE - set(canonical))
        extra = sorted(set(canonical) - CURRENT_EVALUATION_SUITE)
        raise ValueError(
            "Current CRISP protocol requires exactly Kvasir-SEG, CVC-ClinicDB, CVC-300, "
            f"CVC-ColonDB, and ETIS; missing={missing}, extra={extra}."
        )

    eval_config = _require_mapping(config.get("eval"), "eval")
    if bool(eval_config.get("skip_missing_datasets", False)):
        raise ValueError("Current CRISP protocol cannot skip missing controlled datasets.")
    if eval_config.get("membership_count_profile") != CURRENT_PROTOCOL_PROFILE:
        raise ValueError(
            "Current CRISP protocol requires eval.membership_count_profile='current_crisp'."
        )

    manifests = _require_mapping(
        eval_config.get("membership_manifests"), "eval.membership_manifests"
    )
    manifest_names = set(manifests)
    if manifest_names != CURRENT_EVALUATION_SUITE:
        missing = sorted(CURRENT_EVALUATION_SUITE - manifest_names)
        extra = sorted(manifest_names - CURRENT_EVALUATION_SUITE)
        raise ValueError(
            "Current CRISP protocol requires one explicit evaluation manifest per "
            "controlled dataset; "
            f"missing={missing}, extra={extra}."
        )
    for dataset_name in sorted(CURRENT_EVALUATION_SUITE):
        _require_manifest_file(manifests[dataset_name], f"evaluation manifest for {dataset_name}")


def validate_current_evaluation_membership_mode(
    config: Mapping[str, Any], dataset_name: str, membership: Mapping[str, Any]
) -> None:
    """Reject any current-protocol dataset that did not consume its explicit manifest."""
    if not _uses_current_protocol(config):
        return
    if dataset_name not in CURRENT_EVALUATION_SUITE:
        raise ValueError(f"Unsupported current-protocol evaluation dataset: {dataset_name!r}.")
    if membership.get("dataset") != dataset_name:
        raise ValueError(
            f"Current-protocol evaluation membership identity conflicts with {dataset_name!r}."
        )
    if membership.get("mode") != "explicit_manifest":
        raise ValueError(
            f"Current CRISP protocol requires explicit evaluation membership for {dataset_name}; "
            f"received mode={membership.get('mode')!r}."
        )
