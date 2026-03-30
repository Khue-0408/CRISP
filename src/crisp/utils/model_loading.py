"""
Model checkpoint loading helpers.

This utility centralizes the minimal robustness needed for student/teacher
initialization from external repositories without changing any CRISP method
logic. It only normalizes checkpoint container structure and common key
prefixes before calling ``load_state_dict``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


COMMON_STATE_DICT_KEYS = ("model_state_dict", "state_dict", "model")
COMMON_PREFIXES = ("module.", "model.")


def _looks_like_state_dict(candidate: Any) -> bool:
    return isinstance(candidate, Mapping) and all(
        isinstance(key, str) for key in candidate.keys()
    )


def extract_state_dict(
    checkpoint: Any,
    state_dict_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Extract a raw state_dict from a checkpoint object.

    Supported checkpoint styles:
    - plain state_dict
    - nested under ``model_state_dict``
    - nested under ``state_dict``
    - nested under ``model``
    """
    keys = tuple(state_dict_keys or COMMON_STATE_DICT_KEYS)

    if _looks_like_state_dict(checkpoint) and any(
        torch.is_tensor(value) for value in checkpoint.values()
    ):
        return dict(checkpoint)

    if isinstance(checkpoint, Mapping):
        for key in keys:
            candidate = checkpoint.get(key)
            if _looks_like_state_dict(candidate):
                return dict(candidate)

    raise ValueError(
        "Could not extract a model state_dict from checkpoint. "
        f"Tried keys: {list(keys)}."
    )


def strip_state_dict_prefixes(
    state_dict: Mapping[str, Any],
    prefixes_to_strip: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Remove common wrapper prefixes such as ``module.`` or ``model.``.
    """
    prefixes = tuple(prefixes_to_strip or COMMON_PREFIXES)
    normalized: dict[str, Any] = {}

    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if prefix and new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        normalized[new_key] = value

    return normalized


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
    state_dict_keys: Sequence[str] | None = None,
    prefixes_to_strip: Sequence[str] | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, list[str]]:
    """
    Load a checkpoint into ``model`` with minimal normalization.

    Returns missing/unexpected key diagnostics from ``load_state_dict``.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = extract_state_dict(checkpoint, state_dict_keys=state_dict_keys)
    state_dict = strip_state_dict_prefixes(
        state_dict,
        prefixes_to_strip=prefixes_to_strip,
    )

    model_keys = set(model.state_dict().keys())
    matched_keys = model_keys & set(state_dict.keys())
    if not matched_keys:
        raise RuntimeError(
            f"Checkpoint '{path}' does not contain any parameter keys matching "
            f"{type(model).__name__}."
        )

    try:
        incompatible = model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load checkpoint '{path}' into {type(model).__name__}: {exc}"
        ) from exc

    return {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
