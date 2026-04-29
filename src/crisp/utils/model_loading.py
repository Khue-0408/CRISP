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
import shutil
from typing import Any
from urllib.request import urlopen

import torch
import torch.nn as nn

from crisp.utils.paths import ensure_dir, resolve_path


COMMON_STATE_DICT_KEYS = ("model_state_dict", "state_dict", "model", "net")
COMMON_PREFIXES = ("module.", "model.")


def _looks_like_state_dict(candidate: Any) -> bool:
    return isinstance(candidate, Mapping) and all(
        isinstance(key, str) for key in candidate.keys()
    )


def _looks_like_raw_tensor_state_dict(candidate: Any) -> bool:
    return _looks_like_state_dict(candidate) and bool(candidate) and all(
        torch.is_tensor(value) for value in candidate.values()
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
    keys = tuple(dict.fromkeys([*(state_dict_keys or ()), *COMMON_STATE_DICT_KEYS]))

    if isinstance(checkpoint, Mapping):
        for key in keys:
            candidate = checkpoint.get(key)
            if _looks_like_raw_tensor_state_dict(candidate):
                return dict(candidate)

    if _looks_like_raw_tensor_state_dict(checkpoint):
        return dict(checkpoint)

    raise ValueError(
        "Could not extract a model state_dict from checkpoint. "
        f"Tried keys: {list(keys)}."
    )


def remap_legacy_unet_state_dict_keys(
    state_dict: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Remap known legacy U-Net naming variants into this repository's wrapper names.

    This supports external U-Net checkpoints that use module names such as:
    - ``double_conv`` instead of ``block``
    - ``maxpool_conv`` instead of ``pool_conv``
    - ``outc.conv`` instead of ``out_conv``
    """
    remapped: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key
        new_key = new_key.replace(".double_conv.", ".block.")
        new_key = new_key.replace(".maxpool_conv.", ".pool_conv.")
        if new_key.startswith("inc.double_conv."):
            new_key = new_key.replace("inc.double_conv.", "inc.block.", 1)
        if new_key.startswith("outc.conv."):
            new_key = new_key.replace("outc.conv.", "out_conv.", 1)
        remapped[new_key] = value
    return remapped


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
    auto_download: bool = False,
    download_url: str | None = None,
    description: str = "checkpoint",
) -> dict[str, list[str]]:
    """
    Load a checkpoint into ``model`` with minimal normalization.

    Returns missing/unexpected key diagnostics from ``load_state_dict``.
    """
    path = resolve_checkpoint_path(
        checkpoint_path=checkpoint_path,
        auto_download=auto_download,
        download_url=download_url,
        description=description,
    )

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = extract_state_dict(checkpoint, state_dict_keys=state_dict_keys)
    state_dict = strip_state_dict_prefixes(
        state_dict,
        prefixes_to_strip=prefixes_to_strip,
    )
    state_dict = remap_legacy_unet_state_dict_keys(state_dict)

    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    matched_keys = model_keys & set(state_dict.keys())
    if not matched_keys:
        raise RuntimeError(
            f"Checkpoint '{path}' does not contain any parameter keys matching "
            f"{type(model).__name__}."
        )

    if not strict:
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if key in model_state
            and hasattr(value, "shape")
            and value.shape == model_state[key].shape
        }
        if not state_dict:
            raise RuntimeError(
                f"Checkpoint '{path}' does not contain any shape-compatible parameter "
                f"keys matching {type(model).__name__}."
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


def resolve_checkpoint_path(
    checkpoint_path: str | Path,
    *,
    auto_download: bool = False,
    download_url: str | None = None,
    description: str = "checkpoint",
) -> Path:
    """
    Resolve a local checkpoint path and optionally download it when missing.
    """
    path = resolve_path(checkpoint_path)
    if path.exists():
        return path

    if auto_download:
        if download_url is None or not str(download_url).strip():
            raise FileNotFoundError(
                f"Missing {description} at {path}, and auto-download was enabled "
                "without a valid download URL."
            )
        download_checkpoint(download_url=download_url, destination=path)
        if path.exists():
            return path

    raise FileNotFoundError(f"{description.capitalize()} not found: {path}")


def download_checkpoint(
    download_url: str,
    destination: str | Path,
    timeout_sec: float = 120.0,
) -> Path:
    """
    Download a checkpoint into ``destination``.

    The download path is explicit and opt-in so the main CRISP configs still
    fail loudly when teacher assets are missing.
    """
    url = str(download_url).strip()
    if not url:
        raise ValueError("download_url must be a non-empty string.")

    destination_path = Path(destination)
    ensure_dir(destination_path.parent)
    with urlopen(url, timeout=timeout_sec) as response, open(destination_path, "wb") as f:
        shutil.copyfileobj(response, f)
    return destination_path
