"""
Configuration loading helpers.

This module wraps OmegaConf interaction so entry-point scripts remain concise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """
    Load the experiment configuration from a YAML file.

    Parameters
    ----------
    path:
        Path to the YAML config file.  When *None*, an empty config is returned
        (useful for testing).

    Returns
    -------
    Dict[str, Any]
        Unified experiment configuration as a plain dictionary.
    """
    if path is None:
        return {}
    cfg = OmegaConf.load(path)
    # Resolve interpolations and return a plain dict.
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore[return-value]
