"""
Checkpoint save/load utilities.

Checkpointing should preserve:
- student weights,
- projector weights,
- optimizer and scheduler states,
- experiment config,
- random seed and training metadata.

This enables faithful resume and auditability.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("crisp")


def save_checkpoint(
    path: Path,
    state: Dict[str, Any],
) -> None:
    """
    Save a training checkpoint to disk.

    Parameters
    ----------
    path:
        File path for the checkpoint.
    state:
        Dictionary containing model, optimizer, scheduler, and metadata state.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    logger.info("Checkpoint saved to %s", path)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    """
    Load a previously saved checkpoint and return the state dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    logger.info("Checkpoint loaded from %s", path)
    return state


def load_required_projector_state(
    projector: Optional[nn.Module], checkpoint: Dict[str, Any], checkpoint_path: Path
) -> None:
    """Require a compatible learned projector before CRISP projector evaluation."""
    context = f"Projector-on CRISP evaluation from checkpoint '{checkpoint_path}'"
    if projector is None:
        raise ValueError(f"{context} requires a projector module, but none was built.")
    if "projector_state_dict" not in checkpoint:
        raise ValueError(f"{context} requires projector_state_dict, but the key is missing.")
    state = checkpoint["projector_state_dict"]
    if state is None:
        raise ValueError(f"{context} requires a learned projector, but projector_state_dict is None.")
    if not isinstance(state, Mapping):
        raise ValueError(f"{context} requires a projector_state_dict mapping, got {type(state).__name__}.")
    try:
        projector.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"{context} has an incompatible projector_state_dict: {exc}") from exc
