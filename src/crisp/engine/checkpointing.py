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
from pathlib import Path
from typing import Any, Dict

import torch

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
