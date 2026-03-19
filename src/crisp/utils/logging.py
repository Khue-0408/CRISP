"""
Experiment logging helpers.

This module provides:
- stdout logging,
- file logging,
- consistent metric formatting.

Clean logging is critical for reproduction and debugging.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict

_LOGGER_NAME = "crisp"


def setup_logger(output_dir: Path) -> None:
    """
    Initialize repository-wide logging outputs.

    Sets up a root ``crisp`` logger that writes to both stdout and
    ``<output_dir>/train.log``.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers on repeated calls

    fmt = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / "train.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)


def log_metrics(metrics: Dict[str, float], step: int, split: str) -> None:
    """
    Log metrics for a given step and split.

    Parameters
    ----------
    metrics:
        Metric name → value mapping.
    step:
        Global step or epoch number.
    split:
        One of ``train``, ``val``, ``test``.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    parts = [f"{k}={v:.4f}" for k, v in sorted(metrics.items())]
    logger.info("[%s] step=%d  %s", split, step, "  ".join(parts))
