"""
Optional training and evaluation hooks.

Hooks are useful for:
- custom logging,
- qualitative prediction dumps,
- periodic solver diagnostics,
- alpha saturation monitoring.

This file exists so instrumentation logic does not pollute trainer/evaluator code.
"""

from __future__ import annotations

from typing import Dict


class Hook:
    """
    Base hook interface for experiment instrumentation.
    """

    def on_epoch_start(self, epoch: int) -> None:
        """
        Called at the start of an epoch.
        """
        return None

    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> None:
        """
        Called at the end of an epoch.
        """
        return None
