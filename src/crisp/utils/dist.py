"""
Distributed training helpers.

This file is optional in the earliest implementation phase but should exist
from the beginning so scaling to multi-GPU training does not require
restructuring the codebase.
"""

from __future__ import annotations

import torch.distributed as dist


def is_distributed() -> bool:
    """
    Return whether the current process is running in distributed mode.
    """
    return dist.is_available() and dist.is_initialized()
