"""Helpers for importing repo-local baseline implementations without package clashes."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import os
import sys
from types import ModuleType
from typing import Iterator


REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_ROOT = REPO_ROOT / "1_baseline"


@contextmanager
def isolated_baseline_import(root: Path, chdir: bool = False) -> Iterator[None]:
    """
    Temporarily import one baseline repo that uses the generic package name ``lib``.

    Several baseline folders contain a top-level ``lib`` package. Importing them
    normally in one Python process makes those packages collide. This context
    removes only temporary ``lib`` modules during the import and restores any
    previous modules afterwards.
    """
    root = root.resolve()
    old_path = list(sys.path)
    old_cwd = Path.cwd()
    saved_lib_modules: dict[str, ModuleType] = {
        name: module
        for name, module in sys.modules.items()
        if name == "lib" or name.startswith("lib.")
    }
    for name in list(saved_lib_modules):
        sys.modules.pop(name, None)

    sys.path.insert(0, str(root))
    if chdir:
        os.chdir(root)

    try:
        yield
    finally:
        for name in [key for key in sys.modules if key == "lib" or key.startswith("lib.")]:
            sys.modules.pop(name, None)
        sys.modules.update(saved_lib_modules)
        sys.path[:] = old_path
        if chdir:
            os.chdir(old_cwd)
