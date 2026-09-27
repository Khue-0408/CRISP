"""Public release naming, launcher, and workflow contracts."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_crisp_five_seeds.sh"
SEEDS = (2026, 2027, 2028, 2029, 2030)


def _bash() -> str:
    executable = shutil.which("bash")
    if executable is None and os.name == "nt":
        for root in (os.environ.get("ProgramFiles"), os.environ.get("ProgramW6432")):
            if root:
                candidate = Path(root) / "Git" / "bin" / "bash.exe"
                if candidate.is_file():
                    executable = str(candidate)
                    break
    if executable is None:
        pytest.skip("bash is not available")
    return executable


def _runner_fixture(tmp_path: Path, *, fail_seed: int | None = None) -> tuple[Path, Path]:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    runner = scripts / RUNNER.name
    runner.write_text(RUNNER.read_text(encoding="utf-8"), encoding="utf-8")
    log = tmp_path / "runs.log"

    body = """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' \"$*\" >> \"$CRISP_RUN_LOG\"
"""
    if fail_seed is not None:
        body += f'[[ "$1" != "seed={fail_seed}" ]]\n'

    for host in ("unet", "unetpp", "pranet"):
        for mode in ("baseline", "crisp"):
            wrapper = scripts / f"train_crisp_{host}_{mode}.sh"
            wrapper.write_text(body, encoding="utf-8")
    return runner, log


def _run(runner: Path, log: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "CRISP_RUN_LOG": str(log)}
    return subprocess.run(
        [_bash(), str(runner), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.mark.parametrize("host", ["unet", "unetpp", "pranet"])
@pytest.mark.parametrize("mode", ["baseline", "crisp"])
def test_five_seed_runner_invokes_exact_seed_set_and_forwards_overrides(
    tmp_path: Path, host: str, mode: str
) -> None:
    runner, log = _runner_fixture(tmp_path)
    result = _run(runner, log, host, mode, "training.batch_size=2", "+note=public")

    assert result.returncode == 0, result.stderr
    assert log.read_text(encoding="utf-8").splitlines() == [
        f"seed={seed} training.batch_size=2 +note=public" for seed in SEEDS
    ]


def test_five_seed_runner_rejects_unsupported_host_or_mode(tmp_path: Path) -> None:
    runner, log = _runner_fixture(tmp_path)
    result = _run(runner, log, "rabbit", "crisp")

    assert result.returncode == 2
    assert "Unsupported host/mode" in result.stderr
    assert not log.exists()


def test_five_seed_runner_stops_on_first_failure(tmp_path: Path) -> None:
    runner, log = _runner_fixture(tmp_path, fail_seed=2028)
    result = _run(runner, log, "unet", "baseline")

    assert result.returncode != 0
    assert log.read_text(encoding="utf-8").splitlines() == [
        "seed=2026",
        "seed=2027",
        "seed=2028",
    ]


def test_ci_uses_python_310_and_cpu_safe_release_gates() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert 'python-version: "3.10"' in workflow
    assert "python -m compileall -q src tests" in workflow
    for test_file in (
        "test_protocol_guard.py",
        "test_source_manifest.py",
        "test_evaluation_membership.py",
        "test_posthoc_calibration.py",
        "test_locked_training_values.py",
        "test_solver.py",
        "test_metrics.py",
    ):
        assert test_file in workflow
    assert "train_crisp" not in workflow
