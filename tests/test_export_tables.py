"""
Unit tests for table export wiring.

These tests ensure:
- projector_on and projector_off JSON files remain separated,
- exported aggregation preserves dataset/mode keys,
- optional experiment/seed metadata inference does not crash.
"""

import json
from pathlib import Path

from crisp.scripts.export_tables import _collect_metric_files


def test_collect_metric_files_parses_modes(tmp_path: Path) -> None:
    # Create a pseudo run directory structure.
    run_dir = tmp_path / "taskA_pranet_crisp" / "seed_0" / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "colondb_projector_on.json").write_text(json.dumps({"dice": 0.9, "ece": 0.1}))
    (run_dir / "colondb_projector_off.json").write_text(json.dumps({"dice": 0.88, "ece": 0.12}))

    rows = _collect_metric_files(tmp_path)
    modes = sorted({r["mode"] for r in rows if r["dataset"] == "colondb"})
    assert modes == ["projector_off", "projector_on"]

    # Should infer experiment/seed when the directory structure matches.
    for r in rows:
        assert "dataset" in r and "mode" in r and "file" in r
        assert r.get("experiment") == "taskA_pranet_crisp"
        assert r.get("seed") == 0

