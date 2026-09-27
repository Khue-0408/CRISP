"""Synthetic fixtures for the evaluator's nested per-dataset JSON layout."""

import csv
import json
import sys
from pathlib import Path

import pytest

from crisp.scripts.export_tables import _collect_metric_files, main


METRICS = {
    "dice": 0.90,
    "iou": 0.82,
    "boundary_f1": 0.88,
    "hd95": 10.0,
    "bece": 0.04,
    "mDice": 0.90,
    "mIoU": 0.82,
    "B-F1": 0.88,
    "HD95": 10.0,
    "bECE": 0.04,
}


def _write_result(
    root: Path,
    *,
    experiment: str = "crisp_pranet_crisp",
    seed_dir: str = "seed_2026",
    dataset: str = "CVC-300",
    filename: str = "projector_on.json",
    metrics: object = METRICS,
) -> Path:
    path = root / experiment / seed_dir / dataset / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics), encoding="utf-8")
    return path


def test_actual_nested_evaluator_path_and_metrics_are_preserved(tmp_path: Path) -> None:
    path = _write_result(tmp_path)
    assert _collect_metric_files(tmp_path) == [{
        "source_file": str(path),
        "experiment": "crisp_pranet_crisp",
        "seed": 2026,
        "dataset": "CVC-300",
        "mode": "projector_on",
        "metrics": METRICS,
    }]


@pytest.mark.parametrize("dataset", ["CVC-ClinicDB", "ETIS-LaribPolypDB"])
def test_dataset_names_with_hyphens_are_not_tokenized(tmp_path: Path, dataset: str) -> None:
    _write_result(tmp_path, dataset=dataset)
    assert _collect_metric_files(tmp_path)[0]["dataset"] == dataset


def test_projector_modes_remain_distinct_and_summary_is_excluded(tmp_path: Path) -> None:
    _write_result(tmp_path, filename="projector_on.json")
    _write_result(tmp_path, filename="projector_off.json")
    summary = tmp_path / "crisp_pranet_crisp" / "seed_2026" / "summary.json"
    summary.write_text(json.dumps({"results": [{"dataset": "CVC-300"}]}), encoding="utf-8")
    records = _collect_metric_files(tmp_path)
    assert len(records) == 2
    assert {record["mode"] for record in records} == {"projector_on", "projector_off"}
    assert len({record["source_file"] for record in records}) == 2


def test_baseline_projector_off_only_remains_valid(tmp_path: Path) -> None:
    _write_result(tmp_path, experiment="crisp_pranet_baseline", filename="projector_off.json")
    records = _collect_metric_files(tmp_path)
    assert len(records) == 1
    assert records[0]["experiment"] == "crisp_pranet_baseline"
    assert records[0]["mode"] == "projector_off"


@pytest.mark.parametrize("seed_dir", ["seed_x", "2026", "seed_", "seed_2026_extra"])
def test_malformed_seed_directory_fails(tmp_path: Path, seed_dir: str) -> None:
    _write_result(tmp_path, seed_dir=seed_dir)
    with pytest.raises(ValueError, match="invalid seed directory"):
        _collect_metric_files(tmp_path)


@pytest.mark.parametrize("missing", ["experiment", "dataset"])
def test_missing_path_identity_fails(tmp_path: Path, missing: str) -> None:
    parts = (["seed_2026", "CVC-300"] if missing == "experiment"
             else ["crisp_pranet_crisp", "seed_2026"])
    path = tmp_path.joinpath(*parts) / "projector_on.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(METRICS), encoding="utf-8")
    with pytest.raises(ValueError, match="must be under"):
        _collect_metric_files(tmp_path)


@pytest.mark.parametrize("filename", ["other.json", ".json", "colondb_projector_on.json"])
def test_unsupported_or_legacy_flat_mode_fails_explicitly(tmp_path: Path, filename: str) -> None:
    _write_result(tmp_path, filename=filename)
    with pytest.raises(ValueError, match="unsupported mode.*Legacy flat filenames are unsupported"):
        _collect_metric_files(tmp_path)


def test_old_flat_file_layout_is_not_accepted_as_nested(tmp_path: Path) -> None:
    _write_result(tmp_path, dataset="eval", filename="colondb_projector_on.json")
    with pytest.raises(ValueError, match="Legacy flat filenames are unsupported"):
        _collect_metric_files(tmp_path)


@pytest.mark.parametrize("metrics", [[1, 2], {"dice": "bad"}, {"dice": True}, {"dice": float("nan")}])
def test_non_mapping_or_invalid_metric_fails(tmp_path: Path, metrics: object) -> None:
    _write_result(tmp_path, metrics=metrics)
    with pytest.raises(ValueError, match="metric mapping|non-numeric or non-finite"):
        _collect_metric_files(tmp_path)


def test_malformed_json_fails_instead_of_disappearing(tmp_path: Path) -> None:
    path = _write_result(tmp_path)
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        _collect_metric_files(tmp_path)


def test_exporter_emits_per_artifact_rows_without_statistics(tmp_path: Path, monkeypatch) -> None:
    input_root = tmp_path / "metrics"
    output_root = tmp_path / "parsed"
    first = _write_result(input_root)
    second = _write_result(input_root, seed_dir="seed_2027", filename="projector_off.json")
    monkeypatch.setattr(sys, "argv", ["export_tables", "--input-dir", str(input_root), "--output-dir", str(output_root)])
    main()
    with (output_root / "parsed_results.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert {row["source_file"] for row in rows} == {str(first), str(second)}
    assert {row["mode"] for row in rows} == {"projector_on", "projector_off"}
    assert {row["dataset"] for row in rows} == {"CVC-300"}
    assert all(key not in rows[0] for key in ("mean", "std", "ddof", "paired_delta", "ci_low", "ci_high"))
    assert (output_root / "parsed_results.md").exists()
    assert not (output_root / "aggregated_results.csv").exists()
