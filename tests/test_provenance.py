"""Synthetic identity gates for CRISP run, checkpoint, and evaluation artifacts."""

import csv
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from crisp.engine.trainer import Trainer
from crisp.scripts.export_tables import _collect_metric_files, main as export_main
from crisp.utils.provenance import (
    SCHEMA_VERSION,
    checkpoint_provenance,
    content_sha256,
    evaluation_provenance,
    file_sha256,
    git_state,
    run_provenance,
    source_split_provenance,
    teacher_provenance,
    write_provenance,
)


def _config(tmp_path: Path) -> dict:
    return {
        "experiment_name": "experiment_A",
        "seed": 2026,
        "model": {"name": "tiny"},
        "method": {"name": "crisp"},
        "source_data": {"mode": "local_train_test", "local_split": {"val_fraction": 0.1}},
        "eval": {"ece": {"bins": 15}},
        "output_dir": str(tmp_path / "checkpoints"),
    }


def _git() -> dict:
    return {"sha": "a" * 40, "branch": "main", "dirty": False}


def test_resolved_config_hash_ignores_insertion_order_but_not_values() -> None:
    left = {"seed": 2026, "nested": {"beta": 0.35, "tau": 1}}
    right = {"nested": {"tau": 1, "beta": 0.35}, "seed": 2026}
    assert content_sha256(left) == content_sha256(right)
    right["nested"]["beta"] = 0.0
    assert content_sha256(left) != content_sha256(right)


def test_file_hash_uses_bytes_not_path(tmp_path: Path) -> None:
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    first.write_bytes(b"checkpoint\x00bytes")
    second.write_bytes(first.read_bytes())
    assert file_sha256(first) == file_sha256(second)
    second.write_bytes(b"checkpoint\x01bytes")
    assert file_sha256(first) != file_sha256(second)


def test_manifest_provenance_comes_from_validated_dataset(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["source_data"] = {"source_split": {"mode": "manifest"}}
    actual = {
        "mode": "manifest", "train_manifest": "train.txt", "val_manifest": "val.txt",
        "train_sha256": "a" * 64, "val_sha256": "b" * 64,
        "counts": {"train": {"source": 9}, "val": {"source": 1}, "pool": {"source": 10}},
    }
    dataset = type("Dataset", (), {"source_split_provenance": actual})()
    record = run_provenance(config, train_dataset=dataset, git=_git())
    assert record["source_split"] == actual
    assert record["source_split"]["counts"]["pool"]["source"] == 10
    with pytest.raises(ValueError, match="validated dataset"):
        source_split_provenance(config)


def test_fraction_mode_is_honest(tmp_path: Path) -> None:
    config = _config(tmp_path)
    record = run_provenance(config, git=_git())
    assert record["source_split"] == {
        "mode": "fraction", "seed": 2026, "val_fraction": 0.1,
        "manifest_paths": None, "manifest_hashes": None, "counts": None,
    }


def test_teacher_hash_only_when_file_exists(tmp_path: Path) -> None:
    config = _config(tmp_path)
    present = tmp_path / "teacher.pt"
    present.write_bytes(b"teacher checkpoint")
    config["teacher_pool"] = {"teachers": [
        {"name": "present", "model": "tiny", "checkpoint": str(present)},
        {"name": "missing", "model": "tiny", "checkpoint": str(tmp_path / "absent.pt")},
    ]}
    records = teacher_provenance(config)
    assert records[0]["artifact_status"] == "available"
    assert records[0]["checkpoint_sha256"] == file_sha256(present)
    assert records[1]["artifact_status"] == "missing"
    assert records[1]["checkpoint_sha256"] is None
    assert all(record["training_exposure"] is None for record in records)


def test_checkpoint_links_to_exact_run_fields(tmp_path: Path) -> None:
    run = run_provenance(_config(tmp_path), git=_git(), run_id="run-test")
    config = _config(tmp_path)
    trainer = Trainer(nn.Linear(1, 1), None, None, config, run_record=run)
    trainer._optimizer = trainer.build_optimizer()
    trainer._scheduler = None
    payload = trainer._checkpoint_state(epoch=7, train_metrics={})
    checkpoint = payload["provenance"]
    assert checkpoint["schema_version"] == SCHEMA_VERSION
    assert (checkpoint["run_id"], checkpoint["config_sha256"], checkpoint["git_sha"], checkpoint["seed"]) == (
        "run-test", run["config_sha256"], "a" * 40, 2026,
    )
    assert checkpoint["epoch"] == 7
    assert payload["seed"] == checkpoint["seed"]
    assert payload["config"] == run["resolved_config"]


def _evaluation_fixture(tmp_path: Path):
    config = _config(tmp_path)
    run = run_provenance(config, git=_git(), run_id="run-test")
    checkpoint = {"epoch": 7, "config": config, "provenance": checkpoint_provenance(run, 7, {})}
    checkpoint_path = tmp_path / "best.pt"
    torch.save(checkpoint, checkpoint_path)
    metric_path = tmp_path / "experiment_A" / "seed_2026" / "CVC-300" / "projector_on.json"
    metric_path.parent.mkdir(parents=True)
    metric_path.write_text(json.dumps({"dice": 0.8}), encoding="utf-8")
    record = evaluation_provenance(config, checkpoint_path, checkpoint, "CVC-300", "projector_on", metric_path, git=_git())
    return config, checkpoint, checkpoint_path, metric_path, record


def test_evaluation_links_actual_checkpoint_bytes_and_run(tmp_path: Path) -> None:
    config, checkpoint, checkpoint_path, metric_path, record = _evaluation_fixture(tmp_path)
    assert record["checkpoint_sha256"] == file_sha256(checkpoint_path)
    assert record["run_id"] == checkpoint["provenance"]["run_id"]
    assert record["config_sha256"] == content_sha256(config)
    assert record["evaluation_config_sha256"] == content_sha256(config)
    assert record["source_file"] == str(metric_path)
    assert record["metric_config"] == config["eval"]
    with pytest.raises(ValueError, match="Checkpoint seed conflicts"):
        evaluation_provenance({**config, "seed": 2027}, checkpoint_path, checkpoint, "CVC-300", "projector_on", metric_path, git=_git())
    with pytest.raises(ValueError, match="Checkpoint epoch conflicts"):
        evaluation_provenance(config, checkpoint_path, {**checkpoint, "epoch": 8}, "CVC-300", "projector_on", metric_path, git=_git())
    checkpoint["provenance"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="byte SHA-256 conflicts"):
        evaluation_provenance(config, checkpoint_path, checkpoint, "CVC-300", "projector_on", metric_path, git=_git())


@pytest.mark.parametrize("key,value", [
    ("experiment", "experiment_B"), ("seed", 2027),
    ("dataset", "ETIS"), ("mode", "projector_off"),
])
def test_exporter_rejects_path_provenance_conflict(tmp_path: Path, key: str, value: object) -> None:
    _, _, _, metric_path, record = _evaluation_fixture(tmp_path)
    record[key] = value
    write_provenance(metric_path.with_name("projector_on.provenance.json"), record)
    with pytest.raises(ValueError, match=f"path {key} conflicts"):
        _collect_metric_files(tmp_path)


def test_exporter_retains_traceability_and_checks_checkpoint_bytes(tmp_path: Path) -> None:
    _, _, checkpoint_path, metric_path, record = _evaluation_fixture(tmp_path)
    sidecar = metric_path.with_name("projector_on.provenance.json")
    write_provenance(sidecar, record)
    parsed = _collect_metric_files(tmp_path)[0]
    assert parsed["source_file"] == str(metric_path)
    assert parsed["provenance_file"] == str(sidecar)
    assert parsed["run_id"] == "run-test"
    assert parsed["checkpoint_sha256"] == file_sha256(checkpoint_path)
    assert parsed["config_sha256"] == record["config_sha256"]
    assert parsed["git_sha"] == record["git_sha"]
    checkpoint_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checkpoint SHA-256 conflicts"):
        _collect_metric_files(tmp_path)


def test_normalized_csv_keeps_run_checkpoint_and_source_links(tmp_path: Path, monkeypatch) -> None:
    _, _, checkpoint_path, metric_path, record = _evaluation_fixture(tmp_path)
    sidecar = metric_path.with_name("projector_on.provenance.json")
    write_provenance(sidecar, record)
    output_dir = tmp_path / "parsed"
    monkeypatch.setattr(sys, "argv", ["export_tables", "--input-dir", str(tmp_path), "--output-dir", str(output_dir)])
    export_main()
    with (output_dir / "parsed_results.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["source_file"] == str(metric_path)
    assert rows[0]["provenance_file"] == str(sidecar)
    assert rows[0]["run_id"] == "run-test"
    assert rows[0]["checkpoint_sha256"] == file_sha256(checkpoint_path)
    assert rows[0]["config_sha256"] == record["config_sha256"]
    assert rows[0]["git_sha"] == "a" * 40


def test_same_run_id_cannot_have_conflicting_config_hashes(tmp_path: Path) -> None:
    _, _, _, first, record = _evaluation_fixture(tmp_path)
    write_provenance(first.with_name("projector_on.provenance.json"), record)
    second = first.with_name("projector_off.json")
    second.write_text(json.dumps({"dice": 0.7}), encoding="utf-8")
    second_record = {**record, "mode": "projector_off", "source_file": str(second), "config_sha256": "f" * 64}
    second_record["evaluation_id"] = content_sha256({
        "run_id": second_record["run_id"],
        "checkpoint_sha256": second_record["checkpoint_sha256"],
        "evaluation_config_sha256": second_record["evaluation_config_sha256"],
        "dataset": second_record["dataset"],
        "mode": second_record["mode"],
    })
    write_provenance(second.with_name("projector_off.provenance.json"), second_record)
    with pytest.raises(ValueError, match="Conflicting experiment, seed, or config hashes"):
        _collect_metric_files(tmp_path)


def test_orphan_sidecar_fails_and_provenance_cannot_be_overwritten(tmp_path: Path) -> None:
    _, _, _, metric_path, record = _evaluation_fixture(tmp_path)
    sidecar = metric_path.with_name("projector_on.provenance.json")
    write_provenance(sidecar, record)
    with pytest.raises(FileExistsError):
        write_provenance(sidecar, record)
    metric_path.unlink()
    with pytest.raises(ValueError, match="Orphan evaluator provenance"):
        _collect_metric_files(tmp_path)


def test_git_state_parser_can_be_injected_without_mutating_repo(tmp_path: Path, monkeypatch) -> None:
    import crisp.utils.provenance as provenance

    def fake_check_output(args, **kwargs):
        return {"rev-parse": "a" * 40 + "\n", "branch": "main\n", "status": " M file.py\n"}[args[1]]

    monkeypatch.setattr(provenance.subprocess, "check_output", fake_check_output)
    assert git_state(tmp_path) == {"sha": "a" * 40, "branch": "main", "dirty": True}
