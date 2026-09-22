"""Synthetic gates for exact CRISP evaluation sample membership."""

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from crisp.data.evaluation_membership import (
    CURRENT_EVALUATION_COUNTS,
    evaluation_membership_record,
    read_evaluation_manifest,
    validate_current_evaluation_count,
    validate_evaluation_membership,
)
from crisp.data.datasets import discover_local_test_datasets
from crisp.registry import build_dataset
from crisp.scripts.export_tables import _collect_metric_files
from crisp.utils.provenance import (
    checkpoint_provenance, evaluation_provenance, run_provenance, write_provenance,
)


@pytest.fixture(autouse=True)
def _identity_transforms(monkeypatch) -> None:
    module = ModuleType("crisp.data.transforms")
    module.build_train_transforms = lambda config: None
    module.build_eval_transforms = lambda config: None
    monkeypatch.setitem(sys.modules, "crisp.data.transforms", module)


def _files(root: Path, image_ids: tuple[str, ...], mask_ids: tuple[str, ...]) -> None:
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    for image_id in image_ids:
        (images / f"{image_id}.png").write_bytes(b"synthetic image")
    for image_id in mask_ids:
        (masks / f"{image_id}.png").write_bytes(b"synthetic mask")


def _config(root: Path, manifest: Path | None = None) -> dict:
    data = {
        "name": "CVC-300", "root": str(root), "image_dir": "images", "mask_dir": "masks",
        "evaluation_dataset_name": "CVC-300",
    }
    if manifest is not None:
        data["evaluation_manifest"] = str(manifest)
    return {"source_data": data}


def test_membership_sha_ignores_discovery_order_but_not_set_changes() -> None:
    first = evaluation_membership_record(
        "CVC-300", ["CVC-300/b", "CVC-300/a"], mode="discovered_full_dataset"
    )
    reordered = evaluation_membership_record(
        "CVC-300", ["CVC-300/a", "CVC-300/b"], mode="discovered_full_dataset"
    )
    changed = evaluation_membership_record(
        "CVC-300", ["CVC-300/a", "CVC-300/c"], mode="discovered_full_dataset"
    )
    assert first["sample_ids"] == ["CVC-300/a", "CVC-300/b"]
    assert first["membership_sha256"] == reordered["membership_sha256"]
    assert first["membership_sha256"] != changed["membership_sha256"]
    assert first["manifest_path"] is None and first["manifest_sha256"] is None


def test_duplicate_and_wrong_dataset_fail() -> None:
    with pytest.raises(ValueError, match="Duplicate evaluation sample"):
        evaluation_membership_record("CVC-300", ["CVC-300/a", "CVC-300/a"], mode="discovered_full_dataset")
    with pytest.raises(ValueError, match="wrong dataset"):
        evaluation_membership_record("CVC-300", ["ETIS/a"], mode="discovered_full_dataset")


def test_duplicate_filename_stem_fails_before_membership_hash(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _files(root, ("a",), ("a",))
    (root / "images" / "a.jpg").write_bytes(b"second image with same stem")
    with pytest.raises(ValueError, match="Duplicate source file stem"):
        build_dataset(_config(root), split="test")


@pytest.mark.parametrize("dataset,count", list(CURRENT_EVALUATION_COUNTS.items()))
def test_current_count_profile_validates_without_selecting_ids(dataset: str, count: int) -> None:
    ids = [f"{dataset}/synthetic_{index:03d}" for index in range(count)]
    record = evaluation_membership_record(dataset, ids, mode="discovered_full_dataset")
    validate_current_evaluation_count(record)
    wrong = evaluation_membership_record(dataset, ids[:-1], mode="discovered_full_dataset")
    with pytest.raises(ValueError, match="Wrong evaluation count"):
        validate_current_evaluation_count(wrong)


def test_explicit_manifest_selects_exact_pairs_independent_of_file_order(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _files(root, ("c", "a", "b"), ("b", "c", "a"))
    manifest = tmp_path / "selection.txt"
    manifest.write_text("CVC-300/c\nCVC-300/a\n", encoding="utf-8")
    dataset = build_dataset(_config(root, manifest), split="test")
    record = dataset.evaluation_membership_provenance
    assert [sample.image_id for sample in dataset.samples] == ["a", "c"]
    assert record["mode"] == "explicit_manifest"
    assert record["sample_ids"] == ["CVC-300/a", "CVC-300/c"]
    assert record["manifest_path"] == str(manifest.resolve())
    assert record["manifest_sha256"] == read_evaluation_manifest(manifest, "CVC-300")[2]
    manifest.write_text("CVC-300/a\nCVC-300/c\n", encoding="utf-8")
    reordered = build_dataset(_config(root, manifest), split="test")
    assert reordered.evaluation_membership_provenance["membership_sha256"] == record["membership_sha256"]


@pytest.mark.parametrize("contents,error", [
    ("CVC-300/a\nCVC-300/a\n", "Duplicate evaluation sample"),
    ("CVC-300/unknown\n", "Unknown evaluation sample ID"),
    ("ETIS/a\n", "wrong dataset"),
])
def test_explicit_manifest_rejects_duplicate_unknown_or_wrong_dataset(
    tmp_path: Path, contents: str, error: str,
) -> None:
    root = tmp_path / "dataset"
    _files(root, ("a",), ("a",))
    manifest = tmp_path / "selection.txt"
    manifest.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        build_dataset(_config(root, manifest), split="test")


@pytest.mark.parametrize("images,masks", [(("a",), ()), ((), ("a",))])
def test_manifest_referenced_missing_image_or_mask_fails(
    tmp_path: Path, images: tuple[str, ...], masks: tuple[str, ...],
) -> None:
    root = tmp_path / "dataset"
    _files(root, images, masks)
    manifest = tmp_path / "selection.txt"
    manifest.write_text("CVC-300/a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pairing mismatch"):
        build_dataset(_config(root, manifest), split="test")


def test_discovery_is_not_misreported_as_explicit_manifest(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _files(root, ("b", "a"), ("a", "b"))
    dataset = build_dataset(_config(root), split="test")
    record = dataset.evaluation_membership_provenance
    assert record["mode"] == "discovered_full_dataset"
    assert record["sample_ids"] == ["CVC-300/a", "CVC-300/b"]


def test_full_discovery_fails_on_unpaired_files(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _files(root, ("a", "b"), ("a",))
    with pytest.raises(ValueError, match="pairing mismatch"):
        build_dataset(_config(root), split="test")


def test_local_test_discovery_does_not_hide_malformed_dataset(tmp_path: Path) -> None:
    root = tmp_path / "TestDataset" / "CVC-300"
    _files(root, ("a", "b"), ("a",))
    with pytest.raises(ValueError, match="pairing mismatch"):
        discover_local_test_datasets({"root": str(tmp_path), "test_dir": "TestDataset"})


def test_opt_in_count_profile_is_consumed_by_dataset_builder(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _files(root, ("a", "b"), ("a", "b"))
    config = _config(root)
    config["source_data"]["evaluation_count_profile"] = "current_crisp"
    with pytest.raises(ValueError, match="Wrong evaluation count for CVC-300"):
        build_dataset(config, split="test")


def test_existing_split_file_is_labeled_legacy_not_full_discovery(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _files(root, ("a", "b"), ("a", "b"))
    split_file = tmp_path / "legacy.txt"
    split_file.write_text("b\n", encoding="utf-8")
    config = _config(root)
    config["source_data"]["splits"] = {"test": {"split_file": str(split_file)}}
    dataset = build_dataset(config, split="test")
    assert dataset.evaluation_membership_provenance["mode"] == "legacy_split_file"
    assert dataset.evaluation_membership_provenance["sample_ids"] == ["CVC-300/b"]


@pytest.mark.parametrize("contents,error", [
    ("a\na\n", "Duplicate evaluation ID"),
    ("unknown\n", "references ids not present"),
])
def test_legacy_test_split_file_does_not_silently_filter_invalid_ids(
    tmp_path: Path, contents: str, error: str,
) -> None:
    root = tmp_path / "dataset"
    _files(root, ("a",), ("a",))
    split_file = tmp_path / "legacy.txt"
    split_file.write_text(contents, encoding="utf-8")
    config = _config(root)
    config["source_data"]["splits"] = {"test": {"split_file": str(split_file)}}
    with pytest.raises(ValueError, match=error):
        build_dataset(config, split="test")


def _evaluation_artifacts(tmp_path: Path, membership_ids: list[str], mode: str = "projector_on"):
    config = {"experiment_name": "experiment_A", "seed": 2026, "eval": {"ece": {"bins": 15}}}
    run = run_provenance(config, git={"sha": "a" * 40, "branch": "main", "dirty": False}, run_id="run-A")
    checkpoint = {"config": config, "epoch": 4, "provenance": checkpoint_provenance(run, 4, {})}
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    dataset_dir = tmp_path / "experiment_A" / "seed_2026" / "CVC-300"
    dataset_dir.mkdir(parents=True)
    membership = evaluation_membership_record("CVC-300", membership_ids, mode="discovered_full_dataset")
    membership_path = dataset_dir / "dataset.membership.json"
    write_provenance(membership_path, membership)
    metric_path = dataset_dir / f"{mode}.json"
    metric_path.write_text(json.dumps({"dice": 0.8}), encoding="utf-8")
    sidecar = evaluation_provenance(
        config, checkpoint_path, checkpoint, "CVC-300", mode, metric_path,
        git={"sha": "a" * 40, "branch": "main", "dirty": False},
        membership=membership, membership_file=membership_path,
    )
    write_provenance(dataset_dir / f"{mode}.provenance.json", sidecar)
    return config, checkpoint, checkpoint_path, metric_path, membership_path, membership, sidecar


def test_projector_modes_share_membership_but_not_evaluation_id(tmp_path: Path) -> None:
    config, checkpoint, checkpoint_path, metric_path, membership_path, membership, on = _evaluation_artifacts(
        tmp_path, ["CVC-300/a", "CVC-300/b"]
    )
    off_path = metric_path.with_name("projector_off.json")
    off = evaluation_provenance(
        config, checkpoint_path, checkpoint, "CVC-300", "projector_off", off_path,
        git={"sha": "a" * 40, "branch": "main", "dirty": False},
        membership=membership, membership_file=membership_path,
    )
    assert on["evaluation_membership_sha256"] == off["evaluation_membership_sha256"]
    assert on["membership_file"] == off["membership_file"]
    assert on["evaluation_id"] != off["evaluation_id"]


def test_same_dataset_name_different_membership_changes_evaluation_id(tmp_path: Path) -> None:
    first = _evaluation_artifacts(tmp_path / "first", ["CVC-300/a", "CVC-300/b"])
    second = _evaluation_artifacts(tmp_path / "second", ["CVC-300/a", "CVC-300/c"])
    assert first[6]["checkpoint_sha256"] == second[6]["checkpoint_sha256"]
    assert first[6]["evaluation_config_sha256"] == second[6]["evaluation_config_sha256"]
    assert first[6]["evaluation_membership_sha256"] != second[6]["evaluation_membership_sha256"]
    assert first[6]["evaluation_id"] != second[6]["evaluation_id"]


def test_exporter_retains_membership_links_and_detects_file_tampering(tmp_path: Path) -> None:
    _, _, _, metric_path, membership_path, membership, sidecar = _evaluation_artifacts(
        tmp_path, ["CVC-300/a", "CVC-300/b"]
    )
    record = _collect_metric_files(tmp_path)[0]
    assert record["source_file"] == str(metric_path)
    assert record["run_id"] == "run-A"
    assert record["membership_status"] == "verified"
    assert record["evaluation_membership_sha256"] == membership["membership_sha256"]
    assert record["evaluation_sample_count"] == 2
    assert record["membership_file"] == str(membership_path)
    membership_path.write_text(membership_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="Membership file bytes conflict"):
        _collect_metric_files(tmp_path)


@pytest.mark.parametrize("change,error", [
    ("missing", "membership file is missing"),
    ("dataset", "wrong dataset"),
    ("count", "membership sample_count conflicts"),
    ("sha", "membership_sha256 conflicts"),
])
def test_exporter_rejects_membership_conflicts(tmp_path: Path, change: str, error: str) -> None:
    _, _, _, _, membership_path, _, _ = _evaluation_artifacts(tmp_path, ["CVC-300/a", "CVC-300/b"])
    if change == "missing":
        membership_path.unlink()
    else:
        data = json.loads(membership_path.read_text(encoding="utf-8"))
        if change == "dataset":
            data["dataset"] = "ETIS"
        elif change == "count":
            data["sample_count"] = 3
        else:
            data["membership_sha256"] = "0" * 64
        membership_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        _collect_metric_files(tmp_path)


@pytest.mark.parametrize("field,value,error", [
    ("evaluation_membership_sha256", "0" * 64, "Membership SHA-256 conflicts"),
    ("evaluation_sample_count", 3, "Membership count conflicts"),
])
def test_exporter_rejects_sidecar_membership_mismatch(
    tmp_path: Path, field: str, value: object, error: str,
) -> None:
    _, _, _, metric_path, _, _, _ = _evaluation_artifacts(tmp_path, ["CVC-300/a", "CVC-300/b"])
    sidecar_path = metric_path.with_name("projector_on.provenance.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar[field] = value
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        _collect_metric_files(tmp_path)
