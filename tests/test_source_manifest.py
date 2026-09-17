"""Synthetic source manifest gates; no reported-run membership is embedded here."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from types import ModuleType

import pytest
from torch.utils.data import DataLoader, RandomSampler

from crisp.data import datasets as dataset_module
from crisp.data.datasets import build_local_train_val_dataset
from crisp.data.source_manifest import (
    CURRENT_SOURCE_COUNT_PROFILE,
    read_source_manifest,
    validate_current_source_count_profile,
    validate_source_membership,
)
from crisp.registry import build_dataset


@pytest.fixture(autouse=True)
def _identity_transforms_for_membership_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise registry routing without requiring image-transform dependencies."""
    module = ModuleType("crisp.data.transforms")
    module.build_train_transforms = lambda config: None
    module.build_eval_transforms = lambda config: None
    monkeypatch.setitem(sys.modules, "crisp.data.transforms", module)


def _write_ids(path: Path, ids: list[str]) -> None:
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> dict:
    sources = {}
    for name, stems in (("Kvasir-SEG", ("a", "b")), ("CVC-ClinicDB", ("1", "2"))):
        root = tmp_path / name
        for folder in ("images", "masks"):
            (root / folder).mkdir(parents=True)
            for stem in stems:
                (root / folder / f"{stem}.png").write_bytes(b"fixture")
        sources[name] = {"root": str(root), "image_dir": "images", "mask_dir": "masks"}
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    _write_ids(train, ["Kvasir-SEG/b", "CVC-ClinicDB/2"])
    _write_ids(val, ["CVC-ClinicDB/1", "Kvasir-SEG/a"])
    return {
        "seed": 2026,
        "source_data": {
            "name": "synthetic_source",
            "image_size": 16,
            "source_split": {
                "mode": "manifest",
                "train_manifest": str(train),
                "val_manifest": str(val),
                "datasets": sources,
            },
        },
    }


def _ids(config: dict, split: str) -> list[str]:
    return [f"{record.dataset_name}/{record.image_id}" for record in build_dataset(config, split).samples]


def test_synthetic_manifest_preserves_membership_and_order_without_cache(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    assert _ids(config, "train") == ["Kvasir-SEG/b", "CVC-ClinicDB/2"]
    assert _ids(config, "val") == ["CVC-ClinicDB/1", "Kvasir-SEG/a"]
    assert not (tmp_path / "metadata").exists()
    dataset = build_dataset(config, "train")
    assert dataset.source_split_provenance["counts"] == {
        "pool": {"Kvasir-SEG": 2, "CVC-ClinicDB": 2},
        "train": {"Kvasir-SEG": 1, "CVC-ClinicDB": 1},
        "val": {"CVC-ClinicDB": 1, "Kvasir-SEG": 1},
    }


def test_manifest_membership_is_seed_invariant_but_loader_may_shuffle(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    original = (_ids(config, "train"), _ids(config, "val"))
    for seed in (2026, 2027, 2028, 2029, 2030):
        config["seed"] = seed
        assert (_ids(config, "train"), _ids(config, "val")) == original
    loader = DataLoader(build_dataset(config, "train"), batch_size=1, shuffle=True)
    assert isinstance(loader.sampler, RandomSampler)
    assert _ids(config, "train") == original[0]


def test_filesystem_discovery_order_does_not_change_manifest_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture(tmp_path)
    expected = (_ids(config, "train"), _ids(config, "val"))
    original_listing = dataset_module.list_supported_files
    monkeypatch.setattr(dataset_module, "list_supported_files", lambda path: list(reversed(original_listing(path))))
    assert (_ids(config, "train"), _ids(config, "val")) == expected


@pytest.mark.parametrize("key", ["train_manifest", "val_manifest"])
def test_missing_manifest_fails_without_fraction_fallback(tmp_path: Path, key: str) -> None:
    config = _fixture(tmp_path)
    config["source_data"]["source_split"][key] = str(tmp_path / "absent.txt")
    config["source_data"]["local_split"] = {"val_fraction": 0.5, "cache_dir": str(tmp_path / "cache")}
    with pytest.raises(FileNotFoundError, match="Source split manifest not found"):
        build_dataset(config, "train")
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("key", ["train_manifest", "val_manifest"])
def test_duplicate_within_manifest_fails(tmp_path: Path, key: str) -> None:
    config = _fixture(tmp_path)
    path = Path(config["source_data"]["source_split"][key])
    _write_ids(path, ["Kvasir-SEG/a", "Kvasir-SEG/a"])
    with pytest.raises(ValueError, match="Duplicate source sample ID"):
        build_dataset(config, "train")


def test_train_validation_overlap_fails(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    _write_ids(Path(config["source_data"]["source_split"]["val_manifest"]),
               ["Kvasir-SEG/b", "CVC-ClinicDB/1"])
    with pytest.raises(ValueError, match="Train/validation overlap"):
        build_dataset(config, "train")


def test_unknown_id_fails(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    _write_ids(Path(config["source_data"]["source_split"]["train_manifest"]),
               ["Kvasir-SEG/unknown", "CVC-ClinicDB/2"])
    with pytest.raises(ValueError, match="Unknown source sample ID"):
        build_dataset(config, "train")


def test_unsupported_dataset_prefix_fails(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    _write_ids(Path(config["source_data"]["source_split"]["train_manifest"]),
               ["Other/b", "CVC-ClinicDB/2"])
    with pytest.raises(ValueError, match="Unsupported source dataset"):
        build_dataset(config, "train")


def test_unassigned_source_sample_fails(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    _write_ids(Path(config["source_data"]["source_split"]["train_manifest"]),
               ["Kvasir-SEG/b"])
    with pytest.raises(ValueError, match="Unassigned source sample ID"):
        build_dataset(config, "train")


@pytest.mark.parametrize("missing_folder", ["images", "masks"])
def test_incomplete_image_mask_pair_fails(tmp_path: Path, missing_folder: str) -> None:
    config = _fixture(tmp_path)
    (tmp_path / "Kvasir-SEG" / missing_folder / "a.png").unlink()
    with pytest.raises(ValueError, match="Source image/mask pairing mismatch"):
        build_dataset(config, "val")


def test_duplicate_filename_stem_fails(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    (tmp_path / "Kvasir-SEG" / "images" / "a.jpg").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="Duplicate source file stem"):
        build_dataset(config, "train")


@pytest.mark.parametrize("bad_id", ["Kvasir-SEG", "Kvasir-SEG/a/b", "Kvasir-SEG/a b", "kvasir-SEG/", "/a"])
def test_malformed_id_fails(tmp_path: Path, bad_id: str) -> None:
    config = _fixture(tmp_path)
    _write_ids(Path(config["source_data"]["source_split"]["train_manifest"]), [bad_id])
    with pytest.raises(ValueError, match="Malformed source sample ID"):
        build_dataset(config, "train")


def test_normalized_manifest_hash_is_path_independent(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("  Kvasir-SEG/a  \n\nCVC-ClinicDB/1\n", encoding="utf-8")
    second.write_text("Kvasir-SEG/a\nCVC-ClinicDB/1\n", encoding="utf-8")
    left, right = read_source_manifest(first), read_source_manifest(second)
    assert left.ids == right.ids
    assert left.sha256 == right.sha256
    assert left.path != right.path
    assert left.count == 2


def test_manifest_hash_and_paths_are_exposed_on_dataset(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    provenance = build_dataset(config, "train").source_split_provenance
    split_cfg = config["source_data"]["source_split"]
    assert provenance["train_manifest"] == split_cfg["train_manifest"]
    assert provenance["val_manifest"] == split_cfg["val_manifest"]
    assert provenance["train_sha256"] == read_source_manifest(split_cfg["train_manifest"]).sha256
    assert provenance["val_sha256"] == read_source_manifest(split_cfg["val_manifest"]).sha256


def test_empty_manifest_and_unknown_mode_fail(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    Path(config["source_data"]["source_split"]["val_manifest"]).write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Source split manifest is empty"):
        build_dataset(config, "train")
    config["source_data"]["source_split"]["mode"] = "unknown"
    with pytest.raises(ValueError, match="Unknown source split mode"):
        build_dataset(config, "train")


def test_current_count_profile_is_validation_only(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    config["source_data"]["source_split"]["count_profile"] = "current_crisp"
    with pytest.raises(ValueError, match="Wrong source pool counts"):
        build_dataset(config, "train")
    assert not (tmp_path / "metadata").exists()


def test_legacy_fraction_builder_remains_separate_and_cached(tmp_path: Path) -> None:
    image_dir = tmp_path / "TrainDataset" / "image"
    mask_dir = tmp_path / "TrainDataset" / "mask"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for index in range(10):
        (image_dir / f"{index}.png").write_bytes(b"fixture")
        (mask_dir / f"{index}.png").write_bytes(b"fixture")
    data_cfg = {
        "name": "legacy_fixture", "root": str(tmp_path),
        "train_dir": "TrainDataset", "image_dir": "image", "mask_dir": "mask",
        "local_split": {"val_fraction": 0.2, "cache_dir": str(tmp_path / "cache")},
    }
    train = build_local_train_val_dataset(data_cfg, "train", transforms=None, seed=7)
    val = build_local_train_val_dataset(data_cfg, "val", transforms=None, seed=7)
    assert len(train) == 8 and len(val) == 2
    assert train.source_split_provenance is None
    assert (tmp_path / "cache" / "legacy_fixture_seed_7_val_200" / "train.txt").exists()
    assert (tmp_path / "cache" / "legacy_fixture_seed_7_val_200" / "val.txt").exists()


def test_current_source_count_profile_uses_synthetic_ids_only() -> None:
    kvasir = [f"Kvasir-SEG/k{i:03d}" for i in range(900)]
    clinic = [f"CVC-ClinicDB/c{i:03d}" for i in range(550)]
    train = kvasir[:810] + clinic[:495]
    val = kvasir[810:] + clinic[495:]
    counts = validate_current_source_count_profile(kvasir + clinic, train, val)
    assert counts == CURRENT_SOURCE_COUNT_PROFILE
    assert len(train) == 1305 and len(val) == 145 and len(kvasir + clinic) == 1450


@pytest.mark.parametrize("wrong_group", ["pool", "train", "val"])
def test_wrong_count_in_each_profile_category_fails(wrong_group: str) -> None:
    kvasir = [f"Kvasir-SEG/k{i:03d}" for i in range(900)]
    clinic = [f"CVC-ClinicDB/c{i:03d}" for i in range(550)]
    expected = deepcopy(CURRENT_SOURCE_COUNT_PROFILE)
    expected[wrong_group]["Kvasir-SEG"] += 1
    with pytest.raises(ValueError, match=f"Wrong source {wrong_group} counts"):
        validate_source_membership(
            kvasir + clinic, kvasir[:810] + clinic[:495], kvasir[810:] + clinic[495:],
            supported_datasets=expected["pool"], expected_counts=expected,
        )
