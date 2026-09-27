"""Fail-loud current-protocol gates without historical membership fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from crisp.protocol import (
    CURRENT_EVALUATION_SUITE,
    validate_current_evaluation_membership_mode,
    validate_current_evaluation_protocol,
    validate_current_training_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
CURRENT_CONFIGS = {
    "thesis_unet_baseline.yaml",
    "thesis_unet_crisp.yaml",
    "thesis_unet_matched_beta0.yaml",
    "thesis_unet_margin_label_smoothing.yaml",
    "thesis_unet_boundary_weighted_logit_consistency.yaml",
    "thesis_unetpp_baseline.yaml",
    "thesis_unetpp_crisp.yaml",
    "thesis_unetpp_matched_beta0.yaml",
    "thesis_pranet_baseline.yaml",
    "thesis_pranet_crisp.yaml",
    "thesis_pranet_matched_beta0.yaml",
    "thesis_pranet_margin_label_smoothing.yaml",
    "thesis_pranet_boundary_weighted_logit_consistency.yaml",
    "thesis_pranet_teacher_robustness_equal.yaml",
    "thesis_pranet_teacher_robustness_weighted.yaml",
}
SOURCE_CONTRACT = """source_data:
  local_split: null
  source_split:
    mode: manifest
    count_profile: current_crisp
    train_manifest: null
    val_manifest: null
"""


def _manifest(path: Path) -> str:
    path.write_text("synthetic-fixture\n", encoding="utf-8")
    return str(path)


def _valid_source_config(tmp_path: Path) -> dict:
    return {
        "protocol_profile": "current_crisp",
        "source_data": {
            "local_split": None,
            "source_split": {
                "mode": "manifest",
                "count_profile": "current_crisp",
                "train_manifest": _manifest(tmp_path / "train.txt"),
                "val_manifest": _manifest(tmp_path / "val.txt"),
            }
        },
    }


def _valid_evaluation_config(tmp_path: Path) -> dict:
    return {
        "protocol_profile": "current_crisp",
        "eval_datasets": sorted(CURRENT_EVALUATION_SUITE),
        "eval": {
            "auto_discover_local_test_datasets": False,
            "skip_missing_datasets": False,
            "membership_count_profile": "current_crisp",
            "membership_manifests": {
                name: _manifest(tmp_path / f"{name}.txt")
                for name in CURRENT_EVALUATION_SUITE
            },
        },
    }


def test_current_source_fraction_fails_loudly() -> None:
    config = {
        "protocol_profile": "current_crisp",
        "source_data": {
            "source_split": {"mode": "fraction"},
            "local_split": {"val_fraction": 0.1},
        },
    }
    with pytest.raises(ValueError, match="requires source_data.source_split.mode='manifest'"):
        validate_current_training_protocol(config)


def test_current_source_implicit_fraction_fails_loudly() -> None:
    config = {
        "protocol_profile": "current_crisp",
        "source_data": {"local_split": {"val_fraction": 0.1}},
    }
    with pytest.raises(ValueError, match="requires source_data.source_split"):
        validate_current_training_protocol(config)


def test_structured_current_source_fails_only_for_absent_manifest_artifact() -> None:
    config = {
        "protocol_profile": "current_crisp",
        "source_data": {
            "local_split": None,
            "source_split": {
                "mode": "manifest",
                "count_profile": "current_crisp",
                "train_manifest": None,
                "val_manifest": None,
            },
        },
    }
    with pytest.raises(ValueError, match="explicit source train manifest path"):
        validate_current_training_protocol(config)


@pytest.mark.parametrize("key", ["train_manifest", "val_manifest"])
def test_current_source_missing_manifest_fails(tmp_path: Path, key: str) -> None:
    config = _valid_source_config(tmp_path)
    config["source_data"]["source_split"][key] = str(tmp_path / "absent.txt")
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        validate_current_training_protocol(config)


def test_current_source_rejects_inherited_fraction_controls(tmp_path: Path) -> None:
    config = _valid_source_config(tmp_path)
    config["source_data"]["local_split"] = {
        "val_fraction": 0.1,
        "cache_dir": str(tmp_path / "cache"),
    }
    with pytest.raises(ValueError, match="forbids legacy local_split"):
        validate_current_training_protocol(config)


def test_current_source_accepts_explicit_manifest_contract(tmp_path: Path) -> None:
    validate_current_training_protocol(_valid_source_config(tmp_path))


def test_current_protocol_rejects_unverified_stronger_host_scaffold(
    tmp_path: Path,
) -> None:
    config = _valid_source_config(tmp_path)
    config["model"] = {"name": "rabbit"}
    with pytest.raises(ValueError, match="External stronger-host adapters require"):
        validate_current_training_protocol(config)


def test_debug_fraction_mode_remains_available() -> None:
    validate_current_training_protocol(
        {"source_data": {"local_split": {"val_fraction": 0.1}}}
    )


def test_current_evaluation_discovery_fails_before_membership_resolution(tmp_path: Path) -> None:
    config = _valid_evaluation_config(tmp_path)
    config["eval"]["auto_discover_local_test_datasets"] = True
    del config["eval"]["membership_manifests"]
    with pytest.raises(ValueError, match="requires eval.membership_manifests"):
        validate_current_evaluation_protocol(config)


def test_current_evaluation_requires_every_explicit_manifest(tmp_path: Path) -> None:
    config = _valid_evaluation_config(tmp_path)
    del config["eval"]["membership_manifests"]["ETIS"]
    with pytest.raises(ValueError, match="one explicit evaluation manifest"):
        validate_current_evaluation_protocol(config)


@pytest.mark.parametrize(
    "datasets, error",
    [
        (["Kvasir-SEG", "CVC-ClinicDB", "CVC-300", "CVC-ColonDB"], "missing=.*ETIS"),
        (
            [
                "Kvasir-SEG",
                "CVC-ClinicDB",
                "CVC-300",
                "CVC-ColonDB",
                "ETIS",
                "PolypGen",
            ],
            "extra=.*PolypGen",
        ),
        (
            ["Kvasir-SEG", "Kvasir", "CVC-ClinicDB", "CVC-300", "CVC-ColonDB", "ETIS"],
            "duplicate scientific identities",
        ),
        (
            ["Kvasir", "CVC-ClinicDB", "CVC-300", "CVC-ColonDB", "ETIS"],
            "requires canonical evaluation dataset identities",
        ),
    ],
)
def test_current_evaluation_requires_exact_suite(
    tmp_path: Path, datasets: list[str], error: str
) -> None:
    config = _valid_evaluation_config(tmp_path)
    config["eval_datasets"] = datasets
    with pytest.raises(ValueError, match=error):
        validate_current_evaluation_protocol(config)


def test_current_evaluation_accepts_exact_explicit_contract(tmp_path: Path) -> None:
    config = _valid_evaluation_config(tmp_path)
    config["eval"]["auto_discover_local_test_datasets"] = True
    validate_current_evaluation_protocol(config)


def test_current_evaluation_rejects_discovered_membership_after_dataset_build(
    tmp_path: Path,
) -> None:
    config = _valid_evaluation_config(tmp_path)
    with pytest.raises(ValueError, match="requires explicit evaluation membership"):
        validate_current_evaluation_membership_mode(
            config,
            "ETIS",
            {"dataset": "ETIS", "mode": "discovered_full_dataset"},
        )
    validate_current_evaluation_membership_mode(
        config,
        "ETIS",
        {"dataset": "ETIS", "mode": "explicit_manifest"},
    )


def test_debug_discovery_remains_available() -> None:
    validate_current_evaluation_protocol(
        {"eval_datasets": ["toy"], "eval": {"auto_discover_local_test_datasets": True}}
    )


def test_unknown_protocol_profile_cannot_silently_become_debug() -> None:
    with pytest.raises(ValueError, match="Unknown protocol_profile"):
        validate_current_training_protocol({"protocol_profile": "current_crsip"})


def test_current_guards_run_before_output_or_model_side_effects() -> None:
    train_source = (ROOT / "src/crisp/scripts/train.py").read_text(encoding="utf-8")
    evaluate_source = (ROOT / "src/crisp/scripts/evaluate.py").read_text(encoding="utf-8")
    assert train_source.index("validate_current_training_protocol(config)") < train_source.index(
        "seed_everything(seed)"
    )
    assert train_source.index("validate_current_training_protocol(config)") < train_source.index(
        "output_dir = ensure_dir"
    )
    assert evaluate_source.index(
        "validate_current_evaluation_protocol(config)"
    ) < evaluate_source.index("seed_everything(config.get")
    assert evaluate_source.index(
        "validate_current_evaluation_protocol(config)"
    ) < evaluate_source.index("output_dir = ensure_dir")


def test_all_retained_current_configs_opt_into_guard_and_debug_configs_do_not() -> None:
    experiment_dir = ROOT / "configs/experiment"
    assert {path.name for path in experiment_dir.glob("thesis_*.yaml")} == CURRENT_CONFIGS
    for name in CURRENT_CONFIGS:
        text = (experiment_dir / name).read_text(encoding="utf-8")
        assert "protocol_profile: current_crisp" in text
        assert "membership_count_profile: current_crisp" in text
        assert SOURCE_CONTRACT in text
        assert "val_fraction" not in text
        assert 'train_split: "fixed explicit source membership is required"' in text
        assert (
            "eval_datasets: [Kvasir-SEG, CVC-ClinicDB, CVC-300, CVC-ColonDB, ETIS]"
            in text
        )
    for path in experiment_dir.glob("task*.yaml"):
        assert "protocol_profile:" not in path.read_text(encoding="utf-8")


def test_touched_production_files_use_neutral_current_protocol_language() -> None:
    for relative_path in (
        "src/crisp/protocol.py",
        "src/crisp/scripts/train.py",
        "src/crisp/scripts/evaluate.py",
    ):
        text = (ROOT / relative_path).read_text(encoding="utf-8").lower()
        assert "paper-faithful" not in text
