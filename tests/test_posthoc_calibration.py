"""Scientific protocol gates for post-hoc calibration controls."""

import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from crisp.modules.boundary import compute_boundary_weight
from crisp.modules.posthoc import (
    LOCAL_TS_PROTOCOL_BLOCK_MESSAGE,
    LegacyTargetDependentLocalTemperatureScaler,
    LocalTemperatureScaler,
    TemperatureScaler,
)


def test_legacy_local_ts_target_mask_changes_assignment_and_prediction() -> None:
    logits = torch.linspace(-3.0, 3.0, 64).reshape(1, 1, 8, 8)
    left_mask = torch.zeros_like(logits)
    left_mask[:, :, 2:6, 1:4] = 1.0
    right_mask = torch.zeros_like(logits)
    right_mask[:, :, 2:6, 4:7] = 1.0

    left_wb = compute_boundary_weight(left_mask, sigma_b=2.0)
    right_wb = compute_boundary_weight(right_mask, sigma_b=2.0)
    calibrator = LegacyTargetDependentLocalTemperatureScaler(n_bins=2)
    calibrator.temperatures = torch.tensor([0.5, 2.0])  # fixed fitted state

    left_assignment = calibrator.diagnostic_bin_assignments(left_wb)
    right_assignment = calibrator.diagnostic_bin_assignments(right_wb)
    left_probs = calibrator.transform(logits, left_wb)
    right_probs = calibrator.transform(logits, right_wb)

    assert not torch.equal(left_wb, right_wb)
    assert not torch.equal(left_assignment, right_assignment)
    assert not torch.equal(left_probs, right_probs)


def test_global_ts_target_prediction_is_mask_blind_and_preserves_hard_mask() -> None:
    logits = torch.tensor([-4.0, -0.5, 0.0, 0.5, 4.0]).reshape(1, 1, 1, 5)
    first_target_mask = torch.zeros_like(logits)
    second_target_mask = torch.ones_like(logits)
    calibrator = TemperatureScaler()
    calibrator.temperature = 1.7

    first_probs = calibrator.transform(logits)
    second_probs = calibrator.transform(logits)
    assert not torch.equal(first_target_mask, second_target_mask)
    assert torch.equal(first_probs, second_probs)
    assert torch.equal(first_probs >= 0.5, logits >= 0.0)


@pytest.mark.parametrize("temperature", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_positive_global_temperature_preserves_fixed_logit_hard_mask(temperature: float) -> None:
    logits = torch.linspace(-8.0, 8.0, 65)
    calibrator = TemperatureScaler()
    calibrator.temperature = temperature
    assert torch.equal(calibrator.transform(logits) >= 0.5, logits >= 0.0)


def test_canonical_local_ts_class_is_blocked() -> None:
    with pytest.raises(RuntimeError, match="target-derived boundary information"):
        LocalTemperatureScaler(n_bins=2)


def test_public_calibration_entry_blocks_local_ts_before_artifacts(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    hydra = ModuleType("hydra")
    hydra.main = lambda **_kwargs: lambda function: function
    hydra_utils = ModuleType("hydra.utils")
    hydra_utils.to_absolute_path = lambda path: str(path)
    omegaconf = ModuleType("omegaconf")
    omegaconf.DictConfig = dict

    class OmegaConf:
        @staticmethod
        def to_container(cfg: dict, **_kwargs: object) -> dict:
            return cfg

    omegaconf.OmegaConf = OmegaConf
    checkpointing = ModuleType("crisp.engine.checkpointing")
    checkpointing.load_checkpoint = lambda _path: pytest.fail("checkpoint loading must not run")
    registry = ModuleType("crisp.registry")
    registry.build_dataset = lambda *_args, **_kwargs: pytest.fail("dataset building must not run")
    registry.build_model = lambda *_args, **_kwargs: pytest.fail("model building must not run")
    monkeypatch.setitem(sys.modules, "hydra", hydra)
    monkeypatch.setitem(sys.modules, "hydra.utils", hydra_utils)
    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf)
    monkeypatch.setitem(sys.modules, "crisp.engine.checkpointing", checkpointing)
    monkeypatch.setitem(sys.modules, "crisp.registry", registry)

    script = runpy.run_module("crisp.scripts.posthoc_calibrate", run_name="posthoc_block_test")
    output_dir = tmp_path / "posthoc"
    with pytest.raises(RuntimeError, match="target-derived boundary information") as exc:
        script["main"]({
            "posthoc_methods": ["ts", "lts"],
            "posthoc_output_dir": str(output_dir),
        })
    assert str(exc.value) == LOCAL_TS_PROTOCOL_BLOCK_MESSAGE
    assert not output_dir.exists()
    assert list(tmp_path.rglob("*.json")) == []


def test_posthoc_target_evaluation_has_no_polypgen_or_implicit_suite_default() -> None:
    source = (Path(__file__).resolve().parents[1] / "src/crisp/scripts/posthoc_calibrate.py").read_text(
        encoding="utf-8"
    )
    assert 'config.get("eval_datasets", ["colondb", "etis", "polypgen"])' not in source
    assert "requires an explicit nonempty eval_datasets list" in source
    assert 'bnd_cfg.get("sigma_b", 6.0)' in source
