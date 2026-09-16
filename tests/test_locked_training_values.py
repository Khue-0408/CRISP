"""Locked scalar training values in the six journal-canonical experiments."""

from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "configs" / "experiment"


def _value(text: str, key: str, indent: int = 2) -> str:
    match = re.search(r"(?m)^" + " " * indent + rf"{re.escape(key)}: ([^\r\n]+)$", text)
    assert match is not None, f"Missing {key} at indentation {indent}"
    return match.group(1)


@pytest.mark.parametrize(
    ("host", "batch_size"), [("unet", 16), ("unetpp", 14), ("pranet", 12)]
)
@pytest.mark.parametrize("variant", ["baseline", "crisp"])
def test_canonical_experiment_training_values(host: str, batch_size: int, variant: str) -> None:
    text = (EXPERIMENTS / f"thesis_{host}_{variant}.yaml").read_text(encoding="utf-8")
    assert f"  - /model: {host}\n" in text
    assert "  - /data@source_data: thesis_train_test\n" in text
    assert int(_value(text, "batch_size")) == batch_size
    assert int(_value(text, "total_epochs")) == 120
    assert _value(text, "epochs") == "${training.total_epochs}"
    assert [_value(text, key, 4) for key in
            ("baseline_warmup", "crisp_full", "finetune", "phase2_ramp_epochs")] == [
                "25", "65", "30", "10"
            ]
    assert _value(text, "optimizer") == "adamw"
    assert float(_value(text, "lr_student")) == pytest.approx(1e-4)
    assert float(_value(text, "weight_decay")) == pytest.approx(1e-4)
    assert _value(text, "scheduler") == "cosine"
    assert _value(text, "mixed_precision") == "true"
    assert float(_value(text, "gradient_clip_norm")) == pytest.approx(1.0)
    if variant == "crisp":
        assert "  - /crisp: default\n" in text
        assert float(_value(text, "lr_projector")) == pytest.approx(5e-4)
    else:
        assert not re.search(r"(?m)^  lr_projector:", text)


@pytest.mark.parametrize(("host", "batch_size"), [("unet", 16), ("unetpp", 14), ("pranet", 12)])
def test_canonical_baseline_crisp_training_parity(host: str, batch_size: int) -> None:
    baseline = (EXPERIMENTS / f"thesis_{host}_baseline.yaml").read_text(encoding="utf-8")
    crisp = (EXPERIMENTS / f"thesis_{host}_crisp.yaml").read_text(encoding="utf-8")
    for key in ("batch_size", "total_epochs", "lr_student", "weight_decay", "mixed_precision", "gradient_clip_norm"):
        assert _value(baseline, key) == _value(crisp, key)
    assert int(_value(baseline, "batch_size")) == batch_size


def test_canonical_data_resolution_and_solver_config() -> None:
    data = (ROOT / "configs" / "data" / "thesis_train_test.yaml").read_text(encoding="utf-8")
    crisp = (ROOT / "configs" / "crisp" / "default.yaml").read_text(encoding="utf-8")
    assert re.search(r"(?m)^image_size: 352$", data)
    assert _value(crisp, "newton_steps") == "3"
    assert _value(crisp, "bisection_steps") == "12"
