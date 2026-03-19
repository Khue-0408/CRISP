"""
Post-hoc calibration entry point (source-only protocol).

CRISP invariants preserved
-------------------------
This script must not change CRISP’s training objective or inference definition.
It operates strictly as a *post-hoc* calibrator:
- fits calibrator parameters on **source validation** only (no target leakage),
- evaluates on frozen logits/probabilities from a trained checkpoint,
- does not invoke CRISP’s detached local solver at inference,
- does not change geometry training code paths.

Supported post-hoc baselines (minimal faithful)
----------------------------------------------
- global temperature scaling (TS),
- boundary temperature scaling (bTS) on top-k w_b pixels,
- selective temperature scaling (STS),
- local temperature scaling (LTS) via w_b quantile bins,
- histogram binning (BBQ-style lightweight baseline).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from crisp.engine.checkpointing import load_checkpoint
from crisp.metrics.calibration import (
    boundary_area_weighted_ece,
    boundary_expected_calibration_error,
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
    off_boundary_expected_calibration_error,
    thresholded_adaptive_calibration_error,
)
from crisp.modules.boundary import compute_boundary_weight
from crisp.modules.posthoc import (
    BoundaryTemperatureScaler,
    HistogramBinningCalibrator,
    LocalTemperatureScaler,
    PostHocFitArtifacts,
    SelectiveTemperatureScaler,
    TemperatureScaler,
)
from crisp.registry import build_dataset, build_model
from crisp.utils.logging import setup_logger
from crisp.utils.seed import seed_everything
from crisp.utils.serialization import save_json


@torch.no_grad()
def _collect_val_tensors(
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
    sigma_b: float,
    boundary_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collect frozen logits, labels, and boundary weights from a source validation loader.
    """
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_wb: List[torch.Tensor] = []
    for batch in dataloader:
        x = batch["image"].to(device)
        y = batch["mask"].to(device)
        out = model(x)
        z = out.logits.detach()
        wb = compute_boundary_weight(y, sigma_b=sigma_b, mode=boundary_mode).detach()
        all_logits.append(z)
        all_labels.append(y.detach())
        all_wb.append(wb)
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0), torch.cat(all_wb, dim=0)


def _resolve_eval_dataset_config(config: dict, dataset_name: str) -> dict:
    eval_data = config.get("eval_data", {})
    if isinstance(eval_data, dict) and dataset_name in eval_data:
        return {**config, "source_data": eval_data[dataset_name]}
    raise KeyError(
        f"Missing eval_data entry for dataset '{dataset_name}'. "
        "Add it to the experiment config to avoid target-eval/source-data leakage."
    )


def _compute_metrics_from_probs(
    probs: torch.Tensor,
    labels: torch.Tensor,
    wb: torch.Tensor,
    ece_bins: int,
    top_percent: float,
    tace_threshold: float,
) -> Dict[str, float]:
    # Aggregate globally (flatten).
    ece = expected_calibration_error(probs, labels, n_bins=ece_bins).item()
    bece = boundary_expected_calibration_error(
        probs, labels, wb, n_bins=ece_bins, top_percent=top_percent
    ).item()
    ba_ece = boundary_area_weighted_ece(
        probs, labels, wb, n_bins=ece_bins, top_percent=top_percent
    ).item()
    off_bece = off_boundary_expected_calibration_error(
        probs, labels, wb, n_bins=ece_bins, top_percent=top_percent
    ).item()
    tace = thresholded_adaptive_calibration_error(
        probs, labels, threshold=tace_threshold
    ).item()
    brier = brier_score(probs, labels).item()
    nll = negative_log_likelihood(probs, labels).item()
    return {
        "ece": ece,
        "bece": bece,
        "ba_ece": ba_ece,
        "off_bece": off_bece,
        "tace": tace,
        "brier": brier,
        "nll": nll,
    }


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert isinstance(config, dict)

    seed_everything(config.get("seed", 0))
    output_dir = Path(to_absolute_path(config.get("posthoc_output_dir", "outputs/posthoc")))
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(output_dir)

    checkpoint_path = config.get("checkpoint", None)
    if checkpoint_path is None:
        raise ValueError("Missing required `checkpoint` override for post-hoc calibration.")

    # Build model (projector intentionally unused here: post-hoc is applied to frozen logits).
    model = build_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ckpt = load_checkpoint(Path(to_absolute_path(str(checkpoint_path))))
    model.load_state_dict(ckpt["model_state_dict"])

    # Build SOURCE validation loader only (source-only protocol).
    val_cfg = config.get("posthoc_val_data", None)
    if val_cfg is None:
        # Fallback: use source_data with split="val".
        val_cfg = config.get("source_data", {})
    val_ds_config = {**config, "source_data": val_cfg}
    val_dataset = build_dataset(val_ds_config, split="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config.get("posthoc_batch_size", 8)),
        shuffle=False,
        num_workers=int(val_cfg.get("num_workers", 4)),
        pin_memory=bool(val_cfg.get("pin_memory", True)),
    )

    crisp_cfg = config.get("crisp", {})
    bnd_cfg = crisp_cfg.get("boundary", {})
    sigma_b = float(bnd_cfg.get("sigma_b", 3.0))
    boundary_mode = str(bnd_cfg.get("mode", "gaussian_soft_field"))

    eval_cfg = config.get("eval", {})
    ece_bins = int(eval_cfg.get("ece", {}).get("bins", 15)) if isinstance(eval_cfg, dict) else 15
    top_percent = float(eval_cfg.get("boundary_support", {}).get("top_percent", 20.0)) if isinstance(eval_cfg, dict) else 20.0
    tace_threshold = float(eval_cfg.get("tace", {}).get("threshold", 1.0e-3)) if isinstance(eval_cfg, dict) else 1.0e-3

    logits, labels, wb = _collect_val_tensors(model, val_loader, device, sigma_b=sigma_b, boundary_mode=boundary_mode)

    # Fit calibrator(s) on source-val only.
    methods = config.get("posthoc_methods", ["ts"])
    results: Dict[str, Any] = {"checkpoint": str(checkpoint_path), "methods": {}}
    fitted_calibrators: Dict[str, Any] = {}

    for m in methods:
        m = str(m).lower()
        artifacts: PostHocFitArtifacts | None = None
        if m == "ts":
            cal = TemperatureScaler()
            cal.fit(logits, labels)
            artifacts = PostHocFitArtifacts(method="ts", params={"temperature": float(cal.temperature)})
            probs = cal.transform(logits)
        elif m == "bts":
            cal = BoundaryTemperatureScaler()
            artifacts = cal.fit(logits, labels, wb, top_percent=top_percent)
            probs = cal.transform(logits)
        elif m == "sts":
            thr = float(config.get("posthoc_selective_threshold", 0.5))
            cal = SelectiveTemperatureScaler(threshold=thr)
            cal.fit(logits, labels)
            artifacts = PostHocFitArtifacts(method="sts", params={"temperature": float(cal.temperature), "threshold": thr})
            probs = cal.transform(logits)
        elif m == "lts":
            n_bins = int(config.get("posthoc_local_bins", 2))
            cal = LocalTemperatureScaler(n_bins=n_bins)
            artifacts = cal.fit(logits, labels, wb)
            probs = cal.transform(logits, wb)
        elif m == "histbin":
            cal = HistogramBinningCalibrator(n_bins=ece_bins)
            artifacts = cal.fit(torch.sigmoid(logits), labels)
            probs = cal.transform(torch.sigmoid(logits))
        else:
            raise ValueError(f"Unknown posthoc method '{m}'.")

        fitted_calibrators[m] = cal
        metrics = _compute_metrics_from_probs(probs, labels, wb, ece_bins, top_percent, tace_threshold)
        results["methods"][m] = {"fit": artifacts.params if artifacts else {}, "val_metrics": metrics}

        # Save per-method artifact.
        save_json(output_dir / f"posthoc_{m}_source_val.json", {"fit": artifacts.params if artifacts else {}, "val_metrics": metrics})

    # Optional: evaluate these post-hoc calibrators on target datasets using frozen checkpoint logits.
    if bool(config.get("posthoc_eval_targets", False)):
        datasets = config.get("eval_datasets", ["colondb", "etis", "polypgen"])
        for ds_name in datasets:
            try:
                ds_config = _resolve_eval_dataset_config(config, ds_name)
                dataset = build_dataset(ds_config, split="test")
            except (FileNotFoundError, KeyError):
                continue
            tgt_data_cfg = ds_config.get("source_data", {})
            loader = DataLoader(
                dataset,
                batch_size=int(config.get("posthoc_batch_size", 8)),
                shuffle=False,
                num_workers=int(tgt_data_cfg.get("num_workers", 4)),
                pin_memory=bool(tgt_data_cfg.get("pin_memory", True)),
            )

            # Collect logits/labels/wb on target for consistent metric computation (no fitting).
            tgt_logits, tgt_labels, tgt_wb = _collect_val_tensors(
                model, loader, device, sigma_b=sigma_b, boundary_mode=boundary_mode
            )

            for m in methods:
                m = str(m).lower()
                calibrator = fitted_calibrators[m]
                if m in {"ts", "bts", "sts"}:
                    probs = calibrator.transform(tgt_logits)
                elif m == "lts":
                    probs = calibrator.transform(tgt_logits, tgt_wb)
                elif m == "histbin":
                    probs = calibrator.transform(torch.sigmoid(tgt_logits))
                else:
                    continue

                metrics = _compute_metrics_from_probs(
                    probs, tgt_labels, tgt_wb, ece_bins, top_percent, tace_threshold
                )
                save_json(output_dir / f"{ds_name}_posthoc_{m}.json", metrics)

    save_json(output_dir / "posthoc_summary.json", results)


if __name__ == "__main__":
    main()
