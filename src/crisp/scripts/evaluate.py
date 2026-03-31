"""
CLI entry point for evaluation.

This script should:
- load trained checkpoints,
- run target-domain inference,
- compute segmentation and calibration metrics,
- export per-dataset JSON and CSV artifacts,
- optionally run projector-on and projector-off evaluation modes.

CRISP invariants preserved
-------------------------
This file must not alter CRISP’s mathematical identity:
- no teacher usage at inference (per `instruct.md` §14),
- no per-pixel optimization at inference (solver is train-time only),
- projector-on uses bounded alpha_hat; projector-off sets alpha_hat = 1,
- forward path uses raw logits z and calibrated probs p̃ = sigmoid(alpha_hat * z)
  (implemented in `Evaluator.predict_batch`).
"""

from __future__ import annotations

import re

import torch
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf

from crisp.data.datasets import discover_local_test_datasets
from crisp.engine.checkpointing import load_checkpoint
from crisp.engine.evaluator import Evaluator
from crisp.registry import (
    build_dataset,
    build_model,
    build_projector,
    get_model_decoder_channels,
)
from crisp.utils.logging import setup_logger
from crisp.utils.paths import ensure_dir, ensure_local_workspace, resolve_path
from crisp.utils.seed import seed_everything
from crisp.utils.serialization import save_csv, save_json


def _resolve_eval_dataset_config(config: dict, dataset_name: str) -> dict:
    eval_data = config.get("eval_data", {})
    if isinstance(eval_data, dict) and dataset_name in eval_data:
        return {**config, "source_data": eval_data[dataset_name]}
    raise KeyError(
        f"Missing eval_data entry for dataset '{dataset_name}'. "
        "Add it to the experiment config to avoid evaluating source data by accident."
    )


def _safe_dataset_slug(dataset_name: str) -> str:
    """
    Convert a dataset name into a filesystem-safe slug for exports.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", dataset_name)


def _resolve_eval_dataset_entries(config: dict) -> list[tuple[str, dict]]:
    """
    Resolve the ordered evaluation dataset list for either paper mode or local mode.
    """
    eval_cfg = config.get("eval", {})
    source_data_cfg = config.get("source_data", {})
    requested = list(config.get("eval_datasets", []))

    if bool(eval_cfg.get("auto_discover_local_test_datasets", False)):
        discovered = discover_local_test_datasets(source_data_cfg)
        if requested:
            missing = [name for name in requested if name not in discovered]
            if missing:
                raise KeyError(
                    "Requested local evaluation datasets were not discovered under "
                    f"TestDataset: {missing}"
                )
            ordered_names = requested
        else:
            ordered_names = sorted(discovered.keys())
        return [
            (name, {**config, "source_data": discovered[name]})
            for name in ordered_names
        ]

    datasets = requested or ["colondb", "etis", "polypgen"]
    return [
        (name, _resolve_eval_dataset_config(config, name))
        for name in datasets
    ]


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main evaluation entry point.
    """
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert isinstance(config, dict), "Hydra config must resolve to a dict-like structure."

    seed_everything(config.get("seed", 0))
    workspace_cfg = config.get("workspace", {})
    if bool(workspace_cfg.get("auto_create", False)):
        ensure_local_workspace(workspace_cfg.get("root", "."))
    output_dir = ensure_dir(resolve_path(config.get("eval_output_dir", "outputs/eval")))
    setup_logger(output_dir)

    # Build model.
    model = build_model(config)
    projector = None
    method_cfg = config.get("method", {})
    if method_cfg.get("use_projector", False):
        decoder_ch = get_model_decoder_channels(model)
        projector = build_projector(config, in_channels=decoder_ch)

    # Load checkpoint.
    checkpoint_path = config.get("checkpoint", None)
    if checkpoint_path is None:
        raise ValueError(
            "Missing required `checkpoint` in config. Provide it via Hydra override, e.g. "
            "`checkpoint=/path/to/best.pt`."
        )
    ckpt = load_checkpoint(resolve_path(str(checkpoint_path)))
    model.load_state_dict(ckpt["model_state_dict"])
    if projector is not None and ckpt.get("projector_state_dict") is not None:
        projector.load_state_dict(ckpt["projector_state_dict"])

    # Evaluate on each target dataset.
    evaluator = Evaluator(model, projector, config)

    projector_off_only = bool(config.get("projector_off_only", False))
    skip_missing = bool(config.get("eval", {}).get("skip_missing_datasets", False))
    summary_rows: list[dict[str, object]] = []

    for ds_name, ds_config in _resolve_eval_dataset_entries(config):
        try:
            dataset = build_dataset(ds_config, split="test")
        except (FileNotFoundError, KeyError) as exc:
            if skip_missing:
                print(f"Skipping {ds_name}: dataset not found.")
                continue
            raise ValueError(
                f"Requested evaluation dataset '{ds_name}' could not be built. "
                "Paper-faithful target evaluation should fail loudly when a target "
                "dataset is missing."
            ) from exc

        eval_data_cfg = ds_config.get("source_data", {})
        eval_cfg = config.get("eval", {})
        loader = DataLoader(
            dataset,
            batch_size=int(eval_cfg.get("batch_size", 8)),
            shuffle=False,
            num_workers=int(eval_data_cfg.get("num_workers", 4)),
            pin_memory=bool(eval_data_cfg.get("pin_memory", True)),
        )

        # Projector-on evaluation.
        dataset_slug = _safe_dataset_slug(ds_name)
        dataset_dir = ensure_dir(output_dir / dataset_slug)
        if (not projector_off_only) and (projector is not None):
            metrics_on = evaluator.evaluate_dataset(loader, ds_name, projector_on=True)
            save_json(dataset_dir / "projector_on.json", metrics_on)
            summary_rows.append(
                {"dataset": ds_name, "mode": "projector_on", **metrics_on}
            )

        # Projector-off ablation.
        metrics_off = evaluator.evaluate_dataset(loader, ds_name, projector_on=False)
        save_json(dataset_dir / "projector_off.json", metrics_off)
        summary_rows.append(
            {"dataset": ds_name, "mode": "projector_off", **metrics_off}
        )

    save_json(output_dir / "summary.json", {"results": summary_rows})
    save_csv(output_dir / "summary.csv", summary_rows)


if __name__ == "__main__":
    main()
