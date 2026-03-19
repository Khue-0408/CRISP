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

from pathlib import Path

import torch
from torch.utils.data import DataLoader

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from crisp.engine.checkpointing import load_checkpoint
from crisp.engine.evaluator import Evaluator
from crisp.registry import build_dataset, build_model, build_projector
from crisp.utils.logging import setup_logger
from crisp.utils.seed import seed_everything
from crisp.utils.serialization import save_json


def _resolve_eval_dataset_config(config: dict, dataset_name: str) -> dict:
    eval_data = config.get("eval_data", {})
    if isinstance(eval_data, dict) and dataset_name in eval_data:
        return {**config, "source_data": eval_data[dataset_name]}
    raise KeyError(
        f"Missing eval_data entry for dataset '{dataset_name}'. "
        "Add it to the experiment config to avoid evaluating source data by accident."
    )


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main evaluation entry point.
    """
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert isinstance(config, dict), "Hydra config must resolve to a dict-like structure."

    seed_everything(config.get("seed", 0))
    output_dir = Path(to_absolute_path(config.get("eval_output_dir", "outputs/eval")))
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(output_dir)

    # Build model.
    model = build_model(config)
    projector = None
    method_cfg = config.get("method", {})
    if method_cfg.get("use_projector", False):
        decoder_ch = getattr(model, "decoder_channels", 32)
        projector = build_projector(config, in_channels=decoder_ch)

    # Load checkpoint.
    checkpoint_path = config.get("checkpoint", None)
    if checkpoint_path is None:
        raise ValueError(
            "Missing required `checkpoint` in config. Provide it via Hydra override, e.g. "
            "`checkpoint=/path/to/best.pt`."
        )
    ckpt = load_checkpoint(Path(to_absolute_path(str(checkpoint_path))))
    model.load_state_dict(ckpt["model_state_dict"])
    if projector is not None and ckpt.get("projector_state_dict") is not None:
        projector.load_state_dict(ckpt["projector_state_dict"])

    # Evaluate on each target dataset.
    evaluator = Evaluator(model, projector, config)

    datasets = config.get("eval_datasets", ["colondb", "etis", "polypgen"])
    projector_off_only = bool(config.get("projector_off_only", False))

    for ds_name in datasets:
        try:
            ds_config = _resolve_eval_dataset_config(config, ds_name)
            dataset = build_dataset(ds_config, split="test")
        except (FileNotFoundError, KeyError):
            print(f"Skipping {ds_name}: dataset not found.")
            continue

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
        if (not projector_off_only) and (projector is not None):
            metrics_on = evaluator.evaluate_dataset(loader, ds_name, projector_on=True)
            save_json(
                output_dir / f"{ds_name}_projector_on.json", metrics_on
            )

        # Projector-off ablation.
        metrics_off = evaluator.evaluate_dataset(loader, ds_name, projector_on=False)
        save_json(
            output_dir / f"{ds_name}_projector_off.json", metrics_off
        )


if __name__ == "__main__":
    main()
