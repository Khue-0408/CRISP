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
from crisp.data.evaluation_membership import (
    EVALUATION_STORAGE_ALIASES,
    CURRENT_EVALUATION_COUNTS,
    canonical_evaluation_dataset_name,
    resolve_evaluation_dataset_identity,
    validate_evaluation_membership,
)
from crisp.engine.checkpointing import load_checkpoint, load_required_projector_state
from crisp.engine.evaluator import Evaluator
from crisp.protocol import (
    validate_current_evaluation_membership_mode,
    validate_current_evaluation_protocol,
)
from crisp.registry import (
    build_dataset,
    build_model,
    build_projector,
    get_model_decoder_channels,
)
from crisp.utils.logging import setup_logger
from crisp.utils.paths import ensure_dir, ensure_local_workspace, resolve_path
from crisp.utils.provenance import evaluation_provenance, write_provenance
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
            ordered_names = [canonical_evaluation_dataset_name(name) for name in requested]
            if len(ordered_names) != len(set(ordered_names)):
                raise ValueError("Evaluation dataset aliases resolve to duplicate scientific identities.")
            missing = [name for name in ordered_names if name not in discovered]
            if missing:
                raise KeyError(
                    "Requested local evaluation datasets were not discovered under "
                    f"TestDataset: {missing}"
                )
        else:
            ordered_names = sorted(discovered.keys())
        return [
            (name, {**config, "source_data": discovered[name]})
            for name in ordered_names
        ]

    datasets = requested or ["colondb", "etis", "polypgen"]
    entries: list[tuple[str, dict]] = []
    for lookup_name in datasets:
        dataset_config = _resolve_eval_dataset_config(config, lookup_name)
        data_config = dataset_config.get("source_data", {})
        configured_name = data_config.get("name", lookup_name)
        canonical_name = resolve_evaluation_dataset_identity(
            configured_name, data_config.get("evaluation_dataset_name")
        )
        if lookup_name in CURRENT_EVALUATION_COUNTS or lookup_name in EVALUATION_STORAGE_ALIASES:
            if canonical_evaluation_dataset_name(lookup_name) != canonical_name:
                raise ValueError(
                    f"Evaluation lookup identity {lookup_name!r} conflicts with {canonical_name!r}."
                )
        entries.append((canonical_name, dataset_config))
    names = [name for name, _ in entries]
    if len(names) != len(set(names)):
        raise ValueError("Evaluation configs resolve to duplicate scientific dataset identities.")
    return entries


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main evaluation entry point.
    """
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert isinstance(config, dict), "Hydra config must resolve to a dict-like structure."
    validate_current_evaluation_protocol(config)

    seed_everything(config.get("seed", 0))
    workspace_cfg = config.get("workspace", {})
    if bool(workspace_cfg.get("auto_create", False)):
        ensure_local_workspace(workspace_cfg.get("root", "."))
    output_dir = ensure_dir(resolve_path(config.get("eval_output_dir", "outputs/eval")))
    if config.get("experiment_name") is not None and (output_dir.parent.name, output_dir.name) != (
        config["experiment_name"], f"seed_{int(config.get('seed', 0))}"
    ):
        raise ValueError("Evaluation output path conflicts with experiment/seed identity.")
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
    resolved_checkpoint_path = resolve_path(str(checkpoint_path))
    ckpt = load_checkpoint(resolved_checkpoint_path)
    model.load_state_dict(ckpt["model_state_dict"])
    if method_cfg.get("use_projector", False):
        load_required_projector_state(projector, ckpt, resolved_checkpoint_path)

    # Evaluate on each target dataset.
    evaluator = Evaluator(model, projector, config)

    projector_off_only = bool(config.get("projector_off_only", False))
    skip_missing = bool(config.get("eval", {}).get("skip_missing_datasets", False))
    summary_rows: list[dict[str, object]] = []

    def save_metrics(dataset_dir, dataset_name: str, mode: str, metrics: dict, membership: dict, membership_path) -> None:
        metric_path = dataset_dir / f"{mode}.json"
        sidecar_path = dataset_dir / f"{mode}.provenance.json"
        if metric_path.exists() or sidecar_path.exists():
            raise FileExistsError(f"Evaluation artifact already exists: {metric_path}")
        record = evaluation_provenance(
            config, resolved_checkpoint_path, ckpt, dataset_name, mode, metric_path,
            membership=membership, membership_file=membership_path,
        )
        write_provenance(sidecar_path, record)
        save_json(metric_path, metrics)

    eval_cfg = config.get("eval", {})
    manifest_map = eval_cfg.get("membership_manifests", {})
    if not isinstance(manifest_map, dict):
        raise ValueError("eval.membership_manifests must map dataset names to paths.")
    entries = _resolve_eval_dataset_entries(config)
    unknown_manifest_names = set(manifest_map) - {name for name, _ in entries}
    if unknown_manifest_names:
        raise ValueError(f"Evaluation manifests configured for unknown datasets: {sorted(unknown_manifest_names)}")
    if any(not isinstance(path, str) or not path.strip() for path in manifest_map.values()):
        raise ValueError("Every configured evaluation manifest requires a nonempty path.")

    for ds_name, ds_config in entries:
        data_cfg = dict(ds_config.get("source_data", {}))
        data_cfg["evaluation_dataset_name"] = ds_name
        if ds_name in manifest_map:
            if data_cfg.get("evaluation_manifest") and data_cfg["evaluation_manifest"] != manifest_map[ds_name]:
                raise ValueError(f"Conflicting evaluation manifests configured for {ds_name}.")
            data_cfg["evaluation_manifest"] = manifest_map[ds_name]
        if eval_cfg.get("membership_count_profile") is not None:
            data_cfg["evaluation_count_profile"] = eval_cfg["membership_count_profile"]
        ds_config = {**ds_config, "source_data": data_cfg}
        try:
            dataset = build_dataset(ds_config, split="test")
        except (FileNotFoundError, KeyError) as exc:
            if skip_missing and not data_cfg.get("evaluation_manifest"):
                print(f"Skipping {ds_name}: dataset not found.")
                continue
            raise ValueError(
                f"Requested evaluation dataset '{ds_name}' could not be built. "
                "Current-protocol target evaluation should fail loudly when a target "
                "dataset is missing."
            ) from exc

        membership = getattr(dataset, "evaluation_membership_provenance", None)
        if not isinstance(membership, dict):
            raise ValueError(f"Evaluation dataset {ds_name} has no membership provenance.")
        validate_evaluation_membership(membership)
        validate_current_evaluation_membership_mode(config, ds_name, membership)
        if membership["dataset"] != ds_name or membership["sample_count"] != len(dataset):
            raise ValueError(f"Evaluation dataset {ds_name} conflicts with its membership record.")

        eval_data_cfg = ds_config.get("source_data", {})
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
        membership_path = dataset_dir / "dataset.membership.json"
        write_provenance(membership_path, membership)
        if (not projector_off_only) and (projector is not None):
            metrics_on = evaluator.evaluate_dataset(loader, ds_name, projector_on=True)
            save_metrics(dataset_dir, ds_name, "projector_on", metrics_on, membership, membership_path)
            summary_rows.append(
                {"dataset": ds_name, "mode": "projector_on", **metrics_on}
            )

        # Projector-off ablation.
        metrics_off = evaluator.evaluate_dataset(loader, ds_name, projector_on=False)
        save_metrics(dataset_dir, ds_name, "projector_off", metrics_off, membership, membership_path)
        summary_rows.append(
            {"dataset": ds_name, "mode": "projector_off", **metrics_off}
        )

    save_json(output_dir / "summary.json", {"results": summary_rows})
    save_csv(output_dir / "summary.csv", summary_rows)


if __name__ == "__main__":
    main()
