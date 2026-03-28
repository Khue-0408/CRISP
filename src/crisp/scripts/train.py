"""
CLI entry point for training experiments.

CRISP invariants preserved
-------------------------
This file is *wiring only*. It must not modify any mathematical object defined in
`instruct.md`:
- the forward path remains on raw student logits ``z`` (Trainer controls this),
- the local solver branch remains detached and produces explicit ``alpha_star``,
- the projector output is bounded per-pixel in ``[alpha_min, alpha_max]``.

The main responsibility here is to compose Hydra configs coherently with the
`configs/` folder so the repository is runnable via:

  python -m crisp.scripts.train --config-path ../../configs --config-name experiment/taskA_pranet_crisp
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from crisp.models.teacher_wrapper import FrozenTeacher, TeacherEnsemble
from crisp.registry import build_dataset, build_model, build_projector
from crisp.engine.trainer import Trainer
from crisp.utils.logging import setup_logger
from crisp.utils.paths import ensure_dir
from crisp.utils.seed import seed_everything
from crisp.utils.serialization import save_yaml


def _strict_teacher_loading(cfg: dict) -> bool:
    """Return whether the current config requires a fully specified teacher pool."""
    return bool(cfg.get("crisp", {}).get("teacher", {}).get("strict", True))


def _maybe_build_teacher_ensemble(cfg: dict) -> TeacherEnsemble | None:
    """
    Build an optional teacher ensemble from config.

    Expected (minimal) config shape
    -------------------------------
    teachers:
      - model: {name: pranet, in_channels: 3, pretrained: false}
        checkpoint: /path/to/checkpoint.pt

    Returns ``None`` if no valid teachers are specified.

    Notes
    -----
    Teachers are used only during CRISP training to form ``p_T`` and are always
    frozen/detached per `instruct.md` §4 and §13.
    """
    teachers_cfg = cfg.get("teachers", None)
    if not teachers_cfg:
        teachers_cfg = cfg.get("crisp", {}).get("teacher", {}).get("teachers", None)
    if not teachers_cfg:
        return None

    strict = _strict_teacher_loading(cfg)
    teachers: list[FrozenTeacher] = []
    errors: list[str] = []
    for t in teachers_cfg:
        if not isinstance(t, dict):
            errors.append("Teacher config entry must be a mapping.")
            continue
        model_cfg = t.get("model", None)
        ckpt = t.get("checkpoint", None)
        if not model_cfg:
            errors.append("Teacher config entry is missing the `model` block.")
            continue
        if ckpt is None or not str(ckpt).strip():
            errors.append(
                f"Teacher '{model_cfg.get('name', 'unknown')}' is missing a checkpoint path."
            )
            continue

        ckpt_path = to_absolute_path(str(ckpt))
        try:
            teacher_model = build_model({"model": model_cfg})
            teachers.append(
                FrozenTeacher(
                    teacher_model,
                    checkpoint_path=ckpt_path,
                )
            )
        except Exception as exc:
            errors.append(
                f"Failed to build/load teacher '{model_cfg.get('name', 'unknown')}' "
                f"from checkpoint '{ckpt_path}': {exc}"
            )

    if strict and errors:
        joined = "\n- ".join(errors)
        raise ValueError(
            "Paper-faithful CRISP training requires a complete frozen teacher pool.\n"
            f"- {joined}"
        )
    if not teachers:
        return None
    if strict and len(teachers) != len(teachers_cfg):
        raise ValueError(
            "Teacher ensemble is incomplete under strict CRISP mode. "
            f"Built {len(teachers)} teachers from {len(teachers_cfg)} configured entries."
        )
    return TeacherEnsemble(teachers)


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main training entry point invoked from the command line.

    Supports both baseline and CRISP methods from the same entry point
    using configuration switches.
    """
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert isinstance(config, dict), "Hydra config must resolve to a dict-like structure."

    # Setup.
    seed = config.get("seed", 0)
    seed_everything(seed)
    # Hydra changes the working directory; always anchor outputs to the original cwd.
    output_dir = ensure_dir(Path(to_absolute_path(config.get("output_dir", "outputs"))))
    setup_logger(output_dir)
    save_yaml(output_dir / "resolved_config.yaml", config)

    # Build model.
    model = build_model(config)

    # Build projector (if CRISP).
    projector = None
    method_cfg = config.get("method", {})
    if method_cfg.get("use_projector", False):
        decoder_ch = getattr(model, "decoder_channels", 32)
        projector = build_projector(config, in_channels=decoder_ch)

    # Build teacher ensemble (if CRISP + teachers).
    teacher_ensemble = _maybe_build_teacher_ensemble(config)

    # Build datasets and dataloaders.
    train_cfg = config.get("training", {})
    require_validation = bool(train_cfg.get("require_validation", False))
    batch_size = train_cfg.get("batch_size", 16)
    data_cfg = config.get("source_data", {})
    num_workers = int(data_cfg.get("num_workers", train_cfg.get("num_workers", 4)))
    pin_memory = bool(data_cfg.get("pin_memory", True))

    train_dataset = build_dataset(config, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    # Validation loader. Paper-faithful experiment configs require source validation
    # for model selection; debug configs may disable it explicitly.
    val_loader = None
    if require_validation:
        try:
            val_dataset = build_dataset(config, split="val")
        except (FileNotFoundError, KeyError) as exc:
            raise ValueError(
                "This experiment requires source validation for paper-faithful model "
                "selection, but the validation split could not be built."
            ) from exc
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        try:
            val_dataset = build_dataset(config, split="val")
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        except (FileNotFoundError, KeyError):
            val_loader = None

    # Build trainer and run.
    trainer = Trainer(
        model=model,
        projector=projector,
        teacher_ensemble=teacher_ensemble,
        config=config,
    )
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
