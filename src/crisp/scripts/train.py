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

import torch
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

    teachers: list[FrozenTeacher] = []
    for t in teachers_cfg:
        if not isinstance(t, dict):
            continue
        model_cfg = t.get("model", None)
        ckpt = t.get("checkpoint", None)
        if not model_cfg or not ckpt:
            continue
        teacher_model = build_model({"model": model_cfg})
        teachers.append(
            FrozenTeacher(
                teacher_model,
                checkpoint_path=to_absolute_path(str(ckpt)),
            )
        )

    if not teachers:
        return None
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

    # Optional validation loader.
    val_loader = None
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
        pass  # No validation split available.

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
