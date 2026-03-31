"""
Local TrainDataset/TestDataset mode smoke tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from crisp.data.datasets import discover_local_test_datasets
from crisp.engine.evaluator import Evaluator
from crisp.engine.trainer import Trainer
from crisp.registry import build_dataset, build_model, build_projector, get_model_decoder_channels
from crisp.scripts.bootstrap_local_workspace import bootstrap_local_workspace


class _ConstantTeacherEnsemble(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        b, _, h, w = x.shape
        return [
            torch.full((b, 1, h, w), 0.35, device=x.device),
            torch.full((b, 1, h, w), 0.65, device=x.device),
        ]


def _write_rgb(path: Path, value: int) -> None:
    arr = np.full((16, 16, 3), value, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _write_mask(path: Path, value: int) -> None:
    arr = np.zeros((16, 16), dtype=np.uint8)
    arr[4:12, 4:12] = value
    Image.fromarray(arr, mode="L").save(path)


def _make_local_dataset_tree(root: Path) -> None:
    train_img = root / "data" / "TrainDataset" / "image"
    train_mask = root / "data" / "TrainDataset" / "mask"
    train_img.mkdir(parents=True, exist_ok=True)
    train_mask.mkdir(parents=True, exist_ok=True)

    for idx in range(10):
        _write_rgb(train_img / f"{idx}.png", 10 + idx)
        _write_mask(train_mask / f"{idx}.png", 255)

    test_specs = {
        "CVC-ColonDB": ("images", "masks"),
        "ETIS-LaribPolypDB": ("image", "mask"),
    }
    for name, (image_dir, mask_dir) in test_specs.items():
        img_root = root / "data" / "TestDataset" / name / image_dir
        mask_root = root / "data" / "TestDataset" / name / mask_dir
        img_root.mkdir(parents=True, exist_ok=True)
        mask_root.mkdir(parents=True, exist_ok=True)
        for idx in range(2):
            _write_rgb(img_root / f"{idx}.png", 30 + idx)
            _write_mask(mask_root / f"{idx}.png", 255)


def _local_config(root: Path) -> dict:
    return {
        "seed": 7,
        "model": {
            "name": "unet",
            "in_channels": 3,
            "num_classes": 1,
            "base_channels": 8,
        },
        "workspace": {
            "root": str(root),
            "auto_create": True,
        },
        "source_data": {
            "name": "local_train_test",
            "mode": "local_train_test",
            "root": str(root),
            "train_dir": "TrainDataset",
            "test_dir": "TestDataset",
            "image_dir": "image",
            "mask_dir": "mask",
            "train_image_dir": "image",
            "train_mask_dir": "mask",
            "train_image_dir_candidates": ["image", "images"],
            "train_mask_dir_candidates": ["mask", "masks"],
            "test_image_dir_candidates": ["image", "images"],
            "test_mask_dir_candidates": ["mask", "masks"],
            "image_size": 32,
            "random_hflip": False,
            "random_vflip": False,
            "num_workers": 0,
            "pin_memory": False,
            "strict_pairing": True,
            "local_split": {
                "val_fraction": 0.2,
                "cache_dir": str(root / "metadata" / "splits" / "local_train_test"),
            },
        },
        "crisp": {
            "projection": {
                "lambda": 0.8,
                "mu": 0.05,
                "beta": 0.2,
                "eta_dice": 0.5,
                "alpha_min": 0.5,
                "alpha_max": 1.8,
                "eps_target": 1.0e-4,
                "zeta": 1.0e-2,
                "zmax": 12.0,
            },
            "projector_head": {"hidden_channels": 16, "norm": "groupnorm"},
            "boundary": {"sigma_b": 3.0, "mode": "gaussian_soft_field"},
            "teacher": {"tau": 1.0, "gamma": 6.0, "strict": True},
            "solver": {"newton_steps": 3, "bisection_steps": 12},
            "warmup": {"enabled": False, "epochs": 15},
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "require_validation": True,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "lr_student": 1.0e-4,
            "lr_projector": 2.0e-4,
            "weight_decay": 1.0e-4,
            "mixed_precision": False,
            "gradient_clip_norm": 0.0,
        },
        "method": {
            "use_crisp": True,
            "use_projector": True,
            "use_teachers": True,
            "target_mode": "boundary_posterior",
            "use_boundary_weighted_task": True,
            "use_identity_regularization": True,
            "use_amortization_loss": True,
            "allow_self_ensemble_teacher": False,
        },
        "eval": {
            "auto_discover_local_test_datasets": True,
            "boundary_support": {"top_percent": 20.0},
            "ece": {"bins": 15},
            "tace": {"threshold": 1.0e-3},
        },
        "eval_datasets": [],
    }


def test_local_train_dataset_split_is_deterministic_and_cached(tmp_path: Path) -> None:
    _make_local_dataset_tree(tmp_path)
    cfg = _local_config(tmp_path)

    train_dataset = build_dataset(cfg, split="train")
    val_dataset = build_dataset(cfg, split="val")

    assert len(train_dataset) == 8
    assert len(val_dataset) == 2

    split_dir = tmp_path / "metadata" / "splits" / "local_train_test" / "local_train_test_seed_7_val_200"
    assert (split_dir / "train.txt").exists()
    assert (split_dir / "val.txt").exists()

    train_ids_first = [sample.image_id for sample in train_dataset.samples]
    val_ids_first = [sample.image_id for sample in val_dataset.samples]
    train_ids_second = [sample.image_id for sample in build_dataset(cfg, split="train").samples]
    val_ids_second = [sample.image_id for sample in build_dataset(cfg, split="val").samples]

    assert train_ids_first == train_ids_second
    assert val_ids_first == val_ids_second


def test_local_test_dataset_discovery_supports_immediate_subfolders(tmp_path: Path) -> None:
    _make_local_dataset_tree(tmp_path)
    cfg = _local_config(tmp_path)

    discovered = discover_local_test_datasets(cfg["source_data"])
    assert sorted(discovered.keys()) == ["CVC-ColonDB", "ETIS-LaribPolypDB"]
    assert discovered["CVC-ColonDB"]["image_dir"] == "images"
    assert discovered["ETIS-LaribPolypDB"]["mask_dir"] == "mask"


def test_bootstrap_local_workspace_creates_expected_directories(tmp_path: Path) -> None:
    layout = bootstrap_local_workspace(tmp_path)
    assert layout["student_unet_init_root"].is_dir()
    assert layout["teacher_pranet_root"].is_dir()
    assert layout["metadata_splits_root"].is_dir()
    assert (tmp_path / "TrainDataset" / "image").is_dir()
    assert (tmp_path / "TrainDataset" / "mask").is_dir()
    assert (tmp_path / "TestDataset").is_dir()


def test_local_unet_crisp_train_and_eval_smoke(tmp_path: Path) -> None:
    _make_local_dataset_tree(tmp_path)
    cfg = _local_config(tmp_path)

    model = build_model(cfg)
    projector = build_projector(cfg, in_channels=get_model_decoder_channels(model))
    trainer = Trainer(
        model=model,
        projector=projector,
        teacher_ensemble=_ConstantTeacherEnsemble(),
        config=cfg,
    )

    train_loader = DataLoader(build_dataset(cfg, split="train"), batch_size=2, shuffle=False)
    batch = next(iter(train_loader))
    step_output = trainer.train_one_step(batch, epoch=0, step=0)
    assert torch.isfinite(step_output.loss)

    evaluator = Evaluator(model=model, projector=projector, config=cfg)
    discovered = discover_local_test_datasets(cfg["source_data"])
    for dataset_name, dataset_cfg in discovered.items():
        dataset = build_dataset({"source_data": dataset_cfg, "seed": cfg["seed"]}, split="test")
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        metrics_on = evaluator.evaluate_dataset(loader, dataset_name, projector_on=True)
        metrics_off = evaluator.evaluate_dataset(loader, dataset_name, projector_on=False)
        assert "dice" in metrics_on
        assert "ece" in metrics_off
