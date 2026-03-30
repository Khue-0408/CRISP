"""
Smoke tests for the U-Net Task A host path.
"""

from __future__ import annotations

from pathlib import Path

import torch

from crisp.engine.evaluator import Evaluator
from crisp.engine.trainer import Trainer
from crisp.models.projector_head import CRISPProjectorHead
from crisp.models.unet import UNet
from crisp.registry import build_model, build_projector, get_model_decoder_channels
from crisp.scripts.train import _maybe_build_teacher_ensemble, _maybe_initialize_student
from crisp.tests_support.toy_data import make_toy_batch


class _ConstantTeacherEnsemble(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        b, _, h, w = x.shape
        return [
            torch.full((b, 1, h, w), 0.3, device=x.device),
            torch.full((b, 1, h, w), 0.7, device=x.device),
        ]


def _unet_config() -> dict:
    return {
        "model": {
            "name": "unet",
            "in_channels": 3,
            "num_classes": 1,
            "base_channels": 8,
        },
        "crisp": {
            "projection": {"alpha_min": 0.5, "alpha_max": 1.8},
            "projector_head": {"hidden_channels": 16, "norm": "groupnorm"},
            "boundary": {"sigma_b": 3.0, "mode": "gaussian_soft_field"},
            "teacher": {"tau": 1.0, "gamma": 6.0, "strict": True},
            "solver": {"newton_steps": 3, "bisection_steps": 12},
            "warmup": {"enabled": False, "epochs": 15},
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "require_validation": False,
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
    }


def test_unet_build_exposes_decoder_contract() -> None:
    model = build_model(_unet_config())
    decoder_channels = get_model_decoder_channels(model)
    assert decoder_channels == 32

    batch = make_toy_batch(batch_size=2, image_size=32)
    out = model(batch["image"])
    assert out.logits.shape == (2, 1, 32, 32)
    assert out.features.shape == (2, decoder_channels, 8, 8)


def test_build_model_supports_class_path_for_optional_external_teachers() -> None:
    model = build_model(
        {
            "model": {
                "class_path": "crisp.models.unet.UNet",
                "in_channels": 3,
                "num_classes": 1,
                "base_channels": 8,
            }
        }
    )
    assert isinstance(model, UNet)


def test_unet_projector_build_matches_decoder_features() -> None:
    config = _unet_config()
    model = build_model(config)
    projector = build_projector(config, in_channels=get_model_decoder_channels(model))

    batch = make_toy_batch(batch_size=2, image_size=32)
    out = model(batch["image"])
    alpha_hat = projector(out.features, out.logits)
    assert alpha_hat.shape == out.logits.shape


def test_unet_student_init_checkpoint_loads(tmp_path: Path) -> None:
    reference = build_model(_unet_config())
    target = build_model(_unet_config())
    for param in target.parameters():
        param.data.zero_()

    checkpoint_path = tmp_path / "unet_init.pt"
    prefixed = {
        f"module.{key}": value.clone()
        for key, value in reference.state_dict().items()
    }
    torch.save({"model": prefixed}, checkpoint_path)

    _maybe_initialize_student(
        target,
        {
            "student_init": {
                "checkpoint": str(checkpoint_path),
                "strict": True,
                "state_dict_keys": ["model_state_dict", "state_dict", "model"],
                "prefixes_to_strip": ["module.", "model."],
            }
        },
    )

    for key, value in reference.state_dict().items():
        assert torch.allclose(target.state_dict()[key], value)


def test_unet_teacher_pool_loads_practical_two_teacher_setup(tmp_path: Path) -> None:
    pranet_ckpt = tmp_path / "pranet_teacher.pt"
    polyp_pvt_ckpt = tmp_path / "polyp_pvt_teacher.pt"

    torch.save({"state_dict": build_model({"model": {"name": "pranet"}}).state_dict()}, pranet_ckpt)
    torch.save({"model_state_dict": build_model({"model": {"name": "polyp_pvt"}}).state_dict()}, polyp_pvt_ckpt)

    cfg = {
        "crisp": {"teacher": {"strict": True}},
        "teachers": [
            {
                "enabled": True,
                "model": {"name": "pranet", "in_channels": 3, "num_classes": 1, "pretrained": False, "channel": 32},
                "checkpoint": str(pranet_ckpt),
                "checkpoint_loading": {
                    "state_dict_keys": ["model_state_dict", "state_dict", "model"],
                    "prefixes_to_strip": ["module.", "model."],
                },
            },
            {
                "enabled": True,
                "model": {"name": "polyp_pvt", "in_channels": 3, "num_classes": 1, "decoder_channels": 64},
                "checkpoint": str(polyp_pvt_ckpt),
                "checkpoint_loading": {
                    "state_dict_keys": ["model_state_dict", "state_dict", "model"],
                    "prefixes_to_strip": ["module.", "model."],
                },
            },
            {
                "enabled": False,
                "model": {"class_path": ""},
                "checkpoint": "",
            },
        ],
    }

    ensemble = _maybe_build_teacher_ensemble(cfg)
    assert ensemble is not None
    batch = make_toy_batch(batch_size=1, image_size=32)
    teacher_probs = ensemble(batch["image"])
    assert len(teacher_probs) == 2
    assert teacher_probs[0].shape == (1, 1, 32, 32)


def test_unet_crisp_one_training_step_smoke(tmp_path: Path) -> None:
    config = _unet_config()
    config["output_dir"] = str(tmp_path)
    config["seed"] = 0

    model = build_model(config)
    projector = build_projector(config, in_channels=get_model_decoder_channels(model))
    trainer = Trainer(
        model=model,
        projector=projector,
        teacher_ensemble=_ConstantTeacherEnsemble(),
        config=config,
    )

    batch = make_toy_batch(batch_size=2, image_size=32)
    output = trainer.train_one_step(batch, epoch=0, step=0)
    assert torch.isfinite(output.loss)
    assert output.logs["task_loss"] >= 0.0
    assert output.logs["amort_loss"] >= 0.0


def test_unet_evaluator_projector_on_off_smoke() -> None:
    config = _unet_config()
    model = build_model(config)
    projector = build_projector(config, in_channels=get_model_decoder_channels(model))
    evaluator = Evaluator(model=model, projector=projector, config=config)

    batch = make_toy_batch(batch_size=2, image_size=32)
    out_on = evaluator.predict_batch(batch, projector_on=True)
    out_off = evaluator.predict_batch(batch, projector_on=False)

    assert out_on["probs"].shape == (2, 1, 32, 32)
    assert torch.allclose(out_off["alpha_hat"], torch.ones_like(out_off["alpha_hat"]))
