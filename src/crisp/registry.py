"""
Central lightweight registry utilities.

This module exposes helper functions that map config names
to concrete dataset builders, model builders, loss builders,
and evaluator factories.

The goal is to keep the experiment entry points clean while
avoiding hard-coded conditionals scattered across the codebase.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict

from crisp.data.datasets import build_binary_segmentation_dataset
from crisp.data.transforms import build_eval_transforms, build_train_transforms
from crisp.models.projector_head import CRISPProjectorHead


_MODEL_REGISTRY: Dict[str, type] = {}


def _ensure_registry() -> None:
    """Lazy-populate the model registry on first access."""
    if _MODEL_REGISTRY:
        return
    from crisp.models.pranet import PraNet
    from crisp.models.unet import UNet
    from crisp.models.polyp_pvt import PolypPVT
    from crisp.models.rabbit import RaBiT

    _MODEL_REGISTRY.update({
        "pranet": PraNet,
        "unet": UNet,
        "polyp_pvt": PolypPVT,
        "rabbit": RaBiT,
    })


def build_model(config: Dict[str, Any]) -> Any:
    """
    Build and return a segmentation backbone from a configuration object.

    Parameters
    ----------
    config:
        Configuration dictionary or OmegaConf node describing the model.
        Expected key: ``model.name`` → one of {pranet, unet, polyp_pvt, rabbit}.

    Returns
    -------
    Any
        Instantiated model object.
    """
    _ensure_registry()
    model_cfg = dict(config.get("model", config))
    name = str(model_cfg.pop("name", "pranet")).lower()

    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}"
        )

    cls = _MODEL_REGISTRY[name]
    aliases = {
        "feature_channels": "channel",
    }
    normalized_cfg = {
        aliases.get(key, key): value for key, value in model_cfg.items()
    }
    signature = inspect.signature(cls.__init__)
    kwargs: Dict[str, Any] = {
        key: value
        for key, value in normalized_cfg.items()
        if key in signature.parameters
    }

    return cls(**kwargs)


def build_projector(config: Dict[str, Any], in_channels: int) -> Any:
    """
    Build the CRISP amortized projector head.

    Parameters
    ----------
    config:
        CRISP configuration block.
    in_channels:
        Number of feature channels entering the projector head.

    Returns
    -------
    CRISPProjectorHead
        Instantiated projector head module.
    """
    crisp_cfg = config.get("crisp", config)
    proj_cfg = crisp_cfg.get("projector_head", {})
    alpha_cfg = crisp_cfg.get("projection", {})

    return CRISPProjectorHead(
        feature_channels=in_channels,
        hidden_channels=proj_cfg.get("hidden_channels", 64),
        alpha_min=alpha_cfg.get("alpha_min", 0.50),
        alpha_max=alpha_cfg.get("alpha_max", 1.80),
        norm=proj_cfg.get("norm", "groupnorm"),
    )


def build_dataset(config: Dict[str, Any], split: str) -> Any:
    """
    Build a dataset object for a specific split.

    Parameters
    ----------
    config:
        Dataset configuration block.  Expected keys:
        ``root``, ``image_dir``, ``mask_dir``, ``name``, ``image_size``.
    split:
        One of 'train', 'val', or 'test'.

    Returns
    -------
    BinarySegmentationDataset
        Instantiated dataset object.
    """
    data_cfg = dict(config.get("source_data", config))
    split_cfg = dict(data_cfg.get("splits", {}).get(split, {}))
    merged_cfg = {**data_cfg, **split_cfg}

    if split == "train":
        transforms = build_train_transforms(merged_cfg)
    else:
        transforms = build_eval_transforms(merged_cfg)

    return build_binary_segmentation_dataset(
        root=merged_cfg.get("root", "data"),
        image_dir=merged_cfg.get("image_dir", "images"),
        mask_dir=merged_cfg.get("mask_dir", "masks"),
        split=split,
        dataset_name=merged_cfg.get("name", data_cfg.get("name", "unknown")),
        transforms=transforms,
        split_file=merged_cfg.get("split_file"),
    )
