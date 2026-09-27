"""Author-approved Margin Label Smoothing control for binary segmentation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F


CONTROL_NAME = "margin_label_smoothing"
LOCKED_MARGIN = 10.0
LOCKED_WEIGHT = 0.1


@dataclass(frozen=True)
class MarginLabelSmoothingControl:
    """Resolved fixed parameters for the standalone Margin-LS control."""

    margin: float
    weight: float


def margin_label_smoothing_penalty(
    logits: torch.Tensor,
    margin: float = LOCKED_MARGIN,
) -> torch.Tensor:
    """Mean binary MbLS penalty ``relu(abs(z) - margin)`` over batch and pixels."""
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("Margin Label Smoothing requires a finite margin > 0.")
    return F.relu(logits.abs() - margin).mean()


def resolve_margin_label_smoothing_control(
    config: Mapping[str, Any],
) -> MarginLabelSmoothingControl | None:
    """Validate and resolve the fixed standalone control, or return ``None``."""
    method = config.get("method", {})
    control = config.get("calibration_control")
    method_name = method.get("name") if isinstance(method, Mapping) else None
    control_name = control.get("name") if isinstance(control, Mapping) else None

    if method_name != CONTROL_NAME and control_name != CONTROL_NAME:
        return None
    if not isinstance(method, Mapping):
        raise ValueError("Margin Label Smoothing requires a method mapping.")
    if not isinstance(control, Mapping):
        raise ValueError("Margin Label Smoothing requires calibration_control parameters.")
    if method_name != CONTROL_NAME or control.get("name") != CONTROL_NAME:
        raise ValueError(
            "Margin Label Smoothing method and calibration_control identities must both be "
            f"'{CONTROL_NAME}'."
        )

    forbidden_flags = (
        "use_crisp",
        "use_projector",
        "use_teachers",
        "use_amortization_loss",
        "use_boundary_weighted_task",
        "use_identity_regularization",
        "allow_self_ensemble_teacher",
    )
    enabled = [key for key in forbidden_flags if bool(method.get(key, False))]
    if enabled:
        raise ValueError(
            "Margin Label Smoothing forbids CRISP-only machinery: " + ", ".join(enabled)
        )
    if method.get("target_mode", "hard_label") != "hard_label":
        raise ValueError("Margin Label Smoothing requires hard-label baseline targets.")
    configured_teachers = config.get("teachers")
    teacher_pool = config.get("teacher_pool")
    if isinstance(teacher_pool, Mapping):
        configured_teachers = configured_teachers or teacher_pool.get("teachers")
    if configured_teachers:
        raise ValueError("Margin Label Smoothing forbids configured teacher artifacts.")

    margin_value = control.get("margin")
    weight_value = control.get("weight")
    if isinstance(margin_value, bool) or not isinstance(margin_value, (int, float)):
        raise ValueError("Margin Label Smoothing margin must be numeric.")
    if isinstance(weight_value, bool) or not isinstance(weight_value, (int, float)):
        raise ValueError("Margin Label Smoothing weight must be numeric.")
    margin = float(margin_value)
    weight = float(weight_value)
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("Margin Label Smoothing requires a finite margin > 0.")
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("Margin Label Smoothing requires a finite weight >= 0.")
    if margin != LOCKED_MARGIN or weight != LOCKED_WEIGHT:
        raise ValueError(
            "The current CRISP Margin Label Smoothing control locks margin=10.0 and weight=0.1."
        )
    return MarginLabelSmoothingControl(margin=margin, weight=weight)
