"""Resolver for the author-approved BWCR calibration control."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from crisp.data.bwcr_views import BWCRAugmentationContract, current_crisp_bwcr_contract
from crisp.modules.bwcr_consistency import (
    BWCR_ALPHA,
    BWCR_LAMBDA_MAX,
    BWCR_LAMBDA_MIN,
    BWCR_RADIUS,
)


CONTROL_NAME = "boundary_weighted_logit_consistency"
BOUNDARY_WEIGHTING = "native_linear"
VIEW_GEOMETRY = "independent_inverse_aligned"
CONSISTENCY_SIGNAL = "raw_final_logit"


@dataclass(frozen=True)
class BWCRControl:
    """Resolved fixed scientific identity for source-training BWCR."""

    lambda_min: float
    lambda_max: float
    radius: float
    alpha: float
    boundary_weighting: str
    view_geometry: str
    consistency_signal: str
    augmentation: BWCRAugmentationContract


def _locked_number(control: Mapping[str, Any], key: str, expected: float) -> float:
    value = control.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"BWCR {key} must be numeric.")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved != expected:
        raise ValueError(f"BWCR locks {key}={expected}.")
    return resolved


def resolve_bwcr_control(config: Mapping[str, Any]) -> BWCRControl | None:
    """Validate the standalone BWCR control and return its locked parameters."""

    method = config.get("method", {})
    control = config.get("calibration_control")
    method_name = method.get("name") if isinstance(method, Mapping) else None
    control_name = control.get("name") if isinstance(control, Mapping) else None
    if method_name != CONTROL_NAME and control_name != CONTROL_NAME:
        return None
    if not isinstance(method, Mapping):
        raise ValueError("BWCR requires a method mapping.")
    if not isinstance(control, Mapping):
        raise ValueError("BWCR requires calibration_control parameters.")
    if method_name != CONTROL_NAME or control.get("name") != CONTROL_NAME:
        raise ValueError(
            "BWCR method and calibration_control identities must both be "
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
        raise ValueError("BWCR forbids CRISP-only machinery: " + ", ".join(enabled))
    if method.get("target_mode", "hard_label") != "hard_label":
        raise ValueError("BWCR requires hard-label host supervision.")

    configured_teachers = config.get("teachers")
    teacher_pool = config.get("teacher_pool")
    if isinstance(teacher_pool, Mapping):
        configured_teachers = configured_teachers or teacher_pool.get("teachers")
    if configured_teachers:
        raise ValueError("BWCR forbids configured teacher artifacts.")

    expected_strings = {
        "boundary_weighting": BOUNDARY_WEIGHTING,
        "view_geometry": VIEW_GEOMETRY,
        "consistency_signal": CONSISTENCY_SIGNAL,
    }
    for key, expected in expected_strings.items():
        if control.get(key) != expected:
            raise ValueError(f"BWCR locks {key}='{expected}'.")

    return BWCRControl(
        lambda_min=_locked_number(control, "lambda_min", BWCR_LAMBDA_MIN),
        lambda_max=_locked_number(control, "lambda_max", BWCR_LAMBDA_MAX),
        radius=_locked_number(control, "radius", BWCR_RADIUS),
        alpha=_locked_number(control, "alpha", BWCR_ALPHA),
        boundary_weighting=BOUNDARY_WEIGHTING,
        view_geometry=VIEW_GEOMETRY,
        consistency_signal=CONSISTENCY_SIGNAL,
        augmentation=current_crisp_bwcr_contract(config),
    )
