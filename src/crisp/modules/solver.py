"""
Detached local projection solver for CRISP.

The CRISP local projection problem finds alpha*(u) in [alpha_min, alpha_max]
that best fits the target t*(u) through the restricted calibrated family
sigma(alpha * z(u)), while preserving identity away from the boundary. [file:1]

The paper states that after mild numerical stabilization this problem is
one-dimensional, strongly convex, and can be solved with a clipped closed-form
seed followed by safeguarded Newton or short bisection refinement. [file:1]

CRISP reference: instruct.md §7, §8, §9.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from crisp.utils.tensor_ops import safe_logit


def stabilize_logits_for_solver(
    logits: torch.Tensor,
    zmax: float,
    zeta: float,
) -> torch.Tensor:
    """
    Construct the detached stabilized logit z̃(u) used only by the local solver.

    Parameters
    ----------
    logits:
        Raw student logits z(u) of shape [B, 1, H, W].
    zmax:
        Maximum absolute clipping value before stabilization (default 12.0).
    zeta:
        Minimum absolute magnitude enforced after clipping (default 1e-2).

    Returns
    -------
    torch.Tensor
        Stabilized detached logits z̃(u), same shape.

    CRISP reference
    ---------------
    instruct.md §8:
      z_clip(u) = clip(z(u), -Z_max, Z_max)
      z̃(u) = sign(z_clip) · max(|z_clip|, ζ)
    z̃ is used only in the local solver. The forward path still uses raw z.
    alpha_star is treated with stop-gradient.
    """
    if zmax <= 0:
        raise ValueError(f"zmax must be positive, got {zmax}.")
    if zeta <= 0:
        raise ValueError(f"zeta must be positive, got {zeta}.")

    z_detached = logits.detach()
    z_clip = z_detached.clamp(-zmax, zmax)

    # Preserve the sign of non-zero logits while ensuring |z_tilde| >= zeta.
    # For exact zeros, choose the positive branch so the stabilized solver never
    # sees a degenerate zero denominator in the closed-form seed.
    sign = torch.where(z_clip < 0.0, -torch.ones_like(z_clip), torch.ones_like(z_clip))
    z_tilde = sign * torch.clamp(z_clip.abs(), min=zeta)
    return z_tilde


def closed_form_seed(
    stabilized_logits: torch.Tensor,
    clipped_target: torch.Tensor,
    alpha_min: float,
    alpha_max: float,
) -> torch.Tensor:
    """
    Compute the clipped no-regularization seed for alpha*.

    This seed corresponds to the closed-form solution obtained when the identity
    regularization coefficient mu is zero, then projected onto the valid interval.

    CRISP reference
    ---------------
    instruct.md §9: alpha0 = clamp(logit(t_eps) / z̃, alpha_min, alpha_max).
    """
    logit_t = safe_logit(clipped_target)  # logit(t_eps)
    alpha_seed = logit_t / stabilized_logits  # element-wise
    return alpha_seed.clamp(alpha_min, alpha_max)


def projection_gradient(
    alpha: torch.Tensor,
    stabilized_logits: torch.Tensor,
    clipped_target: torch.Tensor,
    boundary_weight: torch.Tensor,
    lambda_value: float,
    mu_value: float,
) -> torch.Tensor:
    """
    Evaluate the first derivative g(alpha; z̃, t, w) of the stabilized local objective.

    Parameters
    ----------
    alpha:
        Current inverse-temperature estimate [B,1,H,W].
    stabilized_logits:
        Detached stabilized logits z̃ [B,1,H,W].
    clipped_target:
        Clipped posterior target t_eps [B,1,H,W].
    boundary_weight:
        Boundary weighting field w_b [B,1,H,W].
    lambda_value:
        Projection mixing coefficient λ.
    mu_value:
        Off-boundary identity regularization coefficient μ.

    Returns
    -------
    torch.Tensor
        Per-pixel derivative values [B,1,H,W].

    CRISP reference
    ---------------
    instruct.md §9:
      g(α; z̃, t, w) = (1 + λw)(σ(αz̃) - t)z̃ + 2μ(1-w)(α - 1)
    """
    sig = torch.sigmoid(alpha * stabilized_logits)
    fit_term = (1.0 + lambda_value * boundary_weight) * (sig - clipped_target) * stabilized_logits
    reg_term = 2.0 * mu_value * (1.0 - boundary_weight) * (alpha - 1.0)
    return fit_term + reg_term


def projection_hessian(
    alpha: torch.Tensor,
    stabilized_logits: torch.Tensor,
    boundary_weight: torch.Tensor,
    lambda_value: float,
    mu_value: float,
) -> torch.Tensor:
    """
    Evaluate the second derivative of the stabilized local objective.

    Returns
    -------
    torch.Tensor
        Positive per-pixel curvature values for Newton updates [B,1,H,W].

    CRISP reference
    ---------------
    Derived from gradient in §9:
      g'(α) = (1 + λw) σ(αz̃)(1-σ(αz̃)) z̃² + 2μ(1-w)
    This is always positive because σ(1-σ)z̃² ≥ 0 and μ(1-w) ≥ 0.
    """
    sig = torch.sigmoid(alpha * stabilized_logits)
    fit_curvature = (
        (1.0 + lambda_value * boundary_weight)
        * sig * (1.0 - sig)
        * stabilized_logits.pow(2)
    )
    reg_curvature = 2.0 * mu_value * (1.0 - boundary_weight)
    return fit_curvature + reg_curvature


def solve_alpha_star(
    logits: torch.Tensor,
    clipped_target: torch.Tensor,
    boundary_weight: torch.Tensor,
    lambda_value: float,
    mu_value: float,
    alpha_min: float,
    alpha_max: float,
    zmax: float,
    zeta: float,
    newton_steps: int = 3,
    bisection_steps: int = 12,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Solve the detached local CRISP projection problem for alpha*(u).

    Parameters
    ----------
    logits:
        Raw student logits z(u) [B,1,H,W].
    clipped_target:
        Clipped boundary posterior target t*_eps(u) [B,1,H,W].
    boundary_weight:
        Boundary weighting field w_b(u) [B,1,H,W].
    lambda_value:
        Projection coefficient λ.
    mu_value:
        Identity regularization coefficient μ.
    alpha_min:
        Lower feasible bound for α.
    alpha_max:
        Upper feasible bound for α.
    zmax:
        Maximum absolute logit clipping magnitude.
    zeta:
        Minimum absolute stabilized logit magnitude.
    newton_steps:
        Number of safeguarded Newton refinement steps.
    bisection_steps:
        Number of bisection fallback steps.

    Returns
    -------
    Tuple[torch.Tensor, Dict[str, torch.Tensor]]
        A tuple containing:
        - alpha_star: detached local optimum field [B,1,H,W],
        - diagnostics: dictionary with solver diagnostics.

    CRISP reference
    ---------------
    instruct.md §7: alpha*(u) = argmin_{α∈[α_min,α_max]} L_proj(α).
    instruct.md §9: safeguarded Newton + bisection.
    instruct.md §13: alpha_star must be detached.
    """
    if alpha_min >= alpha_max:
        raise ValueError(
            f"Expected alpha_min < alpha_max, got {alpha_min} >= {alpha_max}."
        )

    # --- Step 1: Stabilize logits (detached) ---
    z_tilde = stabilize_logits_for_solver(logits, zmax=zmax, zeta=zeta)

    # --- Step 2: Closed-form seed ---
    alpha = closed_form_seed(z_tilde, clipped_target, alpha_min, alpha_max)

    # Track diagnostics.
    newton_invalid_count = torch.zeros_like(alpha)

    # --- Step 3: Safeguarded Newton steps ---
    for _ in range(newton_steps):
        g = projection_gradient(
            alpha, z_tilde, clipped_target, boundary_weight,
            lambda_value, mu_value,
        )
        h = projection_hessian(
            alpha, z_tilde, boundary_weight,
            lambda_value, mu_value,
        )
        # Safeguard: ensure hessian is positive.
        h_safe = h.clamp(min=1e-8)
        newton_update = g / h_safe
        alpha_prop = alpha - newton_update

        # Check validity before clamping: NaN/Inf proposals or NaN/Inf updates.
        invalid = (
            torch.isnan(alpha_prop)
            | torch.isinf(alpha_prop)
            | torch.isnan(newton_update)
            | torch.isinf(newton_update)
        )
        newton_invalid_count += invalid.float()

        # Clamp into the feasible interval (Newton is safeguarded by projection).
        alpha_new = alpha_prop.clamp(alpha_min, alpha_max)
        # Keep old alpha where invalid.
        alpha = torch.where(invalid, alpha, alpha_new)

    # --- Step 4: Bisection refinement (with bound-optimum handling) ---
    # CRISP local objective is strongly convex after stabilization (§7-§9), so
    # g(α) is monotone increasing. The unique minimizer is:
    # - interior root if g(α_min) <= 0 <= g(α_max),
    # - otherwise the nearer bound (α_min if g(α_min) > 0, α_max if g(α_max) < 0).
    g_lo = projection_gradient(
        torch.full_like(alpha, alpha_min),
        z_tilde, clipped_target, boundary_weight,
        lambda_value, mu_value,
    )
    g_hi = projection_gradient(
        torch.full_like(alpha, alpha_max),
        z_tilde, clipped_target, boundary_weight,
        lambda_value, mu_value,
    )

    bracketed = (g_lo <= 0) & (g_hi >= 0)
    choose_lo = g_lo > 0
    choose_hi = g_hi < 0

    # Start from current alpha, then overwrite where we know the bound optimum.
    alpha = torch.where(choose_lo, torch.full_like(alpha, alpha_min), alpha)
    alpha = torch.where(choose_hi, torch.full_like(alpha, alpha_max), alpha)

    # Only bisect where the root is bracketed and the current gradient is not small.
    g_current = projection_gradient(
        alpha, z_tilde, clipped_target, boundary_weight,
        lambda_value, mu_value,
    ).abs()
    needs_bisection = bracketed & (g_current > 1e-4)

    if needs_bisection.any():
        bis_lo = torch.full_like(alpha, alpha_min)
        bis_hi = torch.full_like(alpha, alpha_max)

        for _ in range(bisection_steps):
            mid = 0.5 * (bis_lo + bis_hi)
            g_mid = projection_gradient(
                mid, z_tilde, clipped_target, boundary_weight,
                lambda_value, mu_value,
            )
            # Monotone g: if g(mid) > 0, root lies left; else right.
            move_hi = g_mid > 0
            bis_hi = torch.where(move_hi, mid, bis_hi)
            bis_lo = torch.where(move_hi, bis_lo, mid)

        bis_result = 0.5 * (bis_lo + bis_hi)
        alpha = torch.where(needs_bisection, bis_result, alpha)

    # --- Step 5: Final clamp and detach ---
    alpha_star = alpha.clamp(alpha_min, alpha_max).detach()
    if (alpha_star < alpha_min - 1e-6).any() or (alpha_star > alpha_max + 1e-6).any():
        raise RuntimeError("solve_alpha_star returned values outside the feasible interval.")

    # --- Diagnostics ---
    diagnostics: Dict[str, torch.Tensor] = {
        "sat_lo": (alpha_star <= alpha_min + 1e-6).float().mean(),
        "sat_hi": (alpha_star >= alpha_max - 1e-6).float().mean(),
        "newton_invalid": newton_invalid_count.sum(),
        "bisection_pixels": needs_bisection.float().mean()
        if needs_bisection.any()
        else torch.tensor(0.0, device=alpha_star.device),
        "bracket_rate": bracketed.float().mean(),
    }

    return alpha_star, diagnostics
