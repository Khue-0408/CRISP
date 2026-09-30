"""
Detached local projection solver for CRISP.

The CRISP local projection problem finds alpha*(u) in [alpha_min, alpha_max]
that best fits the target t*(u) through the restricted calibrated family
sigma(alpha * z(u)), while preserving identity away from the boundary.

The paper states that after mild numerical stabilization this problem is
one-dimensional, strongly convex, and can be solved with a clipped closed-form
seed followed by safeguarded Newton or short bisection refinement.

The scientific definitions are governed by the current CRISP manuscript.
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
        Maximum absolute clipping value before stabilization (current protocol
        value 8.0).
    zeta:
        Minimum absolute magnitude enforced after clipping (current protocol
        value 0.1).

    Returns
    -------
    torch.Tensor
        Stabilized detached logits z̃(u), same shape.

    CRISP reference
    ---------------
    Current CRISP projection contract:
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
    Current CRISP contract: alpha0 = clamp(logit(t_eps) / z̃, alpha_min, alpha_max).
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
    Current CRISP solver contract:
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


def _safeguarded_newton_step(
    alpha: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    stabilized_logits: torch.Tensor,
    clipped_target: torch.Tensor,
    boundary_weight: torch.Tensor,
    lambda_value: float,
    mu_value: float,
    active: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use Newton only when its proposal is finite, bracketed, and improves |g|."""
    g_current = projection_gradient(
        alpha, stabilized_logits, clipped_target, boundary_weight,
        lambda_value, mu_value,
    )
    h_current = projection_hessian(
        alpha, stabilized_logits, boundary_weight, lambda_value, mu_value,
    ).clamp(min=1e-8)
    proposal = alpha - g_current / h_current
    finite = torch.isfinite(proposal)
    outside = active & finite & ((proposal <= lo) | (proposal >= hi))
    g_proposal = projection_gradient(
        proposal, stabilized_logits, clipped_target, boundary_weight,
        lambda_value, mu_value,
    )
    no_decrease = (
        active & finite & ~outside
        & (~torch.isfinite(g_proposal) | (g_proposal.abs() >= g_current.abs()))
    )
    accepted = active & finite & ~outside & ~no_decrease
    candidate = torch.where(accepted, proposal, 0.5 * (lo + hi))
    candidate = torch.where(active, candidate, alpha)
    invalid = active & ~finite
    return candidate, accepted, outside, no_decrease, invalid


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
    Current CRISP contract: alpha*(u) = argmin_{α∈[α_min,alpha_max]} L_proj(α).
    The solver uses safeguarded Newton plus bisection, and alpha_star is detached.
    """
    if alpha_min >= alpha_max:
        raise ValueError(
            f"Expected alpha_min < alpha_max, got {alpha_min} >= {alpha_max}."
        )

    # --- Step 1: Stabilize detached inputs and evaluate KKT endpoints. ---
    z_tilde = stabilize_logits_for_solver(logits, zmax=zmax, zeta=zeta)
    clipped_target = clipped_target.detach()
    boundary_weight = boundary_weight.detach()
    alpha_lo = torch.full_like(z_tilde, alpha_min)
    alpha_hi = torch.full_like(z_tilde, alpha_max)
    g_lo = projection_gradient(
        alpha_lo, z_tilde, clipped_target, boundary_weight,
        lambda_value, mu_value,
    )
    g_hi = projection_gradient(
        alpha_hi, z_tilde, clipped_target, boundary_weight,
        lambda_value, mu_value,
    )
    choose_lo = g_lo >= 0
    choose_hi = ~choose_lo & (g_hi <= 0)
    interior = ~choose_lo & ~choose_hi
    alpha = torch.where(choose_lo, alpha_lo, alpha_hi)
    lo = alpha_lo.clone()
    hi = alpha_hi.clone()

    # --- Step 2: Seed and bracket only the interior roots. ---
    if interior.any():
        seed = closed_form_seed(z_tilde, clipped_target, alpha_min, alpha_max)
        alpha = torch.where(interior, seed, alpha)
        g_seed = projection_gradient(
            alpha, z_tilde, clipped_target, boundary_weight,
            lambda_value, mu_value,
        )
        lo = torch.where(interior & (g_seed < 0), alpha, lo)
        hi = torch.where(interior & (g_seed > 0), alpha, hi)

    residual_tol = 1e-4
    newton_invalid_count = torch.zeros_like(alpha)
    newton_outside_count = torch.zeros_like(alpha)
    newton_no_decrease_count = torch.zeros_like(alpha)
    newton_accepted_count = torch.zeros_like(alpha)
    newton_fallback_count = torch.zeros_like(alpha)
    bisection_count = torch.zeros_like(alpha, dtype=torch.int32)
    used_bisection = torch.zeros_like(interior)

    # --- Step 3: Safeguarded Newton attempts, bisecting rejected proposals. ---
    for _ in range(newton_steps):
        g_current = projection_gradient(
            alpha, z_tilde, clipped_target, boundary_weight,
            lambda_value, mu_value,
        )
        active = interior & (g_current.abs() > residual_tol)
        if not active.any():
            break
        candidate, accepted, outside, no_decrease, invalid = _safeguarded_newton_step(
            alpha, lo, hi, z_tilde, clipped_target, boundary_weight,
            lambda_value, mu_value, active,
        )
        fallback = active & ~accepted & (bisection_count < bisection_steps)
        candidate = torch.where(accepted | fallback, candidate, alpha)
        g_candidate = projection_gradient(
            candidate, z_tilde, clipped_target, boundary_weight,
            lambda_value, mu_value,
        )
        newton_invalid_count += invalid.float()
        newton_outside_count += outside.float()
        newton_no_decrease_count += no_decrease.float()
        newton_accepted_count += accepted.float()
        newton_fallback_count += fallback.float()
        bisection_count += fallback.to(bisection_count.dtype)
        used_bisection |= fallback
        updated = accepted | fallback
        lo = torch.where(updated & (g_candidate < 0), candidate, lo)
        hi = torch.where(updated & (g_candidate > 0), candidate, hi)
        alpha = torch.where(updated, candidate, alpha)

    # --- Step 4: At most bisection_steps further bracket refinements. ---
    for _ in range(bisection_steps):
        g_current = projection_gradient(
            alpha, z_tilde, clipped_target, boundary_weight,
            lambda_value, mu_value,
        )
        # Once Newton has fallen back, finish bracket refinement even if a
        # small derivative masks a wider alpha error under low curvature.
        active = (
            interior & (bisection_count < bisection_steps)
            & ((g_current.abs() > residual_tol) | (used_bisection & (g_current != 0)))
        )
        if not active.any():
            break
        mid = 0.5 * (lo + hi)
        g_mid = projection_gradient(
            mid, z_tilde, clipped_target, boundary_weight,
            lambda_value, mu_value,
        )
        lo = torch.where(active & (g_mid < 0), mid, lo)
        hi = torch.where(active & (g_mid > 0), mid, hi)
        alpha = torch.where(active, mid, alpha)
        bisection_count += active.to(bisection_count.dtype)
        used_bisection |= active

    # --- Step 5: Final clamp and detach ---
    alpha_star = alpha.clamp(alpha_min, alpha_max).detach()
    if (alpha_star < alpha_min - 1e-6).any() or (alpha_star > alpha_max + 1e-6).any():
        raise RuntimeError("solve_alpha_star returned values outside the feasible interval.")

    # --- Diagnostics ---
    diagnostics: Dict[str, torch.Tensor] = {
        "sat_lo": (alpha_star <= alpha_min + 1e-6).float().mean(),
        "sat_hi": (alpha_star >= alpha_max - 1e-6).float().mean(),
        "newton_invalid": newton_invalid_count.sum(),
        "newton_outside_bracket": newton_outside_count.sum(),
        "newton_no_residual_decrease": newton_no_decrease_count.sum(),
        "newton_accepted": newton_accepted_count.sum(),
        "newton_accepted_max": newton_accepted_count.max(),
        "newton_fallback": newton_fallback_count.sum(),
        "bisection_updates_max": bisection_count.max(),
        "bisection_pixels": used_bisection.float().mean(),
        "bracket_rate": interior.float().mean(),
    }

    return alpha_star, diagnostics
