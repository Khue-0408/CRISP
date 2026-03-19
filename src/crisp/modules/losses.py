"""
Loss functions for baseline and CRISP training.

CRISP training combines:
- projected task fitting over the calibrated probability p̃(u),
- off-boundary identity regularization on alpha_hat,
- Dice loss on the calibrated output,
- amortization consistency between alpha_hat and detached alpha*. [file:1]

CRISP reference: instruct.md §12.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from crisp.modules.solver import stabilize_logits_for_solver


def dice_loss(
    probs: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """
    Compute soft Dice loss between predicted probabilities and a binary mask.

    Parameters
    ----------
    probs:
        Predicted probabilities [B, 1, H, W].
    target:
        Binary ground-truth mask [B, 1, H, W].
    eps:
        Smoothing constant for numerical stability.

    Returns
    -------
    torch.Tensor
        Scalar Dice loss.

    CRISP reference
    ---------------
    instruct.md §12:
      L_Dice(p̃, y) = 1 - (2 Σ_u p̃(u) y(u) + ε_d) / (Σ_u p̃(u) + Σ_u y(u) + ε_d)
    """
    # Flatten spatial dimensions for numerically stable summation.
    p_flat = probs.reshape(probs.shape[0], -1)       # [B, N]
    t_flat = target.reshape(target.shape[0], -1)      # [B, N]

    intersection = (p_flat * t_flat).sum(dim=1)       # [B]
    cardinality = p_flat.sum(dim=1) + t_flat.sum(dim=1)  # [B]

    dice = (2.0 * intersection + eps) / (cardinality + eps)  # [B]
    return (1.0 - dice).mean()  # scalar


def baseline_bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    dice_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Compute the standard BCE + Dice loss used for baseline segmentation training.

    This function exists so CRISP and non-CRISP baselines can share a common
    training engine with only method-specific branches where necessary.

    Parameters
    ----------
    logits:
        Raw foreground logits [B, 1, H, W].
    target:
        Binary ground-truth mask [B, 1, H, W].
    dice_weight:
        Weight applied to the Dice loss term.

    Returns
    -------
    Dict[str, torch.Tensor]
        ``{'loss': ..., 'bce': ..., 'dice': ...}``
    """
    bce = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="mean")
    probs = torch.sigmoid(logits)
    dl = dice_loss(probs, target.float())
    total = bce + dice_weight * dl
    return {"loss": total, "bce": bce, "dice": dl}


def crisp_task_loss(
    calibrated_probs: torch.Tensor,
    clipped_target: torch.Tensor,
    boundary_weight: torch.Tensor,
    alpha_hat: torch.Tensor,
    mask: torch.Tensor,
    lambda_value: float,
    mu_value: float,
    eta_dice: float,
    apply_boundary_weight: bool = True,
    apply_identity_regularization: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Compute the CRISP projected task-fitting loss.

    Parameters
    ----------
    calibrated_probs:
        Calibrated probability map p̃(u) = σ(α̂ · z) [B,1,H,W].
    clipped_target:
        Clipped boundary posterior target t*_ε(u) [B,1,H,W].
    boundary_weight:
        Boundary weighting field w_b(u) [B,1,H,W].
    alpha_hat:
        Predicted inverse-temperature field α̂(u) [B,1,H,W].
    mask:
        Binary ground-truth mask y(u) [B,1,H,W].
    lambda_value:
        Boundary projection coefficient λ.
    mu_value:
        Identity regularization coefficient μ.
    eta_dice:
        Weight of the Dice term η.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary containing total task loss and individual components.

    CRISP reference
    ---------------
    instruct.md §12:
      L_task = mean_u [ (1 + λ w_b(u)) BCE(p̃(u), t_eps(u))
                        + μ (1-w_b(u))(α̂(u)-1)² ]
              + η L_Dice(p̃, y)

    Notes
    -----
    The default behavior reproduces the paper loss exactly. The optional flags
    are used only by config-driven ablations so the main CRISP objective remains
    unchanged.
    """
    if calibrated_probs.shape != clipped_target.shape:
        raise ValueError("calibrated_probs and clipped_target must have matching shapes.")
    if alpha_hat.shape != boundary_weight.shape:
        raise ValueError("alpha_hat and boundary_weight must have matching shapes.")

    # --- Weighted BCE ---
    # Element-wise BCE: -t log(p̃) - (1-t) log(1-p̃)
    eps_bce = 1e-7
    p_clamped = calibrated_probs.clamp(eps_bce, 1.0 - eps_bce)
    bce_pixelwise = -(
        clipped_target * p_clamped.log()
        + (1.0 - clipped_target) * (1.0 - p_clamped).log()
    )
    # Boundary weighting on BCE.
    bce_weight = (
        1.0 + lambda_value * boundary_weight
        if apply_boundary_weight
        else torch.ones_like(boundary_weight)
    )
    weighted_bce = bce_weight * bce_pixelwise

    # --- Identity regularization ---
    # μ(1 - w_b)(α̂ - 1)² penalizes deviation from identity away from boundary.
    if apply_identity_regularization:
        identity_reg = mu_value * (1.0 - boundary_weight) * (alpha_hat - 1.0).pow(2)
    else:
        identity_reg = torch.zeros_like(weighted_bce)

    # --- Per-pixel combined, then mean ---
    pixel_loss = weighted_bce + identity_reg
    bce_dice_part = pixel_loss.mean()

    # --- Dice loss on calibrated probs vs hard mask ---
    dl = dice_loss(calibrated_probs, mask.float())

    # --- Total task loss ---
    total = bce_dice_part + eta_dice * dl

    return {
        "task_loss": total,
        "weighted_bce": weighted_bce.mean(),
        "identity_reg": identity_reg.mean(),
        "dice": dl,
    }


def crisp_amortization_loss(
    alpha_hat: torch.Tensor,
    alpha_star: torch.Tensor,
    boundary_weight: torch.Tensor,
    logits: torch.Tensor,
    zeta: float,
    zmax: float = 12.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute the CRISP amortization consistency loss.

    Parameters
    ----------
    alpha_hat:
        Predicted inverse-temperature field [B,1,H,W] (keeps gradients).
    alpha_star:
        Detached local optimum field from the solver [B,1,H,W].
    boundary_weight:
        Boundary weighting field w_b [B,1,H,W].
    logits:
        Raw student logits z(u) [B,1,H,W] — used for confidence mask.
    zeta:
        Logit magnitude threshold defining the confidence mask.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary containing amortization loss and auxiliary diagnostics.

    CRISP reference
    ---------------
    instruct.md §12:
      L_amort = mean_u [ ρ(u) (α̂(u) - sg[α*(u)])² ]
      ρ(u) = w_b(u) · 1{|z̄(u)| ≥ ζ}
    instruct.md §12.1:
      ρ = wb · (|z̃| >= ζ).float()
    """
    if alpha_hat.shape != alpha_star.shape:
        raise ValueError("alpha_hat and alpha_star must have matching shapes.")
    if alpha_star.requires_grad:
        raise ValueError("alpha_star must be detached before amortization supervision.")

    # Confidence mask: suppress near-zero-logit regions using the stabilized
    # detached solver logits z̃ (instruct.md §8, §12.1).
    z_tilde = stabilize_logits_for_solver(logits, zmax=zmax, zeta=zeta)
    confident = (z_tilde.abs() >= zeta).float()  # [B,1,H,W]
    rho = boundary_weight * confident  # [B,1,H,W]

    # Amortization MSE with stop-gradient on alpha_star.
    diff_sq = (alpha_hat - alpha_star.detach()).pow(2)

    # Paper objective: mean_u [rho(u) * (alpha_hat - sg[alpha_star])^2].
    amort_loss = (rho * diff_sq).mean()

    return {
        "amort_loss": amort_loss,
        "rho_coverage": rho.mean(),
        "confident_coverage": confident.mean(),
        "alpha_hat_mean": alpha_hat.mean().detach(),
        "alpha_star_mean": alpha_star.mean().detach(),
    }


def crisp_total_loss(
    task_loss_dict: Dict[str, torch.Tensor],
    amort_loss_dict: Dict[str, torch.Tensor],
    beta_value: float,
) -> Dict[str, torch.Tensor]:
    """
    Combine task and amortization losses into the final CRISP objective.

    Parameters
    ----------
    task_loss_dict:
        Dictionary produced by ``crisp_task_loss``.
    amort_loss_dict:
        Dictionary produced by ``crisp_amortization_loss``.
    beta_value:
        Weight applied to the amortization term (default 0.20).

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary containing the total CRISP loss and all named sub-losses.

    CRISP reference
    ---------------
    instruct.md §12: L_CRISP = L_task + β L_amort.
    """
    total = task_loss_dict["task_loss"] + beta_value * amort_loss_dict["amort_loss"]

    combined: Dict[str, torch.Tensor] = {"loss": total}
    combined.update(task_loss_dict)
    combined.update(amort_loss_dict)
    return combined
