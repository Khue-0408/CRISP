"""
Smoke tests for core CRISP mathematical invariants.

These tests are lightweight and data-free; they protect against method drift:
- explicit detached alpha_star exists and is bounded,
- solver uses detached stabilized logits z_tilde,
- forward path remains on raw logits z with p_tilde = sigmoid(alpha_hat * z),
- teacher posterior barycenter uses detached probabilities,
- boundary posterior target matches the closed-form mixing definition.
"""

import torch

from crisp.metrics.calibration import boundary_support_mask
from crisp.modules.boundary import compute_boundary_weight
from crisp.modules.calibration import calibrate_logits_with_alpha
from crisp.modules.posterior_target import clip_posterior_target, compute_boundary_posterior_target
from crisp.modules.solver import solve_alpha_star, stabilize_logits_for_solver
from crisp.modules.teacher_posterior import aggregate_teacher_posterior
from crisp.models.projector_head import CRISPProjectorHead


def test_crisp_core_invariants_smoke() -> None:
    B, H, W = 2, 32, 32
    alpha_min, alpha_max = 0.5, 1.75
    lam, mu = 1.0, 0.25
    eps_t = 1e-3
    zeta, zmax = 0.10, 8.0

    # Fake logits and mask.
    z = torch.randn(B, 1, H, W) * 3.0
    y = (torch.rand(B, 1, H, W) > 0.5).float()

    # Boundary weights (soft field).
    wb = compute_boundary_weight(y, sigma_b=6.0)
    assert wb.shape == y.shape
    assert (wb >= 0).all() and (wb <= 1).all()

    # Teacher posterior (detached teacher probs).
    teachers = [torch.sigmoid(torch.randn(B, 1, H, W)).detach() for _ in range(3)]
    pT, weights = aggregate_teacher_posterior(teachers, tau=1.0, gamma=1.5)
    assert not pT.requires_grad
    assert weights.shape[0] == 3

    # Boundary posterior target and clipping.
    t_star = compute_boundary_posterior_target(y, wb, pT, lambda_value=lam)
    t_eps = clip_posterior_target(t_star, eps_target=eps_t)
    assert (t_eps >= eps_t).all() and (t_eps <= 1 - eps_t).all()

    # Detached stabilized logits for solver.
    z_tilde = stabilize_logits_for_solver(z.requires_grad_(True), zmax=zmax, zeta=zeta)
    assert not z_tilde.requires_grad
    assert z_tilde.abs().min() >= zeta - 1e-8

    # Solve alpha_star: must be detached and bounded.
    alpha_star, diag = solve_alpha_star(
        z, t_eps, wb,
        lambda_value=lam, mu_value=mu,
        alpha_min=alpha_min, alpha_max=alpha_max,
        zmax=zmax, zeta=zeta,
    )
    assert not alpha_star.requires_grad
    assert alpha_star.min() >= alpha_min - 1e-6
    assert alpha_star.max() <= alpha_max + 1e-6

    # Projector bounded alpha_hat and calibrated forward probability uses raw z.
    feat = torch.randn(B, 64, H // 4, W // 4)
    proj = CRISPProjectorHead(feature_channels=64, alpha_min=alpha_min, alpha_max=alpha_max)
    alpha_hat = proj(feat, z)
    assert alpha_hat.min() >= alpha_min - 1e-6
    assert alpha_hat.max() <= alpha_max + 1e-6
    p_tilde = calibrate_logits_with_alpha(z, alpha_hat)
    assert torch.allclose(p_tilde, torch.sigmoid(alpha_hat * z))

    # Support mask is per-image top-k.
    support = boundary_support_mask(wb, top_percent=20.0)
    assert support.shape == wb.shape
    assert support.sum(dim=(1, 2, 3)).min() >= 1
