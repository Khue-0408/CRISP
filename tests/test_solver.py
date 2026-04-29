"""
Unit tests for the detached local projection solver.

This is one of the most critical correctness tests in the repository because
alpha* supervision directly shapes the amortized projector. [file:1]
"""

import torch
from crisp.modules.solver import (
    closed_form_seed,
    projection_gradient,
    solve_alpha_star,
    stabilize_logits_for_solver,
)


def test_alpha_star_within_bounds() -> None:
    """Ensure the solver always returns alpha values inside the feasible interval."""
    alpha_min, alpha_max = 0.5, 1.75
    B, H, W = 2, 16, 16

    logits = torch.randn(B, 1, H, W) * 3.0
    target = torch.rand(B, 1, H, W).clamp(1e-4, 1 - 1e-4)
    wb = torch.rand(B, 1, H, W)

    alpha_star, diag = solve_alpha_star(
        logits, target, wb,
        lambda_value=1.0, mu_value=0.25,
        alpha_min=alpha_min, alpha_max=alpha_max,
        zmax=8.0, zeta=0.10,
    )

    assert alpha_star.min() >= alpha_min - 1e-6, f"Below alpha_min: {alpha_star.min()}"
    assert alpha_star.max() <= alpha_max + 1e-6, f"Above alpha_max: {alpha_star.max()}"


def test_alpha_star_is_detached() -> None:
    """Alpha star must be detached from the gradient graph."""
    logits = torch.randn(1, 1, 8, 8, requires_grad=True)
    target = torch.rand(1, 1, 8, 8).clamp(1e-4, 1 - 1e-4)
    wb = torch.rand(1, 1, 8, 8)

    alpha_star, _ = solve_alpha_star(
        logits, target, wb,
        lambda_value=1.0, mu_value=0.25,
        alpha_min=0.5, alpha_max=1.75,
        zmax=8.0, zeta=0.10,
    )
    assert not alpha_star.requires_grad, "alpha_star must be detached"


def test_stabilized_logits_detached() -> None:
    """Stabilized logits should be detached."""
    logits = torch.randn(1, 1, 8, 8, requires_grad=True)
    z_tilde = stabilize_logits_for_solver(logits, zmax=8.0, zeta=0.10)
    assert not z_tilde.requires_grad, "z_tilde must be detached"


def test_stabilized_logits_magnitude() -> None:
    """Stabilized logits should have |z̃| >= zeta."""
    logits = torch.randn(1, 1, 16, 16) * 0.001  # near-zero logits
    z_tilde = stabilize_logits_for_solver(logits, zmax=8.0, zeta=0.10)
    assert z_tilde.abs().min() >= 0.10 - 1e-8, f"Min magnitude below zeta: {z_tilde.abs().min()}"


def test_stabilized_zero_logits_force_zeta() -> None:
    """Exact zero logits must still stabilize to magnitude zeta."""
    logits = torch.zeros(1, 1, 4, 4)
    z_tilde = stabilize_logits_for_solver(logits, zmax=8.0, zeta=0.10)
    assert torch.allclose(z_tilde.abs(), torch.full_like(z_tilde.abs(), 0.10))


def test_closed_form_seed_in_bounds() -> None:
    """Closed-form seed should be in [alpha_min, alpha_max]."""
    z = torch.randn(1, 1, 8, 8)
    z_tilde = stabilize_logits_for_solver(z, zmax=8.0, zeta=0.10)
    t = torch.rand(1, 1, 8, 8).clamp(1e-4, 1 - 1e-4)
    seed = closed_form_seed(z_tilde, t, 0.5, 1.75)
    assert seed.min() >= 0.5 - 1e-6
    assert seed.max() <= 1.75 + 1e-6
