"""
Unit tests for the detached local projection solver.

This is one of the most critical correctness tests in the repository because
alpha* supervision directly shapes the amortized projector.
"""

import math

import torch
import crisp.modules.solver as solver_module
from crisp.modules.losses import crisp_amortization_loss
from crisp.modules.solver import (
    _safeguarded_newton_step,
    closed_form_seed,
    projection_gradient,
    projection_hessian,
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


def test_amortization_cannot_backpropagate_through_solver_target() -> None:
    logits = torch.tensor([[[[1.0]]]], requires_grad=True)
    target = torch.tensor([[[[0.7]]]], requires_grad=True)
    wb = torch.tensor([[[[0.5]]]])
    alpha_star, _ = solve_alpha_star(
        logits, target, wb, 1.0, 0.25, 0.5, 1.75, 8.0, 0.10,
    )
    alpha_hat = torch.tensor([[[[1.2]]]], requires_grad=True)
    loss = crisp_amortization_loss(alpha_hat, alpha_star, wb, logits, zeta=0.10)
    loss["amort_loss"].backward()
    assert alpha_hat.grad is not None
    assert logits.grad is None
    assert target.grad is None


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


def test_stabilization_clips_floors_and_uses_positive_zero_sign() -> None:
    logits = torch.tensor([[[[-20.0, -0.01, 0.0, 0.01, 20.0]]]], requires_grad=True)
    result = stabilize_logits_for_solver(logits, zmax=8.0, zeta=0.10)
    expected = torch.tensor([[[[-8.0, -0.10, 0.10, 0.10, 8.0]]]])
    assert torch.allclose(result, expected)
    assert not result.requires_grad


def test_closed_form_seed_uses_target_logit_not_probability() -> None:
    z_tilde = torch.tensor([[[[1.0, 1.0, 1.0]]]])
    target = torch.tensor([[[[torch.sigmoid(torch.tensor(1.0)), 0.99, 0.10]]]])
    seed = closed_form_seed(z_tilde, target, 0.5, 1.75)
    assert torch.allclose(seed, torch.tensor([[[[1.0, 1.75, 0.5]]]]), atol=1e-6)


def _reference_root(z: float, t: float, w: float, mu: float) -> float:
    """Independent double-precision monotone bisection reference."""
    lo, hi = 0.5, 1.75
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        sigmoid = 1.0 / (1.0 + math.exp(-mid * z))
        g = (1.0 + w) * (sigmoid - t) * z + 2.0 * mu * (1.0 - w) * (mid - 1.0)
        if g > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def test_kkt_endpoints_include_derivative_zero_ties_and_skip_seed(monkeypatch) -> None:
    z = torch.ones(1, 1, 1, 4)
    lower_tie = torch.sigmoid(torch.tensor(0.5))
    upper_tie = torch.sigmoid(torch.tensor(1.75))
    target = torch.tensor([[[[0.10, lower_tie, 0.95, upper_tie]]]])
    wb = torch.ones_like(z)
    g_lo = projection_gradient(torch.full_like(z, 0.5), z, target, wb, 1.0, 0.0)
    g_hi = projection_gradient(torch.full_like(z, 1.75), z, target, wb, 1.0, 0.0)
    assert g_lo[0, 0, 0, 0] > 0
    assert g_lo[0, 0, 0, 1] == 0
    assert g_hi[0, 0, 0, 2] < 0
    assert g_lo[0, 0, 0, 3] < 0 and g_hi[0, 0, 0, 3] == 0

    def unexpected_seed(*args, **kwargs):
        raise AssertionError("Endpoint-only batch must not enter interior seed/refinement")

    monkeypatch.setattr(solver_module, "closed_form_seed", unexpected_seed)
    result, diag = solve_alpha_star(
        z, target, wb, 1.0, 0.0, 0.5, 1.75, 8.0, 0.10,
    )
    assert torch.equal(result, torch.tensor([[[[0.5, 0.5, 1.75, 1.75]]]]))
    assert diag["bracket_rate"].item() == 0.0
    assert diag["bisection_pixels"].item() == 0.0
    assert diag["newton_accepted"].item() == 0.0


def test_mixed_endpoint_and_interior_roots_match_reference() -> None:
    logits = torch.tensor([[[[1.0, 1.0, 1.0, -1.0]]]], requires_grad=True)
    target = torch.tensor([[[[0.10, 0.95, 0.70, 0.30]]]], requires_grad=True)
    wb = torch.tensor([[[[1.0, 1.0, 0.5, 0.5]]]])
    result, diag = solve_alpha_star(
        logits, target, wb, 1.0, 0.25, 0.5, 1.75, 8.0, 0.10,
    )
    expected = torch.tensor([
        0.5, 1.75,
        _reference_root(1.0, 0.70, 0.5, 0.25),
        _reference_root(-1.0, 0.30, 0.5, 0.25),
    ])
    assert torch.allclose(result.flatten(), expected, atol=5e-4, rtol=0)
    assert diag["bracket_rate"].item() == 0.5
    assert not result.requires_grad
    z_tilde = stabilize_logits_for_solver(logits, 8.0, 0.10)
    residual = projection_gradient(result, z_tilde, target.detach(), wb, 1.0, 0.25)
    assert residual.flatten()[2:].abs().max() < 1e-4


def test_newton_outside_current_bracket_uses_bisection() -> None:
    z = torch.tensor([[[[-8.0]]]])
    target = torch.tensor([[[[0.001]]]])
    wb = torch.zeros_like(z)
    alpha = torch.tensor([[[[1.2]]]])
    lo = torch.full_like(z, 0.5)
    hi = torch.full_like(z, 1.75)
    raw_proposal = alpha - projection_gradient(alpha, z, target, wb, 1.0, 0.0) / (
        projection_hessian(alpha, z, wb, 1.0, 0.0)
    )
    assert raw_proposal.item() < lo.item()
    candidate, accepted, outside, no_decrease, invalid = _safeguarded_newton_step(
        alpha, lo, hi, z, target, wb, 1.0, 0.0, torch.ones_like(z, dtype=torch.bool),
    )
    assert outside.item() and not accepted.item()
    assert not no_decrease.item() and not invalid.item()
    assert torch.equal(candidate, torch.full_like(z, 1.125))


def test_newton_non_decreasing_residual_uses_bisection() -> None:
    z = torch.tensor([[[[-8.0]]]])
    target = torch.tensor([[[[0.001]]]])
    wb = torch.zeros_like(z)
    alpha = torch.tensor([[[[1.0]]]])
    lo = torch.full_like(z, 0.5)
    hi = torch.full_like(z, 1.75)
    current_residual = projection_gradient(alpha, z, target, wb, 1.0, 0.0).abs()
    raw_proposal = alpha - projection_gradient(alpha, z, target, wb, 1.0, 0.0) / (
        projection_hessian(alpha, z, wb, 1.0, 0.0)
    )
    proposal_residual = projection_gradient(raw_proposal, z, target, wb, 1.0, 0.0).abs()
    assert lo.item() < raw_proposal.item() < hi.item()
    assert proposal_residual.item() >= current_residual.item()
    candidate, accepted, outside, no_decrease, invalid = _safeguarded_newton_step(
        alpha, lo, hi, z, target, wb, 1.0, 0.0, torch.ones_like(z, dtype=torch.bool),
    )
    assert no_decrease.item() and not accepted.item()
    assert not outside.item() and not invalid.item()
    assert torch.equal(candidate, torch.full_like(z, 1.125))


def test_interior_newton_steps_preserve_root_bracket(monkeypatch) -> None:
    original_step = solver_module._safeguarded_newton_step
    widths = []

    def checked_step(alpha, lo, hi, z, target, wb, lam, mu, active):
        g_lo = projection_gradient(lo, z, target, wb, lam, mu)
        g_hi = projection_gradient(hi, z, target, wb, lam, mu)
        assert torch.all(g_lo[active] <= 0)
        assert torch.all(g_hi[active] >= 0)
        widths.append((hi - lo)[active].max().item())
        return original_step(alpha, lo, hi, z, target, wb, lam, mu, active)

    monkeypatch.setattr(solver_module, "_safeguarded_newton_step", checked_step)
    z = torch.tensor([[[[1.0]]]])
    target = torch.tensor([[[[0.70]]]])
    wb = torch.tensor([[[[0.5]]]])
    result, _ = solve_alpha_star(z, target, wb, 1.0, 0.25, 0.5, 1.75, 8.0, 0.10)
    assert widths and widths == sorted(widths, reverse=True)
    assert abs(result.item() - _reference_root(1.0, 0.70, 0.5, 0.25)) < 5e-4


def test_production_solver_records_bisection_fallback() -> None:
    z = torch.tensor([[[[-6.0]]]])
    target = torch.tensor([[[[1e-5]]]])
    wb = torch.zeros_like(z)
    result, diag = solve_alpha_star(z, target, wb, 0.0, 0.001, 0.5, 1.75, 8.0, 0.10)
    assert diag["newton_no_residual_decrease"].item() >= 1.0
    assert diag["newton_fallback"].item() >= 1.0
    assert diag["bisection_pixels"].item() == 1.0
    residual = projection_gradient(result, z, target, wb, 0.0, 0.001)
    assert residual.abs().item() < 1e-4
    assert abs(result.item() - _reference_root(-6.0, 1e-5, 0.0, 0.001)) < 5e-4


def test_newton_fallback_consumes_the_only_bisection_update() -> None:
    z = torch.tensor([[[[-6.0]]]])
    target = torch.tensor([[[[1e-5]]]])
    wb = torch.zeros_like(z)
    result, diag = solve_alpha_star(
        z, target, wb, 0.0, 0.001, 0.5, 1.75, 8.0, 0.10,
        newton_steps=3, bisection_steps=1,
    )
    assert diag["newton_fallback"].item() == 1.0
    assert diag["bisection_updates_max"].item() == 1
    assert diag["bisection_pixels"].item() == 1.0
    assert diag["newton_accepted_max"].item() <= 3
    assert 0.5 <= result.item() <= 1.75


def test_canonical_bisection_budget_and_reference_accuracy() -> None:
    z = torch.tensor([[[[-6.0]]]])
    target = torch.tensor([[[[1e-5]]]])
    wb = torch.zeros_like(z)
    result, diag = solve_alpha_star(
        z, target, wb, 0.0, 0.001, 0.5, 1.75, 8.0, 0.10,
        newton_steps=3, bisection_steps=12,
    )
    assert diag["newton_fallback"].item() >= 1.0
    assert 1 <= diag["bisection_updates_max"].item() <= 12
    assert diag["newton_accepted_max"].item() <= 3
    assert 0.5 <= result.item() <= 1.75
    residual = projection_gradient(result, z, target, wb, 0.0, 0.001)
    assert residual.abs().item() < 1e-4
    assert abs(result.item() - _reference_root(-6.0, 1e-5, 0.0, 0.001)) < 5e-4


def test_bisection_budget_is_independent_per_pixel() -> None:
    z = torch.tensor([[[[-6.0, 4.0]]]])
    target = torch.tensor([[[[1e-5, 0.999]]]])
    wb = torch.zeros_like(z)
    result, diag = solve_alpha_star(
        z, target, wb, 1.0, 0.001, 0.5, 1.75, 8.0, 0.10,
        newton_steps=1, bisection_steps=1,
    )
    separate = torch.cat([
        solve_alpha_star(
            z[..., i:i + 1], target[..., i:i + 1], wb[..., i:i + 1],
            1.0, 0.001, 0.5, 1.75, 8.0, 0.10,
            newton_steps=1, bisection_steps=1,
        )[0]
        for i in range(2)
    ], dim=-1)
    assert torch.equal(result, separate)
    assert diag["newton_fallback"].item() == 1.0
    assert diag["newton_accepted"].item() == 1.0
    assert diag["bisection_updates_max"].item() == 1
    assert diag["bisection_pixels"].item() == 1.0


def test_zero_bisection_budget_keeps_rejected_iterate() -> None:
    z = torch.tensor([[[[-6.0]]]])
    target = torch.tensor([[[[1e-5]]]])
    wb = torch.zeros_like(z)
    result, diag = solve_alpha_star(
        z, target, wb, 0.0, 0.001, 0.5, 1.75, 8.0, 0.10,
        newton_steps=3, bisection_steps=0,
    )
    assert diag["newton_no_residual_decrease"].item() >= 1.0
    assert diag["newton_fallback"].item() == 0.0
    assert diag["bisection_updates_max"].item() == 0
    assert 0.5 <= result.item() <= 1.75
