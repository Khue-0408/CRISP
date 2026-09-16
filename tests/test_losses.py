"""
Unit tests for baseline and CRISP losses.
"""

import pytest
import torch
from crisp.modules.losses import (
    baseline_bce_dice_loss,
    crisp_amortization_loss,
    crisp_task_loss,
    crisp_total_loss,
    dice_loss,
)


def test_dice_loss_scalar_output() -> None:
    """Ensure Dice loss returns a scalar tensor."""
    probs = torch.rand(2, 1, 16, 16)
    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    loss = dice_loss(probs, target)
    assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"
    assert loss.item() >= 0.0


def test_dice_loss_perfect_prediction() -> None:
    """Dice loss should be near 0 for perfect predictions."""
    mask = torch.ones(1, 1, 8, 8)
    loss = dice_loss(mask, mask)
    assert loss.item() < 0.01, f"Expected near-zero loss, got {loss.item()}"


def test_dice_loss_uses_manuscript_smoothing() -> None:
    probs = torch.tensor([[[[0.5, 0.25]]]])
    target = torch.tensor([[[[1.0, 0.0]]]])
    expected = 1.0 - (2.0 * 0.5 + 1.0) / (0.5 + 0.25 + 1.0 + 1.0)
    old_smoothing = 1.0 - (2.0 * 0.5 + 1e-6) / (0.5 + 0.25 + 1.0 + 1e-6)

    assert dice_loss(probs, target).item() == pytest.approx(expected)
    assert abs(expected - old_smoothing) > 0.1


def test_dice_loss_empty_target_uses_same_formula() -> None:
    probs = torch.tensor([[[[0.25, 0.75]]]])
    target = torch.zeros_like(probs)
    expected = 1.0 - 1.0 / (probs.sum().item() + 1.0)

    assert dice_loss(probs, target).item() == pytest.approx(expected)


def test_baseline_and_crisp_task_use_same_training_dice_default() -> None:
    probs = torch.tensor([[[[0.5, 0.25]]]])
    target = torch.tensor([[[[1.0, 0.0]]]])
    logits = torch.logit(probs)
    expected = dice_loss(probs, target)
    baseline = baseline_bce_dice_loss(logits, target)
    crisp = crisp_task_loss(
        probs, target, torch.zeros_like(probs), torch.ones_like(probs), target,
        lambda_value=0.0, mu_value=0.0, eta_dice=0.5,
    )

    assert torch.allclose(baseline["dice"], expected)
    assert torch.allclose(crisp["dice"], expected)


def test_baseline_loss_dict_keys() -> None:
    """Baseline loss should return dict with loss, bce, dice."""
    logits = torch.randn(2, 1, 16, 16)
    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    d = baseline_bce_dice_loss(logits, target)
    assert "loss" in d and "bce" in d and "dice" in d


def test_crisp_task_loss_keys() -> None:
    """CRISP task loss should return expected keys."""
    B, H, W = 2, 8, 8
    p_tilde = torch.rand(B, 1, H, W)
    t_eps = torch.rand(B, 1, H, W).clamp(1e-4, 1-1e-4)
    wb = torch.rand(B, 1, H, W)
    alpha_hat = torch.ones(B, 1, H, W)
    mask = (torch.rand(B, 1, H, W) > 0.5).float()

    d = crisp_task_loss(p_tilde, t_eps, wb, alpha_hat, mask,
                        lambda_value=1.0, mu_value=0.25, eta_dice=0.5)
    assert "task_loss" in d
    assert "weighted_bce" in d
    assert "identity_reg" in d
    assert "dice" in d


def test_crisp_amort_loss_detach() -> None:
    """Only the projector prediction receives amortization gradients."""
    alpha_hat = torch.tensor([[[[2.0]]]], requires_grad=True)
    alpha_star_source = torch.tensor([[[[1.0]]]], requires_grad=True)
    logits = torch.tensor([[[[0.25]]]], requires_grad=True)

    d = crisp_amortization_loss(
        alpha_hat, alpha_star_source.detach(), torch.ones_like(logits), logits, zeta=0.10
    )
    d["amort_loss"].backward()
    assert torch.allclose(alpha_hat.grad, torch.tensor([[[[2.0]]]]))
    assert alpha_star_source.grad is None
    assert logits.grad is None


def test_crisp_amort_loss_rejects_gradient_bearing_target() -> None:
    alpha_hat = torch.tensor([[[[2.0]]]])
    alpha_star = torch.tensor([[[[1.0]]]], requires_grad=True)
    wb = torch.ones_like(alpha_hat)
    logits = torch.tensor([[[[0.25]]]])

    with pytest.raises(ValueError, match="alpha_star must be detached"):
        crisp_amortization_loss(alpha_hat, alpha_star, wb, logits, zeta=0.10)


def test_crisp_amort_loss_excludes_near_zero_logits() -> None:
    logits = torch.tensor([[[[-0.099, 0.0, 0.099]]]])
    alpha_hat = torch.full_like(logits, 2.0)
    alpha_star = torch.ones_like(logits)
    wb = torch.tensor([[[[1.0, 0.5, 0.25]]]])

    d = crisp_amortization_loss(alpha_hat, alpha_star, wb, logits, zeta=0.10)
    assert d["amort_loss"].item() == 0.0
    assert d["confident_coverage"].item() == 0.0
    assert d["rho_coverage"].item() == 0.0


def test_crisp_amort_loss_includes_exact_zeta_on_both_sides() -> None:
    logits = torch.tensor([[[[-0.10, 0.10]]]])
    d = crisp_amortization_loss(
        torch.full_like(logits, 2.0), torch.ones_like(logits),
        torch.ones_like(logits), logits, zeta=0.10,
    )
    assert torch.allclose(d["amort_loss"], torch.tensor(1.0))
    assert d["confident_coverage"].item() == 1.0


def test_crisp_amort_loss_includes_above_zeta_on_both_sides() -> None:
    logits = torch.tensor([[[[-0.25, 0.25]]]])
    d = crisp_amortization_loss(
        torch.full_like(logits, 2.0), torch.ones_like(logits),
        torch.ones_like(logits), logits, zeta=0.10,
    )
    assert torch.allclose(d["amort_loss"], torch.tensor(1.0))
    assert d["confident_coverage"].item() == 1.0


def test_crisp_amort_loss_multiplies_boundary_weights() -> None:
    logits = torch.tensor([[[[0.25, 0.25, 0.25]]]])
    wb = torch.tensor([[[[1.0, 0.5, 0.0]]]])
    d = crisp_amortization_loss(
        torch.full_like(logits, 2.0), torch.ones_like(logits), wb, logits, zeta=0.10,
    )
    assert torch.allclose(d["amort_loss"], torch.tensor(0.5))
    assert torch.allclose(d["rho_coverage"], torch.tensor(0.5))


def test_crisp_amort_loss_mixed_manual_value() -> None:
    logits = torch.tensor([[[[0.0, 0.10, -0.25]]]])
    alpha_hat = torch.tensor([[[[2.0, 3.0, 4.0]]]])
    alpha_star = torch.ones_like(logits)
    wb = torch.tensor([[[[1.0, 0.5, 0.25]]]])

    d = crisp_amortization_loss(alpha_hat, alpha_star, wb, logits, zeta=0.10)
    # Full-domain mean of [0, 0.5 * 2^2, 0.25 * 3^2].
    assert torch.allclose(d["amort_loss"], torch.tensor(4.25 / 3.0))
    assert torch.allclose(d["confident_coverage"], torch.tensor(2.0 / 3.0))
    assert torch.allclose(d["rho_coverage"], torch.tensor(0.25))


def test_crisp_amort_loss_clips_logits_before_support() -> None:
    logits = torch.tensor([[[[-9.0, 9.0, 0.0]]]])
    d = crisp_amortization_loss(
        torch.full_like(logits, 2.0), torch.ones_like(logits),
        torch.ones_like(logits), logits, zeta=0.30, zmax=0.20,
    )
    assert d["amort_loss"].item() == 0.0
    assert d["confident_coverage"].item() == 0.0
    assert d["rho_coverage"].item() == 0.0


def test_crisp_amort_loss_matches_global_mean_objective() -> None:
    """Amortization loss should implement mean_u[rho * diff^2], not support renormalization."""
    alpha_hat = torch.tensor([[[[2.0, 0.0]]]])
    alpha_star = torch.tensor([[[[1.0, 0.0]]]])
    wb = torch.tensor([[[[1.0, 0.0]]]])
    logits = torch.tensor([[[[2.0, 2.0]]]])

    d = crisp_amortization_loss(alpha_hat, alpha_star, wb, logits, zeta=0.10)
    expected = torch.tensor(0.5)  # mean over both pixels of [1 * (2-1)^2, 0]
    assert torch.allclose(d["amort_loss"], expected)


def test_crisp_total_loss_combines() -> None:
    """Total loss should combine task + beta * amort."""
    task_dict = {"task_loss": torch.tensor(1.0)}
    amort_dict = {"amort_loss": torch.tensor(2.0)}
    d = crisp_total_loss(task_dict, amort_dict, beta_value=0.35)
    expected = 1.0 + 0.35 * 2.0
    assert abs(d["loss"].item() - expected) < 1e-6
