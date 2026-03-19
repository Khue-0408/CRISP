"""
Unit tests for baseline and CRISP losses.
"""

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
                        lambda_value=0.8, mu_value=0.05, eta_dice=0.5)
    assert "task_loss" in d
    assert "weighted_bce" in d
    assert "identity_reg" in d
    assert "dice" in d


def test_crisp_amort_loss_detach() -> None:
    """Amortization loss should not backprop through alpha_star."""
    alpha_hat = torch.randn(1, 1, 8, 8, requires_grad=True)
    alpha_star = torch.randn(1, 1, 8, 8)  # no grad
    wb = torch.rand(1, 1, 8, 8)
    logits = torch.randn(1, 1, 8, 8)

    d = crisp_amortization_loss(alpha_hat, alpha_star, wb, logits, zeta=1e-2)
    d["amort_loss"].backward()
    assert alpha_hat.grad is not None, "alpha_hat should receive gradients"


def test_crisp_amort_loss_matches_global_mean_objective() -> None:
    """Amortization loss should implement mean_u[rho * diff^2], not support renormalization."""
    alpha_hat = torch.tensor([[[[2.0, 0.0]]]])
    alpha_star = torch.tensor([[[[1.0, 0.0]]]])
    wb = torch.tensor([[[[1.0, 0.0]]]])
    logits = torch.tensor([[[[2.0, 2.0]]]])

    d = crisp_amortization_loss(alpha_hat, alpha_star, wb, logits, zeta=1e-2)
    expected = torch.tensor(0.5)  # mean over both pixels of [1 * (2-1)^2, 0]
    assert torch.allclose(d["amort_loss"], expected)


def test_crisp_total_loss_combines() -> None:
    """Total loss should combine task + beta * amort."""
    task_dict = {"task_loss": torch.tensor(1.0)}
    amort_dict = {"amort_loss": torch.tensor(2.0)}
    d = crisp_total_loss(task_dict, amort_dict, beta_value=0.2)
    expected = 1.0 + 0.2 * 2.0
    assert abs(d["loss"].item() - expected) < 1e-6
