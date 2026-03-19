"""
Unit tests for the CRISP projector head.
"""

import torch
from crisp.models.projector_head import CRISPProjectorHead


def test_projector_output_in_range() -> None:
    """Ensure alpha_hat lies within [alpha_min, alpha_max]."""
    alpha_min, alpha_max = 0.5, 1.8
    projector = CRISPProjectorHead(
        feature_channels=32,
        hidden_channels=16,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )

    features = torch.randn(2, 32, 22, 22)
    logits = torch.randn(2, 1, 88, 88)

    alpha_hat = projector(features, logits)
    assert alpha_hat.shape == (2, 1, 88, 88), f"Shape mismatch: {alpha_hat.shape}"
    assert alpha_hat.min() >= alpha_min - 1e-5, f"Below alpha_min: {alpha_hat.min()}"
    assert alpha_hat.max() <= alpha_max + 1e-5, f"Above alpha_max: {alpha_hat.max()}"


def test_projector_output_size_override() -> None:
    """Projector should respect output_size argument."""
    projector = CRISPProjectorHead(feature_channels=16, hidden_channels=8)
    features = torch.randn(1, 16, 11, 11)
    logits = torch.randn(1, 1, 44, 44)

    alpha = projector(features, logits, output_size=(64, 64))
    assert alpha.shape == (1, 1, 64, 64)


def test_projector_gradients_flow() -> None:
    """Alpha_hat must keep gradients for backpropagation."""
    projector = CRISPProjectorHead(feature_channels=16, hidden_channels=8)
    features = torch.randn(1, 16, 11, 11, requires_grad=True)
    logits = torch.randn(1, 1, 44, 44, requires_grad=True)

    alpha = projector(features, logits)
    loss = alpha.mean()
    loss.backward()
    assert features.grad is not None
    assert logits.grad is not None
