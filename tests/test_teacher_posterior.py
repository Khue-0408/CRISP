"""
Unit tests for teacher posterior aggregation.

Tests should verify:
- weights sum to one over teachers,
- entropy weighting behaves sensibly,
- aggregated posterior has the expected shape.
"""

import torch
from crisp.modules.teacher_posterior import (
    aggregate_teacher_posterior,
    binary_entropy,
    compute_teacher_consensus,
    compute_teacher_weights,
)


def test_teacher_weights_sum_to_one() -> None:
    """Ensure teacher aggregation weights form a proper simplex at each pixel."""
    M = 3
    teacher_probs = [torch.rand(2, 1, 8, 8) for _ in range(M)]
    weights = compute_teacher_weights(teacher_probs, tau=1.0, gamma=6.0)

    assert weights.shape == (M, 2, 1, 8, 8)
    weight_sum = weights.sum(dim=0)
    assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-5), (
        f"Weights don't sum to 1: {weight_sum.min()} to {weight_sum.max()}"
    )


def test_aggregated_posterior_shape() -> None:
    """p_T should have shape [B, 1, H, W]."""
    teacher_probs = [torch.rand(4, 1, 16, 16) for _ in range(3)]
    pT, weights = aggregate_teacher_posterior(teacher_probs, tau=1.0, gamma=6.0)
    assert pT.shape == (4, 1, 16, 16)
    assert weights.shape == (3, 4, 1, 16, 16)


def test_binary_entropy_range() -> None:
    """Binary entropy should be non-negative and max at p=0.5."""
    p = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    H = binary_entropy(p)
    assert (H >= 0).all()
    assert H[2] >= H[0] and H[2] >= H[4], "Entropy should peak at 0.5"


def test_consensus_is_mean() -> None:
    """Teacher consensus should be simple mean."""
    p1 = torch.full((1, 1, 4, 4), 0.2)
    p2 = torch.full((1, 1, 4, 4), 0.8)
    consensus = compute_teacher_consensus([p1, p2])
    assert torch.allclose(consensus, torch.full_like(consensus, 0.5), atol=1e-6)


def test_aggregated_posterior_in_0_1() -> None:
    """Aggregated posterior p_T should remain in [0, 1]."""
    teacher_probs = [torch.rand(2, 1, 8, 8) for _ in range(3)]
    pT, _ = aggregate_teacher_posterior(teacher_probs, tau=1.0, gamma=6.0)
    assert pT.min() >= 0.0 - 1e-6
    assert pT.max() <= 1.0 + 1e-6
