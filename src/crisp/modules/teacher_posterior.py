"""
Teacher posterior aggregation for CRISP.

During training, CRISP uses a teacher set {T_m} whose probability maps are combined
into a boundary-local teacher posterior p_T(u). The paper instantiates p_T(u)
through an entropy-and-agreement weighted barycenter. [file:1]

CRISP reference: instruct.md §4.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


def binary_entropy(prob: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """
    Compute binary entropy H(p) = -p log p - (1 - p) log (1 - p).

    Parameters
    ----------
    prob:
        Probability tensor in [0, 1].
    eps:
        Numerical stability epsilon used before logarithms.

    Returns
    -------
    torch.Tensor
        Entropy tensor with the same shape as ``prob``.

    CRISP reference
    ---------------
    instruct.md §4.2: H(p) = -p log p - (1-p) log(1-p).
    """
    p = prob.clamp(eps, 1.0 - eps)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log())


def compute_teacher_consensus(teacher_probs: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute the mean teacher consensus p̄(u).

    Parameters
    ----------
    teacher_probs:
        List of teacher probability maps, each of shape [B, 1, H, W].
        Must already be detached from teacher computation graphs.

    Returns
    -------
    torch.Tensor
        Mean consensus probability map of shape [B, 1, H, W].

    CRISP reference
    ---------------
    instruct.md §4.2: p̄(u) = (1/M) Σ_m p_m(u).
    """
    # Stack to [M, B, 1, H, W] and average over M.
    stacked = torch.stack([prob.detach() for prob in teacher_probs], dim=0)
    return stacked.mean(dim=0)  # [B, 1, H, W]


def compute_teacher_weights(
    teacher_probs: List[torch.Tensor],
    tau: float,
    gamma: float,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """
    Compute entropy-and-agreement teacher weights π_m(u).

    Parameters
    ----------
    teacher_probs:
        List of teacher probability maps.
    tau:
        Strength of entropy penalization.
    gamma:
        Strength of consensus-deviation penalization.
    eps:
        Numerical stability epsilon.

    Returns
    -------
    torch.Tensor
        Weight tensor of shape [M, B, 1, H, W], normalized to sum to 1
        over the teacher dimension (dim=0).

    CRISP reference
    ---------------
    instruct.md §4.2:
      π_m(u) ∝ exp(-τ H(p_m(u)) - γ (p_m(u) - p̄(u))²)
    """
    p_bar = compute_teacher_consensus(teacher_probs)  # [B, 1, H, W]
    stacked = torch.stack([prob.detach() for prob in teacher_probs], dim=0)  # [M, B, 1, H, W]

    # Per-teacher entropy: H(p_m) — [M, B, 1, H, W]
    H_m = binary_entropy(stacked, eps=eps)

    # Per-teacher deviation from consensus.
    deviation_sq = (stacked - p_bar.unsqueeze(0)).pow(2)  # [M, B, 1, H, W]

    # Unnormalized log-weights.
    log_w = -tau * H_m - gamma * deviation_sq  # [M, B, 1, H, W]

    # Softmax over teacher dimension to normalize.
    weights = torch.softmax(log_w, dim=0)  # [M, B, 1, H, W]
    return weights


def aggregate_teacher_posterior(
    teacher_probs: List[torch.Tensor],
    tau: float,
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregate teacher probabilities into the CRISP teacher posterior p_T(u).

    Parameters
    ----------
    teacher_probs:
        List of teacher probability maps, each [B, 1, H, W].
        Must be detached from teacher computation graphs.
    tau:
        Entropy weighting coefficient (default 1.0).
    gamma:
        Agreement weighting coefficient (default 6.0).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - p_T: aggregated teacher posterior [B, 1, H, W],
        - weights: normalized teacher weights [M, B, 1, H, W].

    CRISP reference
    ---------------
    instruct.md §4.2: p_T(u) = Σ_m π_m(u) p_m(u).
    """
    weights = compute_teacher_weights(teacher_probs, tau=tau, gamma=gamma)
    stacked = torch.stack([prob.detach() for prob in teacher_probs], dim=0)  # [M, B, 1, H, W]

    # Weighted barycenter.
    p_T = (weights * stacked).sum(dim=0)  # [B, 1, H, W]

    return p_T.detach(), weights.detach()
