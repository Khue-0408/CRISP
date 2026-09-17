"""
Teacher posterior aggregation for CRISP.

During training, CRISP uses a teacher set {T_m} whose probability maps are combined
into a boundary-local teacher posterior p_T(u). The paper instantiates p_T(u)
through an entropy-and-agreement weighted barycenter. [file:1]

"""

from __future__ import annotations

import hashlib
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
    mode: str = "weighted",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregate teacher probabilities into the CRISP teacher posterior p_T(u).

    Parameters
    ----------
    teacher_probs:
        List of teacher probability maps, each [B, 1, H, W].
        Must be detached from teacher computation graphs.
    tau:
        Entropy weighting coefficient.
    gamma:
        Agreement weighting coefficient.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - p_T: aggregated teacher posterior [B, 1, H, W],
        - weights: normalized teacher weights [M, B, 1, H, W].

    """
    stacked = torch.stack([prob.detach() for prob in teacher_probs], dim=0)  # [M, B, 1, H, W]
    if mode == "weighted":
        weights = compute_teacher_weights(teacher_probs, tau=tau, gamma=gamma)
        p_T = (weights * stacked).sum(dim=0)
    elif mode == "equal_average":
        weights = torch.full_like(stacked, 1.0 / len(teacher_probs))
        p_T = stacked.mean(dim=0)
    else:
        raise ValueError(f"Unknown teacher aggregation mode '{mode}'.")

    return p_T.detach(), weights.detach()


def stable_teacher_noise_seed(experiment_seed: int, image_id: str) -> int:
    """Derive a process-independent generator seed from seed and image identity."""
    if not image_id:
        raise ValueError("Teacher corruption requires a nonempty stable image ID.")
    digest = hashlib.sha256(f"{experiment_seed}\0{image_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def corrupt_teacher_logits(
    logits: torch.Tensor,
    image_ids: list[str],
    experiment_seed: int,
    std: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add per-image Gaussian noise to raw logits before any sigmoid."""
    if logits.shape[0] != len(image_ids):
        raise ValueError("One stable image ID is required per teacher-logit image.")
    if std != 1.0:
        raise ValueError("The journal robustness control requires logit-noise std=1.0.")
    noises = []
    for image_id in image_ids:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(stable_teacher_noise_seed(experiment_seed, image_id))
        noise = torch.randn(logits.shape[1:], generator=generator, dtype=torch.float32)
        noises.append(noise.to(device=logits.device, dtype=logits.dtype))
    stacked_noise = torch.stack(noises, dim=0)
    return (logits.detach() + std * stacked_noise).detach(), stacked_noise


def teacher_robustness_posterior(
    teacher_logits: list[torch.Tensor],
    image_ids: list[str],
    experiment_seed: int,
    mode: str,
    tau: float,
    gamma: float,
    std: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | list[torch.Tensor]]]:
    """Aggregate the same three-teacher corrupted evidence under either rule."""
    if len(teacher_logits) != 3:
        raise ValueError("Robustness control requires exactly three teachers, SAM-Mamba third.")
    before = [torch.sigmoid(logits.detach()) for logits in teacher_logits]
    degraded_logits, noise = corrupt_teacher_logits(
        teacher_logits[2], image_ids, experiment_seed, std=std,
    )
    after = [before[0], before[1], torch.sigmoid(degraded_logits)]
    posterior, weights = aggregate_teacher_posterior(
        after, tau=tau, gamma=gamma, mode=mode,
    )
    diagnostics: dict[str, torch.Tensor | list[torch.Tensor]] = {
        "teacher_probs_before": before,
        "teacher_probs_after": after,
        "degraded_noise": noise,
        "weights": weights,
        "degraded_mean_weight": weights[2].mean(),
    }
    return posterior, diagnostics
