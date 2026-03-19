"""
Calibration metrics for dense prediction.

The paper reports:
- ECE with 15 equal-width bins,
- boundary-ECE on top boundary-support pixels,
- BA-ECE,
- TACE,
- Brier score,
- NLL. [file:1]

CRISP reference: instruct.md §16.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> torch.Tensor:
    """
    Compute standard Expected Calibration Error (ECE).

    Parameters
    ----------
    probs:
        Foreground probabilities, any shape (will be flattened).
    labels:
        Binary labels of the same shape.
    n_bins:
        Number of equally spaced confidence bins (default 15 per paper).

    Returns
    -------
    torch.Tensor
        Scalar ECE value.

    CRISP reference
    ---------------
    instruct.md §16: Standard ECE uses 15 equal-width bins.
    """
    p = probs.detach().reshape(-1).float()
    y = labels.detach().reshape(-1).float()
    n = p.numel()
    if n == 0:
        return torch.tensor(0.0, device=probs.device)

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=p.device)
    ece = torch.tensor(0.0, device=p.device)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i < n_bins - 1:
            in_bin = (p >= lo) & (p < hi)
        else:
            in_bin = (p >= lo) & (p <= hi)  # include right edge in last bin

        bin_count = in_bin.sum().float()
        if bin_count < 1:
            continue

        avg_confidence = p[in_bin].mean()
        avg_accuracy = y[in_bin].mean()
        ece += (bin_count / n) * (avg_confidence - avg_accuracy).abs()

    return ece


def boundary_support_mask(
    boundary_weight: torch.Tensor,
    top_percent: float = 20.0,
) -> torch.Tensor:
    """
    Build the per-image boundary support mask used for bECE and BA-ECE.

    Parameters
    ----------
    boundary_weight:
        Boundary weighting field w_b(u) [B, 1, H, W].
    top_percent:
        Percentage of highest-weight pixels retained in each image.

    Returns
    -------
    torch.Tensor
        Binary support mask [B, 1, H, W] selecting boundary-focused pixels.

    CRISP reference
    ---------------
    instruct.md §16: bECE restricts evaluation to the top 20% highest-wb pixels.
    """
    if not (0.0 < top_percent <= 100.0):
        raise ValueError(f"top_percent must be in (0, 100], got {top_percent}.")

    B = boundary_weight.shape[0]
    wb_flat = boundary_weight.reshape(B, -1)  # [B, N]
    k = max(1, int(wb_flat.shape[1] * top_percent / 100.0))

    # Select exact per-image top-k support rather than thresholding by value.
    _, topk_indices = wb_flat.topk(k, dim=1)
    mask = torch.zeros_like(wb_flat)
    mask.scatter_(1, topk_indices, 1.0)
    return mask.reshape(boundary_weight.shape)


def boundary_expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    boundary_weight: torch.Tensor,
    n_bins: int = 15,
    top_percent: float = 20.0,
) -> torch.Tensor:
    """
    Compute boundary-ECE restricted to the top boundary-support pixels.

    CRISP reference
    ---------------
    instruct.md §16: bECE restricts evaluation to top 20% highest-wb pixels
    and aggregates globally.
    """
    support = boundary_support_mask(boundary_weight, top_percent)
    mask_flat = support.reshape(-1).bool()

    if mask_flat.sum() == 0:
        return torch.tensor(0.0, device=probs.device)

    p_sel = probs.reshape(-1)[mask_flat]
    y_sel = labels.reshape(-1)[mask_flat]
    return expected_calibration_error(p_sel, y_sel, n_bins=n_bins)


def boundary_area_weighted_ece(
    probs: torch.Tensor,
    labels: torch.Tensor,
    boundary_weight: torch.Tensor,
    n_bins: int = 15,
    top_percent: float = 20.0,
) -> torch.Tensor:
    """
    Compute boundary-area weighted ECE (BA-ECE).

    Uses the same boundary support as bECE but reweights each bin by the
    local boundary mass (sum of w_b values in that bin) instead of raw count.

    CRISP reference
    ---------------
    instruct.md §16: BA-ECE uses same support but reweights bins by local boundary mass.
    """
    support = boundary_support_mask(boundary_weight, top_percent)
    mask_flat = support.reshape(-1).bool()

    if mask_flat.sum() == 0:
        return torch.tensor(0.0, device=probs.device)

    p_sel = probs.detach().reshape(-1)[mask_flat].float()
    y_sel = labels.detach().reshape(-1)[mask_flat].float()
    w_sel = boundary_weight.detach().reshape(-1)[mask_flat].float()

    total_weight = w_sel.sum()
    if total_weight < 1e-8:
        return torch.tensor(0.0, device=probs.device)

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=p_sel.device)
    ba_ece = torch.tensor(0.0, device=p_sel.device)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i < n_bins - 1:
            in_bin = (p_sel >= lo) & (p_sel < hi)
        else:
            in_bin = (p_sel >= lo) & (p_sel <= hi)

        if in_bin.sum() < 1:
            continue

        bin_mass = w_sel[in_bin].sum()
        # Guard degenerate bins where boundary mass is ~0 (possible with hard-band ablations).
        if bin_mass < 1e-12:
            continue
        avg_conf = (p_sel[in_bin] * w_sel[in_bin]).sum() / bin_mass
        avg_acc = (y_sel[in_bin] * w_sel[in_bin]).sum() / bin_mass
        ba_ece += (bin_mass / total_weight) * (avg_conf - avg_acc).abs()

    return ba_ece


def off_boundary_expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    boundary_weight: torch.Tensor,
    n_bins: int = 15,
    top_percent: float = 20.0,
) -> torch.Tensor:
    """
    Compute an off-boundary diagnostic ECE.

    This metric complements bECE by restricting evaluation to pixels *not* in the
    boundary support mask. It is useful for debugging the intended localization
    behavior of CRISP.

    CRISP reference
    ---------------
    instruct.md §16 emphasizes boundary-local calibration; this diagnostic verifies
    calibration does not degrade away from the boundary support.
    """
    support = boundary_support_mask(boundary_weight, top_percent)
    off = (~support.bool()).reshape(-1)
    if off.sum() == 0:
        return torch.tensor(0.0, device=probs.device)
    p_sel = probs.reshape(-1)[off]
    y_sel = labels.reshape(-1)[off]
    return expected_calibration_error(p_sel, y_sel, n_bins=n_bins)


def thresholded_adaptive_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 1.0e-3,
) -> torch.Tensor:
    """
    Compute TACE for dense binary prediction.

    Ignores near-zero-confidence predictions and uses adaptive (equal-count)
    binning on the remaining predictions.

    CRISP reference
    ---------------
    instruct.md §16: TACE uses thresholded confidence bins over non-background predictions.
    """
    p = probs.detach().reshape(-1).float()
    y = labels.detach().reshape(-1).float()

    # Filter out near-zero confidence.
    mask = p >= threshold
    if mask.sum() < 1:
        return torch.tensor(0.0, device=probs.device)

    p_sel = p[mask]
    y_sel = y[mask]

    # Adaptive (equal-count) binning with ~15 bins.
    n_bins = 15
    n = p_sel.numel()
    sorted_indices = p_sel.argsort()
    tace = torch.tensor(0.0, device=p_sel.device)

    bin_size = max(1, n // n_bins)
    for i in range(0, n, bin_size):
        idx = sorted_indices[i : i + bin_size]
        if len(idx) == 0:
            continue
        avg_conf = p_sel[idx].mean()
        avg_acc = y_sel[idx].mean()
        tace += (len(idx) / n) * (avg_conf - avg_acc).abs()

    return tace


def brier_score(
    probs: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Brier score for binary segmentation probabilities.

    Brier = mean((p - y)²)
    """
    return (probs.float() - labels.float()).pow(2).mean()


def negative_log_likelihood(
    probs: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """
    Compute binary negative log-likelihood.

    NLL = mean(-y log p - (1-y) log(1-p))
    """
    p = probs.float().clamp(eps, 1.0 - eps)
    y = labels.float()
    nll = -(y * p.log() + (1.0 - y) * (1.0 - p).log())
    return nll.mean()
