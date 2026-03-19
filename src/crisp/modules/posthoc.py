"""
Post-hoc calibration utilities.

The paper compares CRISP against several post-hoc calibration baselines
such as temperature scaling, local temperature scaling, and selective scaling. [file:1]

This file reserves the module boundary for those methods so they can be added
without polluting the core CRISP implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from crisp.metrics.calibration import boundary_support_mask


class TemperatureScaler:
    """
    Simple global temperature scaling calibrator.

    This class is intended for baseline comparison only and should be fit
    on source-validation data to match the paper's source-only setting.
    """

    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 200,
    ) -> None:
        """
        Fit the scalar temperature on validation logits and labels.

        Uses L-BFGS to minimize NLL on the validation set.

        Parameters
        ----------
        logits:
            Flattened or batched foreground logits.
        labels:
            Corresponding binary labels (same shape as logits).
        lr:
            Learning rate for optimization.
        max_iter:
            Maximum iterations for L-BFGS.
        """
        # Create a learnable temperature parameter.
        log_temp = torch.nn.Parameter(torch.zeros(1, device=logits.device))

        optimizer = torch.optim.LBFGS([log_temp], lr=lr, max_iter=max_iter)

        flat_logits = logits.detach().reshape(-1)
        flat_labels = labels.detach().float().reshape(-1)

        def closure():
            optimizer.zero_grad()
            temp = log_temp.exp()  # ensure positive temperature
            scaled_logits = flat_logits / temp
            loss = F.binary_cross_entropy_with_logits(
                scaled_logits, flat_labels, reduction="mean"
            )
            loss.backward()
            return loss

        optimizer.step(closure)

        self.temperature = log_temp.exp().item()

    def transform(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply the learned temperature and return calibrated probabilities.

        Parameters
        ----------
        logits:
            Raw foreground logits.

        Returns
        -------
        torch.Tensor
            Calibrated probabilities after temperature scaling.
        """
        return torch.sigmoid(logits / self.temperature)


@dataclass
class PostHocFitArtifacts:
    """
    Optional structured artifacts from post-hoc fitting.

    These artifacts are intended for reproducibility/debugging exports and should
    be saved alongside source-only validation results.
    """

    method: str
    params: Dict[str, float]


class BoundaryTemperatureScaler(TemperatureScaler):
    """
    Boundary Temperature Scaling (bTS).

    Fits a single global temperature using only boundary-support pixels defined
    by the CRISP boundary support mask (top-k w_b per image).

    Protocol invariants
    -------------------
    - Fit must be performed on *source validation only* (no target leakage).
    - Transformation is post-hoc on frozen logits; it must not affect training.
    """

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        boundary_weight: torch.Tensor,
        top_percent: float = 20.0,
        lr: float = 0.01,
        max_iter: int = 200,
    ) -> PostHocFitArtifacts:
        support = boundary_support_mask(boundary_weight.detach(), top_percent=top_percent).bool()
        flat_logits = logits.detach().reshape(-1)[support.reshape(-1)]
        flat_labels = labels.detach().float().reshape(-1)[support.reshape(-1)]
        if flat_logits.numel() == 0:
            # Degenerate: fall back to identity temperature.
            self.temperature = 1.0
            return PostHocFitArtifacts(method="boundary_ts", params={"temperature": self.temperature})

        # Learnable log-temperature.
        log_temp = torch.nn.Parameter(torch.zeros(1, device=logits.device))
        optimizer = torch.optim.LBFGS([log_temp], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            temp = log_temp.exp()
            loss = F.binary_cross_entropy_with_logits(flat_logits / temp, flat_labels, reduction="mean")
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = log_temp.exp().item()
        return PostHocFitArtifacts(method="boundary_ts", params={"temperature": self.temperature})


class SelectiveTemperatureScaler(TemperatureScaler):
    """
    Selective scaling baseline.

    Applies temperature scaling only to pixels whose *foreground probability*
    exceeds a threshold; other pixels remain unscaled. This is a minimal, stable
    variant suitable for dense binary prediction.

    Protocol invariants
    -------------------
    - Fit on source validation only.
    - Post-hoc transformation of frozen logits only.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = float(threshold)

    def transform(self, logits: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        scale_mask = (p >= self.threshold).float()
        scaled = torch.sigmoid(logits / self.temperature)
        # Leave unscaled pixels at original probability.
        return scale_mask * scaled + (1.0 - scale_mask) * p


class LocalTemperatureScaler:
    """
    Local Temperature Scaling (LTS) via a small number of spatial bins.

    This is a minimal faithful implementation for dense prediction: it fits K
    temperatures over disjoint pixel partitions derived from the boundary weight
    field (e.g., boundary-near vs off-boundary, or multi-quantile bins).

    This is *not* CRISP and must remain post-hoc:
    - no effect on training,
    - fit on source validation only,
    - evaluated on frozen logits.
    """

    def __init__(self, n_bins: int = 2) -> None:
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2 for local temperature scaling.")
        self.n_bins = int(n_bins)
        self.temperatures = torch.ones(self.n_bins)

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        boundary_weight: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 200,
    ) -> PostHocFitArtifacts:
        # Bin pixels by boundary_weight quantiles (per-image) then aggregate globally.
        wb = boundary_weight.detach().reshape(boundary_weight.shape[0], -1)  # [B, N]
        qs = torch.linspace(0.0, 1.0, self.n_bins + 1, device=wb.device)
        # Per-image quantile thresholds.
        thr = torch.quantile(wb, qs, dim=1)  # [n_bins+1, B]

        flat_logits = logits.detach().reshape(logits.shape[0], -1)  # [B, N]
        flat_labels = labels.detach().float().reshape(labels.shape[0], -1)  # [B, N]

        temps = []
        for b in range(self.n_bins):
            lo = thr[b].unsqueeze(1)
            hi = thr[b + 1].unsqueeze(1)
            if b < self.n_bins - 1:
                in_bin = (wb >= lo) & (wb < hi)
            else:
                in_bin = (wb >= lo) & (wb <= hi)

            l_bin = flat_logits[in_bin]
            y_bin = flat_labels[in_bin]
            if l_bin.numel() == 0:
                temps.append(torch.tensor(1.0, device=logits.device))
                continue

            log_temp = torch.nn.Parameter(torch.zeros(1, device=logits.device))
            opt = torch.optim.LBFGS([log_temp], lr=lr, max_iter=max_iter)

            def closure():
                opt.zero_grad()
                temp = log_temp.exp()
                loss = F.binary_cross_entropy_with_logits(l_bin / temp, y_bin, reduction="mean")
                loss.backward()
                return loss

            opt.step(closure)
            temps.append(log_temp.exp().detach().reshape(()))

        self.temperatures = torch.stack(temps).detach().cpu()
        return PostHocFitArtifacts(
            method="local_ts",
            params={f"temperature_bin_{i}": float(t.item()) for i, t in enumerate(self.temperatures)},
        )

    def transform(self, logits: torch.Tensor, boundary_weight: torch.Tensor) -> torch.Tensor:
        wb = boundary_weight.detach().reshape(boundary_weight.shape[0], -1)  # [B, N]
        qs = torch.linspace(0.0, 1.0, self.n_bins + 1, device=wb.device)
        thr = torch.quantile(wb, qs, dim=1)  # [n_bins+1, B]

        flat_logits = logits.reshape(logits.shape[0], -1)
        out = torch.sigmoid(flat_logits)  # default identity

        temps = self.temperatures.to(logits.device)
        for b in range(self.n_bins):
            lo = thr[b].unsqueeze(1)
            hi = thr[b + 1].unsqueeze(1)
            if b < self.n_bins - 1:
                in_bin = (wb >= lo) & (wb < hi)
            else:
                in_bin = (wb >= lo) & (wb <= hi)
            if in_bin.any():
                out[in_bin] = torch.sigmoid(flat_logits[in_bin] / temps[b])

        return out.reshape(logits.shape)


class HistogramBinningCalibrator:
    """
    Histogram binning calibration (a minimal BBQ-style baseline).

    Fits a piecewise-constant mapping from predicted probability to calibrated
    probability using equal-width bins on [0,1]. This is a lightweight baseline
    that approximates Bayesian binning schemes in a stable, reproducible way.

    Protocol invariants
    -------------------
    - Fit on source validation only.
    - Applies post-hoc to frozen probabilities/logits only.
    """

    def __init__(self, n_bins: int = 15, eps: float = 1e-6) -> None:
        self.n_bins = int(n_bins)
        self.eps = float(eps)
        self.bin_edges = torch.linspace(0.0, 1.0, self.n_bins + 1)
        self.bin_values = torch.full((self.n_bins,), 0.5)

    def fit(self, probs: torch.Tensor, labels: torch.Tensor) -> PostHocFitArtifacts:
        p = probs.detach().reshape(-1).clamp(self.eps, 1.0 - self.eps)
        y = labels.detach().float().reshape(-1)
        edges = self.bin_edges.to(p.device)
        values = []
        for i in range(self.n_bins):
            lo, hi = edges[i], edges[i + 1]
            in_bin = (p >= lo) & (p < hi) if i < self.n_bins - 1 else (p >= lo) & (p <= hi)
            if in_bin.sum() < 1:
                values.append(torch.tensor(0.5, device=p.device))
            else:
                values.append(y[in_bin].mean())
        self.bin_values = torch.stack(values).detach().cpu()
        return PostHocFitArtifacts(
            method="histogram_binning",
            params={f"bin_{i}": float(v.item()) for i, v in enumerate(self.bin_values)},
        )

    def transform(self, probs: torch.Tensor) -> torch.Tensor:
        p = probs.detach().clone().float()
        edges = self.bin_edges.to(p.device)
        vals = self.bin_values.to(p.device)
        out = torch.empty_like(p)
        flat = p.reshape(-1)
        flat_out = out.reshape(-1)
        for i in range(self.n_bins):
            lo, hi = edges[i], edges[i + 1]
            in_bin = (flat >= lo) & (flat < hi) if i < self.n_bins - 1 else (flat >= lo) & (flat <= hi)
            flat_out[in_bin] = vals[i]
        return out
