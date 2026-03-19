"""
Evaluation engine for geometry and calibration metrics.

This module handles:
- target-domain evaluation,
- optional projector-on / projector-off inference,
- dataset-level metric aggregation,
- prediction export and reproducibility metadata capture.

CRISP reference: instruct.md §14, §16.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from crisp.metrics.aggregation import average_metric_dicts
from crisp.metrics.calibration import (
    boundary_area_weighted_ece,
    boundary_expected_calibration_error,
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
    off_boundary_expected_calibration_error,
    thresholded_adaptive_calibration_error,
)
from crisp.metrics.segmentation import boundary_f1_score, dice_score, hd95_score
from crisp.modules.boundary import compute_boundary_weight
from crisp.modules.calibration import calibrate_logits_with_alpha
from crisp.utils.tensor_ops import threshold_mask

logger = logging.getLogger("crisp")


class Evaluator:
    """
    High-level evaluation engine.

    Parameters
    ----------
    model:
        Student segmentation model.
    projector:
        Optional CRISP projector head.
    config:
        Evaluation configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        projector: Optional[nn.Module],
        config: Dict[str, Any],
    ) -> None:
        self.model = model
        self.projector = projector
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if self.projector is not None:
            self.projector.to(self.device)

        # Boundary config for metric computation.
        crisp_cfg = config.get("crisp", {})
        bnd_cfg = crisp_cfg.get("boundary", {})
        self.sigma_b = bnd_cfg.get("sigma_b", 3.0)
        self.boundary_mode = bnd_cfg.get("mode", "gaussian_soft_field")
        self.top_percent = float(config.get("eval", {}).get("boundary_support", {}).get("top_percent", 20.0)) \
            if isinstance(config.get("eval", {}), dict) else 20.0
        self.ece_bins = int(config.get("eval", {}).get("ece", {}).get("bins", 15)) \
            if isinstance(config.get("eval", {}), dict) else 15
        self.tace_threshold = float(config.get("eval", {}).get("tace", {}).get("threshold", 1.0e-3)) \
            if isinstance(config.get("eval", {}), dict) else 1.0e-3

        proj_cfg = crisp_cfg.get("projection", {})
        self.alpha_min = proj_cfg.get("alpha_min", 0.50)
        self.alpha_max = proj_cfg.get("alpha_max", 1.80)

    @torch.no_grad()
    def predict_batch(
        self,
        batch: Dict[str, Any],
        projector_on: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict segmentation outputs for one batch.

        Parameters
        ----------
        batch:
            Batch dictionary from an evaluation dataloader.
        projector_on:
            Whether to apply the CRISP projector at inference time.
            If False, sets α̂ = 1 (projector-off ablation per §14).

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing logits, probabilities, predictions,
            and any optional diagnostic tensors.

        CRISP reference
        ---------------
        instruct.md §14:
          - Inference = student backbone + projector only, no teachers/solver.
          - Projector-off: α̂ = 1; projector-on: use learned projector.
          - Binary: ŷ(u) = 1{p̃(u) > 0.5}.
        """
        images = batch["image"].to(self.device)

        out = self.model(images)
        logits = out.logits
        features = out.features

        if projector_on and self.projector is not None:
            alpha_hat = self.projector(features, logits)
        else:
            alpha_hat = torch.ones_like(logits)

        probs = calibrate_logits_with_alpha(logits, alpha_hat)
        preds = threshold_mask(probs, 0.5)

        return {
            "logits": logits,
            "probs": probs,
            "preds": preds,
            "alpha_hat": alpha_hat,
        }

    @torch.no_grad()
    def evaluate_dataset(
        self,
        dataloader: Any,
        dataset_name: str,
        projector_on: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate one dataset and return aggregated metrics.

        Computes all geometry and calibration metrics per §16.
        """
        self.model.eval()
        if self.projector is not None:
            self.projector.eval()

        geometry_metrics: List[Dict[str, float]] = []
        all_probs: List[torch.Tensor] = []
        all_masks: List[torch.Tensor] = []
        all_wb: List[torch.Tensor] = []

        for batch in dataloader:
            results = self.predict_batch(batch, projector_on=projector_on)

            masks = batch["mask"].to(self.device)
            probs = results["probs"]
            preds = results["preds"]
            wb_batch = compute_boundary_weight(
                masks,
                sigma_b=self.sigma_b,
                mode=self.boundary_mode,
            )

            all_probs.append(probs.detach())
            all_masks.append(masks.detach())
            all_wb.append(wb_batch.detach())

            B = masks.shape[0]
            for i in range(B):
                pred_i = preds[i]    # [1, H, W]
                mask_i = masks[i]    # [1, H, W]

                # Geometry metrics.
                dice = dice_score(pred_i, mask_i).item()
                bf1 = boundary_f1_score(pred_i, mask_i).item()
                hd = hd95_score(pred_i, mask_i).item()

                geometry_metrics.append({
                    "dice": dice,
                    "boundary_f1": bf1,
                    "hd95": hd,
                })

        if not all_probs:
            raise ValueError(f"Evaluation dataset '{dataset_name}' is empty.")

        probs_all = torch.cat(all_probs, dim=0)
        masks_all = torch.cat(all_masks, dim=0)
        wb_all = torch.cat(all_wb, dim=0)

        calibration_metrics = {
            "ece": expected_calibration_error(
                probs_all,
                masks_all,
                n_bins=self.ece_bins,
            ).item(),
            "bece": boundary_expected_calibration_error(
                probs_all,
                masks_all,
                wb_all,
                n_bins=self.ece_bins,
                top_percent=self.top_percent,
            ).item(),
            "ba_ece": boundary_area_weighted_ece(
                probs_all,
                masks_all,
                wb_all,
                n_bins=self.ece_bins,
                top_percent=self.top_percent,
            ).item(),
            "off_bece": off_boundary_expected_calibration_error(
                probs_all,
                masks_all,
                wb_all,
                n_bins=self.ece_bins,
                top_percent=self.top_percent,
            ).item(),
            "tace": thresholded_adaptive_calibration_error(
                probs_all,
                masks_all,
                threshold=self.tace_threshold,
            ).item(),
            "brier": brier_score(probs_all, masks_all).item(),
            "nll": negative_log_likelihood(probs_all, masks_all).item(),
        }

        avg = average_metric_dicts(geometry_metrics)
        avg.update(calibration_metrics)
        logger.info("[%s] projector=%s  %s", dataset_name, projector_on, avg)
        return avg
