"""
Training engine for baseline and CRISP methods.

This module orchestrates:
- dataloaders,
- backbone and projector forward passes,
- teacher execution,
- boundary and target construction,
- detached projection solving,
- loss computation,
- optimization and checkpointing.

The trainer is designed to keep high-level control flow explicit and debuggable.
"""

from __future__ import annotations

import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from crisp.data.bwcr_views import (
    BWCRAugmentationContract,
    BWCRSourceViewPair,
    bwcr_validity_intersection,
    inverse_align_bwcr_tensor,
    sample_bwcr_pair,
)
from crisp.engine.checkpointing import save_checkpoint
from crisp.modules.boundary import compute_boundary_weight
from crisp.modules.bwcr_consistency import (
    bwcr_consistency_loss,
    native_bwcr_boundary_field,
)
from crisp.modules.bwcr_control import resolve_bwcr_control
from crisp.modules.calibration import calibrate_logits_with_alpha
from crisp.modules.losses import (
    baseline_bce_dice_loss,
    crisp_amortization_loss,
    crisp_task_loss,
    crisp_total_loss,
    pranet_native_side_losses,
)
from crisp.modules.margin_label_smoothing import (
    margin_label_smoothing_penalty,
    resolve_margin_label_smoothing_control,
)
from crisp.models.pranet import PraNet
from crisp.modules.posterior_target import (
    clip_posterior_target,
    compute_boundary_posterior_target,
)
from crisp.modules.solver import solve_alpha_star
from crisp.modules.teacher_posterior import aggregate_teacher_posterior, teacher_robustness_posterior
from crisp.utils.logging import log_metrics
from crisp.utils.provenance import checkpoint_provenance, run_provenance

logger = logging.getLogger("crisp")

BOUNDARY_F1_WINDOW = 0.002
SELECTION_METRIC = "validation_boundary_f1_within_0.002_then_bece_then_dice_then_earliest_epoch"


@dataclass(frozen=True)
class _ValidationCandidate:
    epoch: int
    boundary_f1: float
    bece: float
    dice: float


@dataclass
class TrainStepOutput:
    """
    Structured output returned by one training step.

    Attributes
    ----------
    loss:
        Scalar tensor used for backpropagation.
    logs:
        Dictionary of scalar logging values.
    tensors:
        Optional dictionary of intermediate tensors for debugging.
    """

    loss: torch.Tensor
    logs: Dict[str, float]
    tensors: Optional[Dict[str, torch.Tensor]] = None


@dataclass(frozen=True)
class BWCRTrainingBatch:
    """Two keyed source views plus canonical tensors and replay records."""

    canonical_images: torch.Tensor
    canonical_masks: torch.Tensor
    view1_images: torch.Tensor
    view1_masks: torch.Tensor
    view2_images: torch.Tensor
    pairs: tuple[BWCRSourceViewPair, ...]


def _metadata_sequence(
    metadata: Dict[str, Any], key: str, batch_size: int
) -> tuple[str, ...]:
    values = metadata.get(key)
    if not isinstance(values, (list, tuple)) or len(values) != batch_size:
        raise ValueError(f"BWCR requires one stable {key} value per source sample.")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"BWCR requires non-empty string {key} values.")
    return tuple(values)


def prepare_bwcr_training_batch(
    batch: Dict[str, Any],
    *,
    contract: BWCRAugmentationContract,
    experiment_seed: int,
    epoch: int,
    device: torch.device,
) -> BWCRTrainingBatch:
    """Create the sole two stochastic views from canonical source tensors."""

    canonical_images = batch.get("canonical_image")
    canonical_masks = batch.get("canonical_mask")
    if not isinstance(canonical_images, torch.Tensor) or not isinstance(
        canonical_masks, torch.Tensor
    ):
        raise ValueError("BWCR requires canonical_image and canonical_mask tensors.")
    if canonical_images.ndim != 4 or canonical_masks.ndim != 4:
        raise ValueError("BWCR canonical tensors must have batched BCHW shapes.")
    if canonical_images.shape[0] != canonical_masks.shape[0]:
        raise ValueError("BWCR canonical image/mask batch sizes must match.")

    metadata = batch.get("meta")
    if not isinstance(metadata, dict):
        raise ValueError("BWCR requires collated source metadata.")
    batch_size = canonical_images.shape[0]
    dataset_names = _metadata_sequence(metadata, "dataset_name", batch_size)
    image_ids = _metadata_sequence(metadata, "image_id", batch_size)
    splits = _metadata_sequence(metadata, "split", batch_size)
    representations = _metadata_sequence(
        metadata, "source_representation", batch_size
    )
    if any(split != "train" for split in splits):
        raise ValueError("BWCR is restricted to source-training samples.")
    if any(value != "canonical_pre_stochastic" for value in representations):
        raise ValueError("BWCR refuses source tensors that may already be augmented.")

    canonical_images = canonical_images.to(device)
    canonical_masks = canonical_masks.to(device)
    pairs = tuple(
        sample_bwcr_pair(
            canonical_images[index],
            canonical_masks[index],
            contract,
            experiment_seed=experiment_seed,
            epoch=epoch,
            dataset_name=dataset_names[index],
            image_id=image_ids[index],
        )
        for index in range(batch_size)
    )
    return BWCRTrainingBatch(
        canonical_images=canonical_images,
        canonical_masks=canonical_masks,
        view1_images=torch.stack([pair.view_0.image for pair in pairs]),
        view1_masks=torch.stack([pair.view_0.mask for pair in pairs]),
        view2_images=torch.stack([pair.view_1.image for pair in pairs]),
        pairs=pairs,
    )


def _training_output_with_native_aux(
    final_loss: torch.Tensor,
    logs: Dict[str, float],
    native_aux: Optional[Dict[str, torch.Tensor]],
) -> TrainStepOutput:
    if native_aux is None:
        return TrainStepOutput(loss=final_loss, logs=logs)
    total = final_loss + native_aux["native_aux_loss"]
    logs["final_loss"] = final_loss.item()
    logs.update({key: value.item() for key, value in native_aux.items()})
    logs["loss"] = total.item()
    return TrainStepOutput(loss=total, logs=logs)


class Trainer:
    """
    High-level training engine.

    Responsibilities
    ----------------
    - initialize optimizer and scheduler,
    - run epoch loops,
    - dispatch method-specific branches,
    - handle checkpointing and validation,
    - expose hooks for experiment logging.

    Parameters
    ----------
    model:
        Student segmentation model.
    projector:
        Optional CRISP projector head.
    teacher_ensemble:
        Optional teacher ensemble used only during CRISP training.
    config:
        Experiment configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        projector: Optional[nn.Module],
        teacher_ensemble: Optional[nn.Module],
        config: Dict[str, Any],
        run_record: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.projector = projector
        self.teacher_ensemble = teacher_ensemble
        self.config = config
        self.run_record = run_record
        self.margin_label_smoothing = resolve_margin_label_smoothing_control(config)
        self.bwcr_control = resolve_bwcr_control(config)
        if self.margin_label_smoothing is not None:
            if projector is not None or teacher_ensemble is not None:
                raise ValueError(
                    "Margin Label Smoothing cannot receive a projector or teacher ensemble."
                )
        if self.bwcr_control is not None:
            if projector is not None or teacher_ensemble is not None:
                raise ValueError("BWCR cannot receive a projector or teacher ensemble.")

        # Determine device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_device_type = "cuda" if self.device.type == "cuda" else "cpu"
        self.model.to(self.device)
        if self.projector is not None:
            self.projector.to(self.device)
        if self.teacher_ensemble is not None:
            self.teacher_ensemble.to(self.device)

        # Method flags.
        method = config.get("method", {})
        self.use_crisp = method.get("use_crisp", False)
        self.method_name = str(
            method.get("name") or ("crisp" if self.use_crisp else "baseline")
        )
        self.use_projector = method.get("use_projector", False)
        self.use_teachers = method.get("use_teachers", False)
        self.use_amortization_loss = method.get(
            "use_amortization_loss",
            self.use_crisp and self.use_projector,
        )
        self.target_mode = method.get(
            "target_mode",
            "boundary_posterior" if self.use_crisp else "hard_label",
        )
        self.use_boundary_weighted_task = method.get(
            "use_boundary_weighted_task",
            self.use_crisp,
        )
        self.use_identity_regularization = method.get(
            "use_identity_regularization",
            self.use_crisp and self.use_projector,
        )
        self.allow_self_ensemble_teacher = method.get(
            "allow_self_ensemble_teacher",
            False,
        )

        # Training config.
        train_cfg = config.get("training", {})
        self.seed = int(config.get("seed", 0))
        self.epochs = int(train_cfg.get("total_epochs", train_cfg.get("epochs", 100)))
        self.mixed_precision = train_cfg.get("mixed_precision", False)
        self.gradient_clip_norm = train_cfg.get("gradient_clip_norm", 0.0)
        self.require_validation = bool(train_cfg.get("require_validation", False))
        self.optimizer_name = str(train_cfg.get("optimizer", "adamw")).lower()
        self.scheduler_name = str(train_cfg.get("scheduler", "cosine")).lower()

        # CRISP hyperparameters.
        crisp_cfg = config.get("crisp", {})
        proj_cfg = crisp_cfg.get("projection", {})
        self.lambda_value = proj_cfg.get("lambda", 1.0)
        self.mu_value = proj_cfg.get("mu", 0.25)
        self.beta_value = proj_cfg.get("beta", 0.35)
        self.eta_dice = proj_cfg.get("eta_dice", 0.50)
        self.alpha_min = proj_cfg.get("alpha_min", 0.50)
        self.alpha_max = proj_cfg.get("alpha_max", 1.75)
        self.eps_target = proj_cfg.get("eps_target", 1e-3)
        self.zeta = proj_cfg.get("zeta", 0.10)
        self.zmax = proj_cfg.get("zmax", 8.0)

        # Boundary config.
        bnd_cfg = crisp_cfg.get("boundary", {})
        self.sigma_b = bnd_cfg.get("sigma_b", 6.0)
        self.boundary_mode = bnd_cfg.get("mode", "gaussian_soft_field")

        # Teacher config.
        teacher_cfg = crisp_cfg.get("teacher", {})
        self.tau = teacher_cfg.get("tau", 1.0)
        self.gamma = teacher_cfg.get("gamma", 1.5)
        self.strict_teacher_requirement = teacher_cfg.get("strict", True)
        robustness_cfg = teacher_cfg.get("robustness", {})
        self.teacher_robustness_enabled = bool(robustness_cfg.get("enabled", False))
        self.teacher_robustness_mode = robustness_cfg.get("aggregation", "weighted")
        self.teacher_robustness_std = float(robustness_cfg.get("logit_noise_std", 1.0))
        if self.teacher_robustness_enabled:
            if not self.use_teachers or self.target_mode != "boundary_posterior":
                raise ValueError("Teacher robustness requires boundary-posterior teachers.")
            if [entry.get("name") for entry in config.get("teacher_pool", {}).get("teachers", [])] != [
                "uacanet_l", "polyp_pvt", "sammamba",
            ]:
                raise ValueError("Teacher robustness requires UACANet-L, Polyp-PVT, SAM-Mamba in order.")
            if teacher_cfg.get("teacher_names") != ["uacanet_l", "polyp_pvt", "sammamba"]:
                raise ValueError("Teacher robustness names must match its three-teacher pool.")
            if robustness_cfg.get("degraded_teacher") != "sammamba" \
                    or robustness_cfg.get("distribution") != "gaussian" \
                    or robustness_cfg.get("seed_keys") != ["seed", "dataset_name", "image_id"]:
                raise ValueError("Teacher robustness corruption/identity config is not manuscript-aligned.")
            if self.teacher_robustness_mode not in {"weighted", "equal_average"}:
                raise ValueError("Unknown teacher robustness aggregation mode.")
            if self.teacher_robustness_std != 1.0:
                raise ValueError("Teacher robustness requires logit-noise std=1.0.")

        # Solver config.
        solver_cfg = crisp_cfg.get("solver", {})
        self.newton_steps = solver_cfg.get("newton_steps", 3)
        self.bisection_steps = solver_cfg.get("bisection_steps", 12)

        # Current CRISP schedule config. When absent, keep legacy warmup behavior for
        # focused unit tests and explicit debug configs.
        schedule_cfg = crisp_cfg.get("schedule", {})
        phases_cfg = train_cfg.get("phases", {})
        self.schedule_enabled = bool(schedule_cfg.get("enabled", bool(phases_cfg)))
        self.phase_i_epochs = int(
            phases_cfg.get("baseline_warmup", schedule_cfg.get("phase_i_epochs", 25))
        )
        self.phase_ii_epochs = int(
            phases_cfg.get("crisp_full", schedule_cfg.get("phase_ii_epochs", 65))
        )
        self.phase_iii_epochs = int(
            phases_cfg.get("finetune", schedule_cfg.get("phase_iii_epochs", 30))
        )
        self.phase_ii_ramp_epochs = int(
            phases_cfg.get("phase2_ramp_epochs", schedule_cfg.get("phase_ii_ramp_epochs", 10))
        )

        # Legacy warmup config.
        warmup_cfg = crisp_cfg.get("warmup", {})
        self.warmup_enabled = warmup_cfg.get("enabled", True)
        self.warmup_epochs = warmup_cfg.get("epochs", 15)

        # Mixed precision scaler.
        self.scaler = torch.amp.GradScaler(
            self.amp_device_type,
            enabled=self.mixed_precision and self.amp_device_type == "cuda",
        )

        if self.use_teachers and self.teacher_ensemble is None and self.strict_teacher_requirement:
            raise ValueError(
                "Configuration requests frozen teachers, but no teacher ensemble was built. "
                "Provide teacher checkpoints or disable teacher-based targets explicitly."
            )
        if self.use_amortization_loss and not self.use_projector:
            raise ValueError(
                "Amortization loss requires a learnable projector. Disable "
                "`use_amortization_loss` for projector-free ablations."
            )
        if self.target_mode not in {"hard_label", "boundary_posterior"}:
            raise ValueError(f"Unknown target_mode '{self.target_mode}'.")

    def _warmup_factor(self, epoch: int) -> float:
        """Linear warmup factor for λ, μ, β in early epochs (§15)."""
        if not self.warmup_enabled or epoch >= self.warmup_epochs:
            return 1.0
        return float(epoch + 1) / float(self.warmup_epochs)

    def _crisp_schedule_state(self, epoch: int) -> Dict[str, Any]:
        """
        Return the current CRISP schedule state for one epoch.

        Updated experimental contract:
        - Phase I: baseline-only warm-up for 25 epochs.
        - Phase II: CRISP active for 65 epochs with lambda/beta ramp over the
          first 10 epochs of this phase.
        - Phase III: full CRISP joint fine-tuning for 30 epochs.
        """
        if not self.schedule_enabled:
            factor = self._warmup_factor(epoch)
            return {
                "phase": "legacy",
                "phase_id": 0,
                "crisp_active": True,
                "lambda_factor": factor,
                "mu_factor": factor,
                "beta_factor": factor,
            }

        if epoch < self.phase_i_epochs:
            return {
                "phase": "phase_i_baseline",
                "phase_id": 1,
                "crisp_active": False,
                "lambda_factor": 0.0,
                "mu_factor": 1.0,
                "beta_factor": 0.0,
            }

        phase_ii_end = self.phase_i_epochs + self.phase_ii_epochs
        if epoch < phase_ii_end:
            phase_epoch = epoch - self.phase_i_epochs
            ramp_epochs = max(1, self.phase_ii_ramp_epochs)
            ramp = min(1.0, float(phase_epoch + 1) / float(ramp_epochs))
            return {
                "phase": "phase_ii_crisp",
                "phase_id": 2,
                "crisp_active": True,
                "lambda_factor": ramp,
                "mu_factor": 1.0,
                "beta_factor": ramp,
            }

        return {
            "phase": "phase_iii_finetune",
            "phase_id": 3,
            "crisp_active": True,
            "lambda_factor": 1.0,
            "mu_factor": 1.0,
            "beta_factor": 1.0,
        }

    def build_optimizer(self) -> torch.optim.Optimizer:
        """
        Build AdamW optimizer with separate lr for student and projector (§15).
        """
        if self.optimizer_name != "adamw":
            raise ValueError(
                f"Unsupported optimizer '{self.optimizer_name}'. "
                "The current CRISP protocol supports only AdamW."
            )
        train_cfg = self.config.get("training", {})
        lr_student = train_cfg.get("lr_student", 1e-4)
        lr_projector = train_cfg.get("lr_projector", 5e-4)
        wd = train_cfg.get("weight_decay", 1e-4)

        param_groups = [
            {"params": self.model.parameters(), "lr": lr_student},
        ]
        if self.projector is not None:
            param_groups.append(
                {"params": self.projector.parameters(), "lr": lr_projector}
            )

        return torch.optim.AdamW(param_groups, weight_decay=wd)

    def build_scheduler(self, optimizer: torch.optim.Optimizer) -> Any:
        """
        Build the configured learning-rate scheduler (§15).
        """
        if self.scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs, eta_min=1e-6
            )
        if self.scheduler_name in {"none", "off", "disabled"}:
            return None
        raise ValueError(
            f"Unsupported scheduler '{self.scheduler_name}'. "
            "Use `cosine` for the current protocol or `none` for explicit ablations."
        )

    def _train_bwcr_step(
        self, batch: Dict[str, Any], epoch: int
    ) -> TrainStepOutput:
        """Run the standalone source-only BWCR control at full fixed strength."""

        if self.bwcr_control is None:
            raise RuntimeError("BWCR step requested without a resolved control.")
        prepared = prepare_bwcr_training_batch(
            batch,
            contract=self.bwcr_control.augmentation,
            experiment_seed=self.seed,
            epoch=epoch,
            device=self.device,
        )

        with torch.amp.autocast(
            device_type=self.amp_device_type,
            enabled=self.mixed_precision and self.amp_device_type == "cuda",
        ):
            view1_output = self.model(prepared.view1_images)
            view2_output = self.model(prepared.view2_images)

            final_losses = baseline_bce_dice_loss(
                view1_output.logits, prepared.view1_masks
            )
            final_loss = final_losses["loss"]
            native_aux = (
                pranet_native_side_losses(view1_output.aux, prepared.view1_masks)
                if isinstance(self.model, PraNet)
                else None
            )
            supervised_host_loss = final_loss
            if native_aux is not None:
                supervised_host_loss = supervised_host_loss + native_aux[
                    "native_aux_loss"
                ]

            z1_inverse = torch.stack(
                [
                    inverse_align_bwcr_tensor(
                        view1_output.logits[index],
                        pair.view_0.record.geometry,
                        interpolation="bilinear",
                    )
                    for index, pair in enumerate(prepared.pairs)
                ]
            )
            z2_inverse = torch.stack(
                [
                    inverse_align_bwcr_tensor(
                        view2_output.logits[index],
                        pair.view_1.record.geometry,
                        interpolation="bilinear",
                    )
                    for index, pair in enumerate(prepared.pairs)
                ]
            )
            validity = torch.stack(
                [bwcr_validity_intersection(pair) for pair in prepared.pairs]
            )
            boundary_field = native_bwcr_boundary_field(prepared.canonical_masks)
            consistency = bwcr_consistency_loss(
                z1_inverse,
                z2_inverse,
                validity,
                boundary_field.lambda_map,
            )
            total = supervised_host_loss + consistency

        logs = {key: value.item() for key, value in final_losses.items()}
        logs["final_loss"] = final_loss.item()
        if native_aux is not None:
            logs.update({key: value.item() for key, value in native_aux.items()})
        logs.update(
            {
                "supervised_host_loss": supervised_host_loss.item(),
                "bwcr_consistency_loss": consistency.item(),
                "loss": total.item(),
            }
        )
        return TrainStepOutput(
            loss=total,
            logs=logs,
            tensors={
                "bwcr_view1_image": prepared.view1_images,
                "bwcr_view1_mask": prepared.view1_masks,
                "bwcr_view2_image": prepared.view2_images,
                "bwcr_z1_inverse": z1_inverse,
                "bwcr_z2_inverse": z2_inverse,
                "bwcr_validity": validity,
                "bwcr_lambda_map": boundary_field.lambda_map,
                "bwcr_consistency_loss": consistency,
                "supervised_host_loss": supervised_host_loss,
            },
        )

    def train_one_step(
        self, batch: Dict[str, Any], epoch: int, step: int
    ) -> TrainStepOutput:
        """
        Run one training step with the selected explicit method/control branch.
        """
        if self.bwcr_control is not None:
            return self._train_bwcr_step(batch, epoch)

        images = batch["image"].to(self.device)
        masks = batch["mask"].to(self.device)

        with torch.amp.autocast(
            device_type=self.amp_device_type,
            enabled=self.mixed_precision and self.amp_device_type == "cuda",
        ):
            # --- Student forward ---
            out = self.model(images)
            logits = out.logits       # [B,1,H,W] — raw z, keeps gradients
            features = out.features   # decoder features for projector
            native_aux = (
                pranet_native_side_losses(out.aux, masks)
                if isinstance(self.model, PraNet) else None
            )

            if not self.use_crisp:
                # Baseline: standard BCE + Dice.
                loss_dict = baseline_bce_dice_loss(logits, masks)
                final_loss = loss_dict["loss"]
                logs = {k: v.item() for k, v in loss_dict.items()}
                if self.margin_label_smoothing is not None:
                    margin_penalty = margin_label_smoothing_penalty(
                        logits, self.margin_label_smoothing.margin
                    )
                    weighted_margin = self.margin_label_smoothing.weight * margin_penalty
                    final_loss = final_loss + weighted_margin
                    logs.update({
                        "margin_penalty": margin_penalty.item(),
                        "weighted_margin_penalty": weighted_margin.item(),
                        "loss": final_loss.item(),
                    })
                return _training_output_with_native_aux(
                    final_loss,
                    logs,
                    native_aux,
                )

            # --- CRISP pipeline ---
            schedule = self._crisp_schedule_state(epoch)
            if not schedule["crisp_active"]:
                loss_dict = baseline_bce_dice_loss(logits, masks)
                logs = {k: v.item() for k, v in loss_dict.items()}
                logs.update({
                    "schedule/phase_id": float(schedule["phase_id"]),
                    "schedule/lambda_factor": float(schedule["lambda_factor"]),
                    "schedule/mu_factor": float(schedule["mu_factor"]),
                    "schedule/beta_factor": float(schedule["beta_factor"]),
                })
                return _training_output_with_native_aux(loss_dict["loss"], logs, native_aux)

            # 1. Boundary weighting field.
            wb = compute_boundary_weight(masks, sigma_b=self.sigma_b, mode=self.boundary_mode)
            wb = wb.to(self.device)

            # 2. Target construction.
            lam = self.lambda_value * float(schedule["lambda_factor"])
            teacher_diagnostics = None
            if self.target_mode == "boundary_posterior":
                if self.use_teachers and self.teacher_ensemble is not None:
                    if self.teacher_robustness_enabled:
                        meta = batch.get("meta", {})
                        ids = meta.get("image_id")
                        datasets = meta.get("dataset_name")
                        if not isinstance(ids, (list, tuple)) or not isinstance(datasets, (list, tuple)) \
                                or len(ids) != images.shape[0] or len(datasets) != images.shape[0] \
                                or any(not isinstance(value, str) or not value for value in (*ids, *datasets)):
                            raise ValueError("Teacher robustness requires stable dataset_name/image_id per image.")
                        stable_ids = [f"{dataset}/{image_id}" for dataset, image_id in zip(datasets, ids)]
                        teacher_logits = self.teacher_ensemble.forward_logits(images)
                        pT, teacher_diagnostics = teacher_robustness_posterior(
                            teacher_logits,
                            stable_ids,
                            self.seed,
                            mode=self.teacher_robustness_mode,
                            tau=self.tau,
                            gamma=self.gamma,
                            std=self.teacher_robustness_std,
                        )
                    else:
                        teacher_probs = self.teacher_ensemble(images)
                        pT, _ = aggregate_teacher_posterior(
                            teacher_probs,
                            tau=self.tau,
                            gamma=self.gamma,
                        )
                elif self.allow_self_ensemble_teacher:
                    pT = torch.sigmoid(logits.detach())
                else:
                    raise ValueError(
                        "boundary_posterior target requested without a frozen teacher ensemble. "
                        "Set `allow_self_ensemble_teacher=true` only for explicit teacher-free ablations."
                    )
                t_star = compute_boundary_posterior_target(
                    masks,
                    wb,
                    pT.detach(),
                    lambda_value=lam,
                )
            else:
                t_star = masks.float()
            t_eps = clip_posterior_target(t_star, eps_target=self.eps_target)

            # 4. Projector forward or identity alpha.
            if self.use_projector and self.projector is not None:
                alpha_hat = self.projector(features, logits)
            else:
                alpha_hat = torch.ones_like(logits)

            # 5. Calibrated probability.
            p_tilde = calibrate_logits_with_alpha(logits, alpha_hat)

            # 6. Task loss.
            mu_w = self.mu_value * float(schedule["mu_factor"])
            task_dict = crisp_task_loss(
                p_tilde,
                t_eps,
                wb,
                alpha_hat,
                masks,
                lambda_value=lam,
                mu_value=mu_w,
                eta_dice=self.eta_dice,
                apply_boundary_weight=self.use_boundary_weighted_task,
                apply_identity_regularization=self.use_identity_regularization,
            )

            solver_diag: Dict[str, torch.Tensor] = {}
            if self.use_amortization_loss:
                # 7. Detached local solver for alpha*.
                alpha_star, solver_diag = solve_alpha_star(
                    logits,
                    t_eps,
                    wb,
                    lambda_value=lam,
                    mu_value=mu_w,
                    alpha_min=self.alpha_min,
                    alpha_max=self.alpha_max,
                    zmax=self.zmax,
                    zeta=self.zeta,
                    newton_steps=self.newton_steps,
                    bisection_steps=self.bisection_steps,
                )

                # 8. Amortization loss.
                beta_w = self.beta_value * float(schedule["beta_factor"])
                amort_dict = crisp_amortization_loss(
                    alpha_hat,
                    alpha_star,
                    wb,
                    logits,
                    zeta=self.zeta,
                    zmax=self.zmax,
                )

                # 9. Total CRISP loss.
                total_dict = crisp_total_loss(task_dict, amort_dict, beta_value=beta_w)
            else:
                total_dict = {"loss": task_dict["task_loss"], **task_dict}

        logs = {k: v.item() for k, v in total_dict.items() if v.ndim == 0}
        if teacher_diagnostics is not None:
            logs["teacher/degraded_mean_weight"] = teacher_diagnostics[
                "degraded_mean_weight"
            ].item()
        logs.update({f"solver/{k}": v.item() for k, v in solver_diag.items()})
        logs.update({
            "schedule/phase_id": float(schedule["phase_id"]),
            "schedule/lambda_factor": float(schedule["lambda_factor"]),
            "schedule/mu_factor": float(schedule["mu_factor"]),
            "schedule/beta_factor": float(schedule["beta_factor"]),
        })

        return _training_output_with_native_aux(total_dict["loss"], logs, native_aux)

    def train_one_epoch(self, dataloader: Any, epoch: int) -> Dict[str, float]:
        """Train for one epoch and aggregate logging statistics."""
        self.model.train()
        if self.projector is not None:
            self.projector.train()

        optimizer = self._optimizer
        all_logs: list[Dict[str, float]] = []

        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            output = self.train_one_step(batch, epoch, step)

            self.scaler.scale(output.loss).backward()
            if self.gradient_clip_norm > 0:
                self.scaler.unscale_(optimizer)
                params = list(self.model.parameters())
                if self.projector is not None:
                    params += list(self.projector.parameters())
                nn.utils.clip_grad_norm_(params, self.gradient_clip_norm)
            self.scaler.step(optimizer)
            self.scaler.update()

            all_logs.append(output.logs)

        if self._scheduler is not None:
            self._scheduler.step()

        # Aggregate logs.
        from crisp.metrics.aggregation import average_metric_dicts
        avg = average_metric_dicts(all_logs)
        return avg

    def _checkpoint_state(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        best_val_metric: Optional[float] = None,
        best_boundary_f1: Optional[float] = None,
        best_bece: Optional[float] = None,
        best_epoch: Optional[int] = None,
        selection_metric: str = SELECTION_METRIC,
    ) -> Dict[str, Any]:
        """
        Build a reproducibility-focused checkpoint payload.

        This preserves the exact composed config, optimizer/scheduler/scaler
        states, and the run seed so checkpoints remain resumable and auditable.
        """
        if self.run_record is None:
            self.run_record = run_provenance(self.config)
        selection = {
            "metric": selection_metric,
            "best_val_metric": best_val_metric,
            "best_boundary_f1": best_boundary_f1,
            "best_bece": best_bece,
            "best_epoch": best_epoch,
        }
        return {
            "epoch": epoch,
            "seed": self.seed,
            "model_state_dict": self.model.state_dict(),
            "projector_state_dict": (
                self.projector.state_dict() if self.projector is not None else None
            ),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "scheduler_state_dict": (
                self._scheduler.state_dict() if self._scheduler is not None else None
            ),
            "grad_scaler_state_dict": self.scaler.state_dict(),
            "config": self.config,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_val_metric": best_val_metric,
            "best_boundary_f1": best_boundary_f1,
            "best_bece": best_bece,
            "best_epoch": best_epoch,
            "selection_metric": selection_metric,
            "provenance": checkpoint_provenance(self.run_record, epoch, selection),
        }

    @staticmethod
    def _validation_candidate(epoch: int, val_metrics: Dict[str, float]) -> _ValidationCandidate:
        """Read and validate the three source-validation selection metrics."""
        def metric(name: str, aliases: tuple[str, ...]) -> float:
            for key in aliases:
                if key in val_metrics:
                    value = float(val_metrics[key])
                    if not math.isfinite(value):
                        raise ValueError(f"Epoch {epoch} validation {name} is non-finite: {value}")
                    return value
            raise ValueError(f"Epoch {epoch} validation {name} is missing")

        return _ValidationCandidate(
            epoch=epoch,
            boundary_f1=metric("boundary_f1", ("boundary_f1", "B-F1", "bf1")),
            bece=metric("bece", ("bece", "bECE")),
            dice=metric("dice", ("dice", "mDice")),
        )

    @staticmethod
    def _select_validation_candidate(candidates: Sequence[_ValidationCandidate]) -> _ValidationCandidate:
        """Select from the global B-F1 window, then bECE, Dice, and epoch."""
        if not candidates:
            raise ValueError("Checkpoint selection requires validation candidates")
        b_max = max(candidate.boundary_f1 for candidate in candidates)
        eligible = (
            candidate for candidate in candidates
            if candidate.boundary_f1 >= b_max - BOUNDARY_F1_WINDOW
        )
        return min(eligible, key=lambda candidate: (candidate.bece, -candidate.dice, candidate.epoch))

    def fit(
        self, train_loader: Any, val_loader: Optional[Any] = None
    ) -> None:
        """
        Run the full training loop across all epochs.

        Parameters
        ----------
        train_loader:
            Training dataloader.
        val_loader:
            Optional validation dataloader used for checkpoint selection.
        """
        self._optimizer = self.build_optimizer()
        self._scheduler = self.build_scheduler(self._optimizer)

        output_dir = self.config.get("output_dir", "outputs")
        best_val_metric = -float("inf")
        best_boundary_f1: Optional[float] = None
        best_bece: Optional[float] = None
        best_epoch: Optional[int] = None
        selection_metric = SELECTION_METRIC
        candidates: list[_ValidationCandidate] = []
        candidate_paths: dict[int, Path] = {}
        if self.require_validation and val_loader is None:
            raise ValueError(
                "This experiment requires a source-validation loader for current-protocol "
                "checkpoint selection, but no validation loader was provided."
            )
        if val_loader is not None and (Path(output_dir) / "best.pt").exists():
            raise FileExistsError(f"Selection checkpoint already exists: {Path(output_dir) / 'best.pt'}")
        if self.run_record is None:
            self.run_record = run_provenance(
                self.config, train_dataset=getattr(train_loader, "dataset", None)
            )

        train_batches = len(train_loader) if hasattr(train_loader, "__len__") else "unknown"
        val_batches = len(val_loader) if (val_loader is not None and hasattr(val_loader, "__len__")) else 0
        logger.info(
            "Starting training: device=%s method=%s epochs=%d train_batches=%s val_batches=%s output_dir=%s",
            self.device,
            self.method_name,
            self.epochs,
            train_batches,
            val_batches,
            output_dir,
        )

        for epoch in range(self.epochs):
            schedule = self._crisp_schedule_state(epoch) if self.use_crisp else {
                "phase": "baseline",
                "phase_id": 0,
                "crisp_active": False,
                "lambda_factor": 0.0,
                "mu_factor": 0.0,
                "beta_factor": 0.0,
            }
            logger.info(
                "Starting epoch %d/%d phase=%s lambda_factor=%.4f beta_factor=%.4f",
                epoch + 1,
                self.epochs,
                schedule["phase"],
                float(schedule["lambda_factor"]),
                float(schedule["beta_factor"]),
            )
            avg_logs = self.train_one_epoch(train_loader, epoch)
            logger.info("Epoch %d/%d — %s", epoch + 1, self.epochs, avg_logs)
            log_metrics(avg_logs, epoch, "train")

            # Validation (if loader provided).
            if val_loader is not None:
                from crisp.engine.evaluator import Evaluator
                evaluator = Evaluator(self.model, self.projector, self.config)
                val_metrics = evaluator.evaluate_dataset(
                    val_loader, "val", projector_on=self.use_projector,
                )
                log_metrics(val_metrics, epoch, "val")

                candidate = self._validation_candidate(epoch, val_metrics)
                candidates.append(candidate)
                b_max = max(item.boundary_f1 for item in candidates)
                if candidate.boundary_f1 >= b_max - BOUNDARY_F1_WINDOW:
                    candidate_path = Path(output_dir) / f"selection_epoch_{epoch + 1}.pt"
                    if candidate_path.exists():
                        raise FileExistsError(f"Selection candidate already exists: {candidate_path}")
                    save_checkpoint(
                        candidate_path,
                        self._checkpoint_state(
                            epoch=epoch,
                            train_metrics=avg_logs,
                            val_metrics=val_metrics,
                            best_val_metric=candidate.boundary_f1,
                            best_boundary_f1=candidate.boundary_f1,
                            best_bece=candidate.bece,
                            best_epoch=candidate.epoch,
                            selection_metric=selection_metric,
                        ),
                    )
                    candidate_paths[epoch] = candidate_path

                selected = self._select_validation_candidate(candidates)
                if selected.epoch != best_epoch:
                    best_boundary_f1 = selected.boundary_f1
                    best_bece = selected.bece
                    best_epoch = selected.epoch
                    best_val_metric = selected.boundary_f1
                    shutil.copyfile(candidate_paths[selected.epoch], Path(output_dir) / "best.pt")

            # Periodic checkpoint.
            if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                save_checkpoint(
                    f"{output_dir}/epoch_{epoch + 1}.pt",
                    self._checkpoint_state(
                        epoch=epoch,
                        train_metrics=avg_logs,
                        best_val_metric=best_val_metric,
                        best_boundary_f1=best_boundary_f1,
                        best_bece=best_bece,
                        best_epoch=best_epoch,
                        selection_metric=selection_metric,
                    ),
                )
