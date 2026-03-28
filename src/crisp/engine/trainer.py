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

CRISP reference: instruct.md §12, §13, §15.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from crisp.engine.checkpointing import save_checkpoint
from crisp.modules.boundary import compute_boundary_weight
from crisp.modules.calibration import calibrate_logits_with_alpha
from crisp.modules.losses import (
    baseline_bce_dice_loss,
    crisp_amortization_loss,
    crisp_task_loss,
    crisp_total_loss,
)
from crisp.modules.posterior_target import (
    clip_posterior_target,
    compute_boundary_posterior_target,
)
from crisp.modules.solver import solve_alpha_star
from crisp.modules.teacher_posterior import aggregate_teacher_posterior
from crisp.utils.logging import log_metrics

logger = logging.getLogger("crisp")


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
    ) -> None:
        self.model = model
        self.projector = projector
        self.teacher_ensemble = teacher_ensemble
        self.config = config

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
        self.epochs = train_cfg.get("epochs", 100)
        self.mixed_precision = train_cfg.get("mixed_precision", False)
        self.gradient_clip_norm = train_cfg.get("gradient_clip_norm", 0.0)
        self.require_validation = bool(train_cfg.get("require_validation", False))
        self.optimizer_name = str(train_cfg.get("optimizer", "adamw")).lower()
        self.scheduler_name = str(train_cfg.get("scheduler", "cosine")).lower()

        # CRISP hyperparameters.
        crisp_cfg = config.get("crisp", {})
        proj_cfg = crisp_cfg.get("projection", {})
        self.lambda_value = proj_cfg.get("lambda", 0.80)
        self.mu_value = proj_cfg.get("mu", 0.05)
        self.beta_value = proj_cfg.get("beta", 0.20)
        self.eta_dice = proj_cfg.get("eta_dice", 0.50)
        self.alpha_min = proj_cfg.get("alpha_min", 0.50)
        self.alpha_max = proj_cfg.get("alpha_max", 1.80)
        self.eps_target = proj_cfg.get("eps_target", 1e-4)
        self.zeta = proj_cfg.get("zeta", 1e-2)
        self.zmax = proj_cfg.get("zmax", 12.0)

        # Boundary config.
        bnd_cfg = crisp_cfg.get("boundary", {})
        self.sigma_b = bnd_cfg.get("sigma_b", 3.0)
        self.boundary_mode = bnd_cfg.get("mode", "gaussian_soft_field")

        # Teacher config.
        teacher_cfg = crisp_cfg.get("teacher", {})
        self.tau = teacher_cfg.get("tau", 1.0)
        self.gamma = teacher_cfg.get("gamma", 6.0)
        self.strict_teacher_requirement = teacher_cfg.get("strict", True)

        # Solver config.
        solver_cfg = crisp_cfg.get("solver", {})
        self.newton_steps = solver_cfg.get("newton_steps", 2)
        self.bisection_steps = solver_cfg.get("bisection_steps", 8)

        # Warmup config.
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

    def build_optimizer(self) -> torch.optim.Optimizer:
        """
        Build AdamW optimizer with separate lr for student and projector (§15).
        """
        if self.optimizer_name != "adamw":
            raise ValueError(
                f"Unsupported optimizer '{self.optimizer_name}'. "
                "The paper-faithful implementation currently supports only AdamW."
            )
        train_cfg = self.config.get("training", {})
        lr_student = train_cfg.get("lr_student", 1e-4)
        lr_projector = train_cfg.get("lr_projector", 2e-4)
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
            "Use `cosine` for the paper default or `none` for explicit ablations."
        )

    def train_one_step(
        self, batch: Dict[str, Any], epoch: int, step: int
    ) -> TrainStepOutput:
        """
        Run one training step with full CRISP pipeline or baseline.

        CRISP reference: instruct.md §12 (full objective), §13 (detach boundaries).
        """
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

            if not self.use_crisp:
                # Baseline: standard BCE + Dice.
                loss_dict = baseline_bce_dice_loss(logits, masks)
                return TrainStepOutput(
                    loss=loss_dict["loss"],
                    logs={k: v.item() for k, v in loss_dict.items()},
                )

            # --- CRISP pipeline ---
            wf = self._warmup_factor(epoch)

            # 1. Boundary weighting field.
            wb = compute_boundary_weight(masks, sigma_b=self.sigma_b, mode=self.boundary_mode)
            wb = wb.to(self.device)

            # 2. Target construction.
            lam = self.lambda_value * wf
            if self.target_mode == "boundary_posterior":
                if self.use_teachers and self.teacher_ensemble is not None:
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
            mu_w = self.mu_value * wf
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
                beta_w = self.beta_value * wf
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
        logs.update({f"solver/{k}": v.item() for k, v in solver_diag.items()})

        return TrainStepOutput(loss=total_dict["loss"], logs=logs)

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
    ) -> Dict[str, Any]:
        """
        Build a reproducibility-focused checkpoint payload.

        This preserves the exact composed config, optimizer/scheduler/scaler
        states, and the run seed so checkpoints remain resumable and auditable.
        """
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
        }

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
        if self.require_validation and val_loader is None:
            raise ValueError(
                "This experiment requires a source-validation loader for paper-faithful "
                "checkpoint selection, but no validation loader was provided."
            )

        for epoch in range(self.epochs):
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

                monitor = val_metrics.get("dice", 0.0)
                if monitor > best_val_metric:
                    best_val_metric = monitor
                    save_checkpoint(
                        f"{output_dir}/best.pt",
                        self._checkpoint_state(
                            epoch=epoch,
                            train_metrics=avg_logs,
                            val_metrics=val_metrics,
                            best_val_metric=best_val_metric,
                        ),
                    )

            # Periodic checkpoint.
            if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                save_checkpoint(
                    f"{output_dir}/epoch_{epoch + 1}.pt",
                    self._checkpoint_state(
                        epoch=epoch,
                        train_metrics=avg_logs,
                        best_val_metric=best_val_metric,
                    ),
                )
