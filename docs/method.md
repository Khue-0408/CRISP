# Method

This document maps the CRISP method objects to implementation modules.

## Core Objects

- Student logits: `z(u)` -> `SegmentationOutput.logits`
- Decoder features: `F_theta(x)` -> `SegmentationOutput.features`
- Boundary weight: `w_b(u)` -> `crisp.modules.boundary.compute_boundary_weight`
- Teacher posterior: `p_T(u)` -> `crisp.modules.teacher_posterior.aggregate_teacher_posterior`
- Boundary posterior target: `t*(u)` -> `crisp.modules.posterior_target.compute_boundary_posterior_target`
- Detached local optimum: `alpha*(u)` -> `crisp.modules.solver.solve_alpha_star`
- Amortized projector: `alpha_hat(u)` -> `crisp.models.projector_head.CRISPProjectorHead`
- Calibrated probability: `p_tilde(u)` -> `crisp.modules.calibration.calibrate_logits_with_alpha`

## Training Loss Decomposition

- Task fitting
- Identity regularization
- Dice regularization
- Amortization consistency

## Numerical Invariants

- `alpha_star` is solved only on detached stabilized logits `z_tilde`.
- The forward path always uses raw student logits `z`.
- `alpha_hat` is explicitly bounded in `[alpha_min, alpha_max]`.
- Teacher probabilities and barycenters are detached.

## Evaluation

- Segmentation geometry metrics
- Boundary-local calibration metrics on per-image top-k `w_b` support, aggregated globally across the dataset
