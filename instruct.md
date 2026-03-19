# instruct.md

## CRISP replication source of truth

This file is the implementation source of truth for replicating **CRISP: Amortized Boundary Posterior Projection for Robust and Calibrated Polyp Segmentation under Domain Shift**.

Its purpose is to let a coding assistant audit, refactor, and extend code **without drifting away from the paper-defined method**.

This document separates three categories of information:
- **[PAPER]**: directly specified in the paper.
- **[INFERRED]**: strongly implied by the paper but not fully pinned down.
- **[RECOMMENDED]**: engineering choices chosen to maximize faithfulness, stability, and reproducibility.

## 1. Method identity

- CRISP is **not** generic soft-label learning.
- CRISP is **not** generic spatial temperature scaling.
- CRISP is **not** a detached uncertainty head.
- CRISP is **not** a purely post-hoc calibrator.
- CRISP **is** a boundary-posterior projection method over a restricted calibrated family.
- CRISP **is** an amortized approximation to a per-pixel local projection optimum.
- CRISP **is** a train-time decision-margin correction mechanism with a lightweight test-time projector.
- The conceptual center of the method is the projection operator over inverse temperature.
- Any implementation that removes the explicit local projection target alpha_star is not faithful CRISP.
- Any implementation that trains alpha_hat only through segmentation loss, without approximating a projection optimum, weakens the method identity.
- Any implementation that replaces the restricted family with a generic unconstrained confidence head is method drift.
- When resolving ambiguity, preserve the projection interpretation before pursuing code simplification.

## 2. Notation and setting

- Input image: x in R^(H x W x 3).
- Binary mask: y in {0,1}^(H x W).
- Pixel domain: Omega = {1,...,H} x {1,...,W}.
- Student segmentation network: f_theta.
- Decoder features aligned to logit resolution: F_theta(x).
- Foreground logits: z_theta(x).
- Per-pixel raw logit: z(u).
- Per-pixel raw probability: p_theta(u) = sigmoid(z(u)).
- Training regime: source-only.
- Test regime: target-only under domain shift.

## 3. Boundary weighting field

### 3.1 Definition
- [PAPER] w_b(u) = exp( - d(u, ∂y)^2 / (2 sigma_b^2) ).
- [PAPER] d(u, ∂y) is Euclidean distance from pixel u to the annotation boundary.
- [PAPER] sigma_b > 0 controls spread.

### 3.2 Required behavior
- Boundary-near pixels must receive larger weights than far-interior pixels.
- The field must be soft and continuous by default.
- This is distance to boundary, not distance to foreground center.
- The weighting field must be regenerated after geometric augmentation if augmentations alter mask geometry.
- The field must share spatial resolution with the training logits used in loss computation.
- The same conceptual field also defines boundary support for boundary-restricted calibration metrics.

### 3.3 Recommended implementation
- [RECOMMENDED] Function name: build_boundary_weight_field(mask, sigma_b).
- [RECOMMENDED] Input shape: [B,1,H,W].
- [RECOMMENDED] Output shape: [B,1,H,W].
- [RECOMMENDED] Dtype: float32.
- [RECOMMENDED] Range: [0,1].
- [RECOMMENDED] Preferred implementation: compute boundary map, run Euclidean distance transform, then apply Gaussian weighting.
- [RECOMMENDED] Boundary map may be extracted using morphological gradient, XOR(mask, erode(mask)), or equivalent binary edge extraction.
- [RECOMMENDED] Caching is recommended when no post-cache geometry augmentation is used.
- [RECOMMENDED] For deterministic experiments, distance transform should follow a stable implementation path.

### 3.4 Default and ablations
- [PAPER] Default sigma_b = 3.0.
- [PAPER] Sensitivity table includes sigma_b in {2.0, 3.0, 4.0}.
- [PAPER] Hard band and logistic ramp are ablation boundary constructions, not the default method.
- [PAPER] Soft Gaussian boundary field outperforms hard band in the reported sensitivity and robustness analyses.

## 4. Teacher posterior

### 4.1 Teacher outputs
- [PAPER] Each teacher T_m outputs a foreground probability map p_m(u) in [0,1].
- [PAPER] Teachers are frozen during CRISP training.
- [PAPER] Teachers are trained on source data only.
- [PAPER] Teachers do not use target data for training or checkpoint selection.

### 4.2 Reliability-aware barycenter
- [PAPER] Mean teacher consensus: p_bar(u) = (1/M) sum_m p_m(u).
- [PAPER] Entropy: H(p) = -p log p - (1-p) log(1-p).
- [PAPER] Teacher weights: pi_m(u) proportional to exp( -tau H(p_m(u)) - gamma (p_m(u)-p_bar(u))^2 ).
- [PAPER] Normalized teacher weights sum to one across teachers.
- [PAPER] Teacher barycenter: p_T(u) = sum_m pi_m(u) p_m(u).

### 4.3 Implementation requirements
- [RECOMMENDED] Teacher probabilities must be detached before entropy and agreement weighting.
- [RECOMMENDED] Probability values should be clamped away from 0 and 1 before entropy computation.
- [RECOMMENDED] Function name: compute_teacher_barycenter(teacher_probs, tau, gamma, eps=1e-6).
- [RECOMMENDED] Input shape: [M,B,1,H,W].
- [RECOMMENDED] Output shape: [B,1,H,W].
- [RECOMMENDED] Default tau = 1.0 and gamma = 6.0.
- [RECOMMENDED] Online teacher recomputation from frozen checkpoints is faithful to Appendix B.
- [RECOMMENDED] For efficiency, precomputing teacher predictions is acceptable if preprocessing and augmentations preserve alignment assumptions.

### 4.4 Teacher pool defaults
- [PAPER] Default teacher pool for Task A contains PraNet, Polyp-PVT, and a stronger RaBiT checkpoint.
- [PAPER] For skin-lesion transfer, the paper uses UNet++, DeepLabV3+, and SegFormer-B0.
- [PAPER] Teacher diversity matters: three diverse teachers perform better than same-family or biased-pool variants.
- [PAPER] Teacher-free self-ensemble CRISP remains competitive but does not fully match the diverse teacher pool.

## 5. Boundary posterior target

- [PAPER] t_star(u) = ( y(u) + lambda w_b(u) p_T(u) ) / (1 + lambda w_b(u)).
- [PAPER] If w_b(u) is near 0, t_star(u) stays close to the original label.
- [PAPER] If w_b(u) is near 1, the target mixes label and teacher evidence.
- [PAPER] This is the exact boundary-posterior reduction implied by Proposition 1.
- [RECOMMENDED] Function name: build_boundary_posterior_target(mask, wb, pT, lam).
- [RECOMMENDED] Input shapes should all be [B,1,H,W].
- [RECOMMENDED] Output shape should be [B,1,H,W].
- [RECOMMENDED] Mask should be float before mixing with teacher probabilities.
- [PAPER] Default lambda = 0.80.

## 6. Restricted calibrated family

- [PAPER] Q(z(u)) = { sigmoid(alpha z(u)) : alpha in [alpha_min, alpha_max] }.
- [PAPER] 0 < alpha_min < 1 < alpha_max < infinity.
- [PAPER] alpha < 1 contracts the margin.
- [PAPER] alpha > 1 sharpens the margin.
- [PAPER] Default bounds are [0.50, 1.80].
- [PAPER] Narrower bounds such as [0.7, 1.4] increase clipping and worsen results.
- [PAPER] Wider bounds such as [0.4, 2.0] remain viable but are not clearly better than the default.
- [PAPER] Richer monotone families are ablation objects, not the main method.
- [RECOMMENDED] The main implementation must keep scalar inverse temperature per pixel as the default family.
- [RECOMMENDED] Affine-logit or piecewise monotone variants should live under ablation configs only.

## 7. Local projection problem

- [PAPER] alpha_star(u) = argmin over alpha in [alpha_min, alpha_max] of L_proj^(u)(alpha).
- [PAPER] L_proj^(u)(alpha) = (1 + lambda w_b(u)) * ell(alpha; z(u), t_star(u)) + mu (1 - w_b(u)) (alpha - 1)^2.
- [PAPER] ell(alpha; z, t) = -t log sigmoid(alpha z) - (1-t) log(1-sigmoid(alpha z)).
- [PAPER] The first term fits the boundary posterior target within the restricted calibrated family.
- [PAPER] The second term preserves identity away from the boundary.
- [PAPER] This local projection operator is the core mathematical object of CRISP.
- [PAPER] Proposition 2 establishes strong convexity of the stabilized local problem.
- [PAPER] Proposition 3 establishes single-valuedness and Lipschitz stability of the projection map on the compact stabilized domain.
- [RECOMMENDED] The implementation should preserve this exact 1D objective even if the solver routine is optimized or vectorized.
- [RECOMMENDED] Never replace alpha_star supervision with a hand-crafted heuristic temperature target.

## 8. Stabilization and detached solver branch

- [PAPER] Target clipping: t_eps(u) = clip(t_star(u), eps, 1-eps).
- [PAPER] Detached solver logit clipping: z_clip(u) = clip(z(u), -Z_max, Z_max).
- [PAPER] Stabilized solver logit: z_tilde(u) = sign(z_clip(u)) * max(|z_clip(u)|, zeta).
- [PAPER] z_tilde is used only in the local solver.
- [PAPER] The forward prediction path still uses raw z(u).
- [PAPER] alpha_star is treated with stop-gradient and is never backpropagated through.

### 8.1 Recommended explicit numeric defaults
- [INFERRED/RECOMMENDED] eps_target = 1e-4.
- [INFERRED/RECOMMENDED] z_max = 12.0.
- [INFERRED/RECOMMENDED] zeta = 1e-2.
- [INFERRED/RECOMMENDED] These values are not explicitly specified in the paper, but they are a reasonable faithful stabilization regime.
- [INFERRED/RECOMMENDED] eps_target should be small enough to minimally disturb targets while avoiding degenerate logit values.
- [INFERRED/RECOMMENDED] z_max should be large enough to preserve practical logit dynamic range while keeping Newton updates stable.
- [INFERRED/RECOMMENDED] zeta should prevent division by near-zero logits in the seed and avoid unstable supervision around zero-confidence pixels.

## 9. Projection derivative and solver

- [PAPER] g(alpha; z_tilde, t, w) = (1 + lambda w) (sigmoid(alpha z_tilde) - t) z_tilde + 2 mu (1-w) (alpha - 1).
- [PAPER] alpha_star is the unique root of g(alpha)=0 over the feasible interval.
- [PAPER] Closed-form seed without identity regularization: alpha0 = clamp(logit(t_eps)/z_tilde, alpha_min, alpha_max).
- [PAPER] The paper recommends safeguarded Newton or a short bisection routine.
- [PAPER] Because the problem is one-dimensional and strongly convex after stabilization, the solver is stable and inexpensive.

### 9.1 Recommended solver implementation
- [RECOMMENDED] Function name: solve_local_alpha_star(z_tilde, t_eps, wb, alpha_min, alpha_max, mu, lam, max_newton_steps=3, max_bisect_steps=12).
- [RECOMMENDED] Vectorize over all pixels where possible.
- [RECOMMENDED] Initialize from the clipped no-regularization seed.
- [RECOMMENDED] Perform a few Newton steps with derivative safeguarding.
- [RECOMMENDED] If a Newton proposal exits the interval or becomes NaN/Inf, switch that location to derivative-root bisection.
- [RECOMMENDED] Clamp the final result to [alpha_min, alpha_max].
- [RECOMMENDED] Return a detached tensor.
- [RECOMMENDED] Log saturation rates at both alpha bounds for diagnostics and bound-tuning heuristics.

## 10. Amortized projector head

- [PAPER] alpha_hat(u) = alpha_min + (alpha_max-alpha_min) * sigmoid( a_phi(F_theta(x)(u), z(u)) ).
- [PAPER] The calibrated probability is p_tilde(u) = sigmoid(alpha_hat(u) * z(u)).
- [PAPER] The projector is a lightweight prediction head.
- [PAPER] The projector is a two-layer 3x3 conv block with 64 hidden channels, GroupNorm, GELU, and a final 1x1 conv.
- [PAPER] It predicts at quarter resolution and is bilinearly upsampled.
- [PAPER] The role of the projector is not to learn arbitrary spatial scaling but to approximate the projection optimum alpha_star.

### 10.1 Recommended implementation structure
- [RECOMMENDED] Class name: CRISPProjectorHead.
- [RECOMMENDED] Input: decoder features and logits.
- [RECOMMENDED] Recommended fusion: concatenate quarter-resolution feature map with downsampled logits.
- [RECOMMENDED] Output: single-channel alpha map before sigmoid-affine remapping.
- [RECOMMENDED] Upsample to full logit resolution with bilinear interpolation.
- [RECOMMENDED] Apply explicit bound mapping to enforce [alpha_min, alpha_max].
- [RECOMMENDED] Add a simple runtime assertion in debug mode that alpha_hat.min() >= alpha_min - tol and alpha_hat.max() <= alpha_max + tol.

## 11. Gradient view

- [PAPER] d/dz BCE(p_tilde(u), t_eps(u)) = alpha_hat(u) * (p_tilde(u) - t_eps(u)).
- [PAPER] Therefore the projector modifies both the local target and the local gradient magnitude.
- [PAPER] This is why CRISP can improve geometry even though positive inverse-temperature scaling alone is threshold invariant for fixed logits.
- [PAPER] Projector-off-at-test-time ablation should preserve most geometry gains and lose more calibration than geometry.
- [RECOMMENDED] If projector-off ablation destroys most geometry gains, inspect whether training truly used p_tilde = sigmoid(alpha_hat * z) in the task loss.

## 12. Training objective

- [PAPER] L_task = mean_u [ (1 + lambda w_b(u)) BCE(p_tilde(u), t_eps(u)) + mu (1-w_b(u))(alpha_hat(u)-1)^2 ] + eta L_Dice(p_tilde, y).
- [PAPER] L_Dice(p_tilde, y) = 1 - (2 sum_u p_tilde(u) y(u) + eps_d) / (sum_u p_tilde(u) + sum_u y(u) + eps_d).
- [PAPER] L_amort = mean_u [ rho(u) ( alpha_hat(u) - sg[alpha_star(u)] )^2 ].
- [PAPER] rho(u) = w_b(u) * 1{|z_bar(u)| >= zeta}.
- [PAPER] Final loss: L_CRISP = L_task + beta L_amort.
- [PAPER] Default eta = 0.50 and beta = 0.20.
- [PAPER] Default mu = 0.05.
- [PAPER] Default lambda = 0.80.

### 12.1 Practical interpretation of rho
- [INFERRED/RECOMMENDED] The paper uses a boundary-focused confidence mask to avoid fitting unstable detached optima in near-zero-logit regions.
- [INFERRED/RECOMMENDED] A clean implementation is rho = wb * (abs(z_tilde) >= zeta).float().
- [INFERRED/RECOMMENDED] This interpretation is consistent with the stabilized detached solver branch described in the method section.

## 13. Required detach boundaries

- [RECOMMENDED] Teacher outputs must be detached.
- [RECOMMENDED] Teacher barycenter must be detached.
- [RECOMMENDED] The local solver branch must be detached from the student gradient graph.
- [RECOMMENDED] alpha_star must be detached.
- [RECOMMENDED] Raw logits z in the forward path must not be replaced with detached solver logits.
- [RECOMMENDED] alpha_hat must keep gradients.
- [RECOMMENDED] Features feeding alpha_hat must keep gradients.
- [RECOMMENDED] BCE and Dice through p_tilde must keep gradients.

## 14. Inference

- [PAPER] At inference, teachers are not used.
- [PAPER] At inference, no per-pixel optimization is solved.
- [PAPER] Inference consists of student backbone + amortized projector only.
- [PAPER] Binary prediction is y_hat(u) = 1{ p_tilde(u) > 0.5 }.
- [PAPER] For the explicit projector-off ablation, set alpha_hat(u) = 1 at test time while keeping the trained backbone fixed.
- [PAPER] The projector-on path further improves boundary-local calibration and yields smaller additional geometry gains.

## 15. Experimental defaults for polyp segmentation

- [PAPER] Train on Kvasir-SEG.
- [PAPER] Evaluate on ColonDB, ETIS, and PolypGen.
- [PAPER] Hyperparameters are selected only on a Kvasir validation split.
- [PAPER] Center-wise and patient-wise separation are preserved for PolypGen.
- [PAPER] Hosts: U-Net, PraNet, RaBiT, Polyp-PVT.
- [PAPER] Input size 352x352 for U-Net, PraNet, Polyp-PVT.
- [PAPER] Input size 384x384 for RaBiT.
- [PAPER] Optimizer: AdamW.
- [PAPER] Student learning rate = 1e-4.
- [PAPER] Projector learning rate = 2e-4.
- [PAPER] Batch size = 16.
- [PAPER] Weight decay = 1e-4.
- [PAPER] Cosine decay schedule.
- [PAPER] Warm up lambda, mu, and beta linearly over the first 15 epochs.
- [PAPER] Default hyperparameters: sigma_b=3.0, lambda=0.80, mu=0.05, eta=0.50, beta=0.20, alpha bounds [0.50,1.80], tau=1.0, gamma=6.0.

## 16. Metrics

- [PAPER] Standard ECE uses 15 equal-width bins.
- [PAPER] Boundary-ECE restricts evaluation to the top 20% highest-wb pixels in each image and aggregates globally.
- [PAPER] BA-ECE uses the same support but reweights bins by local boundary mass.
- [PAPER] TACE uses thresholded confidence bins over non-background predictions.
- [PAPER] Main geometry metrics include Dice, Boundary-F1, and HD95.
- [PAPER] Additional scalar calibration metrics include Brier and NLL.
- [PAPER] The paper includes a sensitivity sweep over top 10%, 20%, and 30% boundary support on PraNet/ColonDB.
- [RECOMMENDED] Evaluation code should use the same wb construction logic as training, not a separate arbitrary edge detector.
- [RECOMMENDED] Export per-image and aggregated metric files for debugging support selection bugs.

## 17. Expected qualitative reproduction behavior

- [PAPER/RECOMMENDED] CRISP should improve bECE more strongly than Dice in many settings.
- [PAPER/RECOMMENDED] Boundary-F1 and HD95 gains should be clearer than global overlap gains.
- [PAPER/RECOMMENDED] Improvements should be stronger under larger shift, especially ETIS and PolypGen.
- [PAPER/RECOMMENDED] CRISP should outperform distance-relaxed soft labels only and spatial-alpha only on the core causal-isolation comparison.
- [PAPER/RECOMMENDED] Projector-off test-time ablation should retain most geometry gains while losing some boundary-local calibration.
- [PAPER/RECOMMENDED] Soft Gaussian boundary field should outperform hard band in the default setting.
- [PAPER/RECOMMENDED] Too narrow alpha bounds should increase clipping and weaken performance.
- [PAPER/RECOMMENDED] Moderate mu should outperform both mu=0 and over-regularized settings.
- [PAPER/RECOMMENDED] Teacher diversity should help more than using multiple near-identical teachers.
- [PAPER/RECOMMENDED] Compute overhead at inference should remain close to single-model inference relative to deep ensemble and MC Dropout baselines.
