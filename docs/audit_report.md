# CRISP Audit Report

## Scope

This audit treats [`instruct.md`](/Users/minhnguyen/Desktop/CRISP_MIDL/instruct.md) as the implementation source of truth and checks the repository against the paper-defined CRISP identity for:

- Task A: Kvasir-SEG -> ColonDB / ETIS / PolypGen
- primary host path: PraNet
- core controls: baseline, CRISP, soft-label control, spatial-alpha control, boundary-posterior CE control

## Paper-Faithful Parts Already Correct

- Boundary weighting uses the soft Gaussian field `w_b(u) = exp(-d(u, ∂y)^2 / (2 sigma_b^2))` by default.
- Teacher barycenter uses frozen detached probabilities with entropy/agreement weighting.
- Boundary posterior target uses `t_star = (y + lambda * w_b * p_T) / (1 + lambda * w_b)`.
- The calibrated family remains scalar inverse-temperature scaling over logits.
- The local solver keeps explicit detached `alpha_star` and uses stabilized detached logits only for supervision.
- The forward path uses raw student logits `z`, not detached solver logits.
- Projector outputs stay bounded in `[alpha_min, alpha_max]`.
- CRISP loss keeps projected task fitting, identity regularization, Dice, and detached amortization supervision.
- Inference uses backbone + projector only, with projector-off implemented by forcing `alpha_hat = 1`.
- Boundary-local calibration metrics use per-image top-k support and global aggregation.

## Issues Found And Patched

### 1. Validation Could Still Be Silently Skipped

- Problem: [`src/crisp/scripts/train.py`](/Users/minhnguyen/Desktop/CRISP_MIDL/src/crisp/scripts/train.py) previously swallowed missing validation splits.
- Why this mattered: [`instruct.md`](/Users/minhnguyen/Desktop/CRISP_MIDL/instruct.md) §15 requires Kvasir validation for faithful model selection.
- Patch: paper-faithful experiment configs now set `training.require_validation: true`, and the train entry point fails loudly if the validation split cannot be built.

### 2. Teacher Pool Could Degrade Silently

- Problem: strict CRISP mode could still build an incomplete teacher ensemble when some checkpoint paths were empty.
- Why this mattered: the main paper-faithful CRISP path requires a frozen teacher pool and must not silently fall back to a reduced ensemble.
- Patch: [`src/crisp/scripts/train.py`](/Users/minhnguyen/Desktop/CRISP_MIDL/src/crisp/scripts/train.py) now enforces strict teacher completeness and surfaces checkpoint/config errors explicitly.

### 3. Checkpoints Missed Reproducibility State

- Problem: saved checkpoints did not include scheduler state, scaler state, or seed metadata.
- Why this mattered: reproducible resume/audit requires more than model and optimizer weights.
- Patch: [`src/crisp/engine/trainer.py`](/Users/minhnguyen/Desktop/CRISP_MIDL/src/crisp/engine/trainer.py) now stores scheduler state, GradScaler state, seed, config, and latest metrics. [`src/crisp/utils/serialization.py`](/Users/minhnguyen/Desktop/CRISP_MIDL/src/crisp/utils/serialization.py) also now saves a resolved YAML config snapshot.

### 4. Evaluation Could Skip Requested Target Datasets

- Problem: [`src/crisp/scripts/evaluate.py`](/Users/minhnguyen/Desktop/CRISP_MIDL/src/crisp/scripts/evaluate.py) previously skipped missing target datasets.
- Why this mattered: paper-faithful target-only evaluation should fail loudly when a requested benchmark target is unavailable.
- Patch: evaluation now raises by default on missing requested target datasets, with an explicit `eval.skip_missing_datasets` escape hatch for exploratory runs.

### 5. Trainer-Level Smoke Coverage Was Missing

- Problem: unit tests covered math components but not trainer-level one-step CRISP execution or strict entry-point discipline.
- Patch: new tests now cover one-step CRISP training, validation enforcement, strict teacher loading, and checkpoint metadata preservation.

## Remaining Partials

- [PAPER] Exact paper-number reproduction still depends on official host architectures and trained checkpoints.
- [INFERRED/RECOMMENDED] The stabilized solver constants (`eps_target`, `zmax`, `zeta`) follow a faithful numeric regime but are not spelled out in the paper.
- [PARTIAL] [`src/crisp/models/pranet.py`](/Users/minhnguyen/Desktop/CRISP_MIDL/src/crisp/models/pranet.py), [`src/crisp/models/polyp_pvt.py`](/Users/minhnguyen/Desktop/CRISP_MIDL/src/crisp/models/polyp_pvt.py), and [`src/crisp/models/rabbit.py`](/Users/minhnguyen/Desktop/CRISP_MIDL/src/crisp/models/rabbit.py) are CRISP-compatible wrappers, not bundled official upstream implementations.
- [PARTIAL] [`src/crisp/scripts/benchmark.py`](/Users/minhnguyen/Desktop/CRISP_MIDL/src/crisp/scripts/benchmark.py) is a lightweight engineering benchmark utility, not a full compute-reproduction script tied to the paper protocol.
- [PARTIAL] [`LICENSE`](/Users/minhnguyen/Desktop/CRISP_MIDL/LICENSE) is still blank.

## Risky Assumptions To Keep In Mind

- The primary scientific claim supported here is method-faithful CRISP integration, not exact leaderboard parity without official teacher/backbone checkpoints.
- Teacher diversity remains config-dependent; replacing the default diverse pool with same-family teachers changes the experiment.
- Post-hoc calibrator comparisons are source-validation-only by design and should stay separate from the CRISP method path.

## Replication Status

- Faithful for Task A / PraNet core path:
  explicit detached `alpha_star`, bounded amortized projector, raw-logit forward path, frozen teacher barycenter, paper-style boundary target, projector-on/off evaluation, and source-validation-driven training discipline.
- Partial:
  exact upstream host implementations/checkpoints, lightweight benchmark script, repository licensing.
- Not yet implemented:
  bundled official pretrained teacher assets and exact host-model reproduction packages.
