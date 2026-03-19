# Experiments Guide

This document maps each config file to a replication target.

- `taskA_pranet_baseline.yaml`: baseline binary segmentation on Kvasir-SEG.
- `taskA_pranet_crisp.yaml`: full CRISP training on PraNet with frozen teacher barycenter, boundary posterior target, projector, and detached `alpha_star` amortization.
- `taskA_pranet_spatial_alpha.yaml`: spatial-alpha control with projector and identity regularization, but no teacher posterior target and no amortization target.
- `taskA_pranet_softlabels.yaml`: soft-label control using the teacher-driven boundary posterior target without projector or identity regularization.
- `taskA_pranet_boundary_posterior_ce.yaml`: boundary-posterior CE control using teacher-driven target softening and CRISP boundary weighting, but no amortized projector.

The ablation interpretation encoded here is conservative: it preserves CRISP’s calibration family and boundary localization while removing the specific component under test.
