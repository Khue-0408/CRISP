# CRISP Journal Development Goals

## Scientific objective

CRISP studies boundary-local constrained posterior projection with amortized inference for reliable dense prediction under cross-dataset shift.

The implementation should support the journal manuscript without turning the repository into a paper-number hard-coding exercise.

## Core method identity

Stable conceptual components are:

- annotation-derived soft boundary weighting;
- train-time teacher consensus posterior;
- boundary-local posterior target;
- bounded positive inverse-temperature family;
- one-dimensional constrained local projection;
- detached stabilized solver target;
- amortized projector;
- teacher- and solver-free deployment;
- fixed-logit threshold invariance under positive scaling.

Manuscript equations and locked hyperparameters remain in the manuscript and canonical experiment configuration rather than being duplicated here.

## Experimental philosophy

The repository should support retained student baselines, canonical CRISP variants, controlled ablations, calibration controls, robustness diagnostics, stronger-host scope tests, and reproducible multi-seed evaluation.

Configuration does not imply execution.

## Evidence chain

Scientific claim
→ declared experiment
→ resolved configuration
→ actual run
→ provenance record
→ verified metrics
→ generated table or figure
→ manuscript claim

No step may be silently skipped.
