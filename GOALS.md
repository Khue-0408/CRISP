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

## Current audit status

The initial read-only audit found unresolved scientific risks involving:

- metric correctness;
- checkpoint and projector integrity;
- configuration-to-manuscript drift;
- ablation isolation;
- provenance;
- dataset and teacher identity;
- result export.

These are audit findings, not verified corrections.

## Near-term engineering priorities

1. Establish the canonical journal experiment contract.
2. Establish the experiment and provenance registry.
3. Add configuration-diff and scientific-invariant gates.
4. Resolve P0 scientific implementation mismatches.
5. Add missing controlled experiments.
6. Establish a verified result-to-table pipeline.
7. Perform a final public-surface and reproducibility audit.

The canonical journal reference experiment remains intentionally undecided.
