# CRISP Implementation Rules

This subtree contains the active CRISP implementation.

## Method Discipline

Before changing method behavior, identify which scientific component is
affected:

- boundary weighting,
- teacher posterior construction,
- teacher aggregation,
- projection/solver,
- projector,
- task loss,
- amortization loss,
- identity regularization,
- training schedule,
- inference behavior,
- evaluation behavior.

Do not modify multiple scientific components incidentally.

## Canonical Behavior

Preserve canonical/default CRISP behavior unless the task explicitly targets
the canonical method.

Ablation support should normally be opt-in through configuration.

The default path must not silently become an ablation path.

## Implementation

Prefer extending an existing clean abstraction over duplicating the training
or evaluation pipeline.

Do not introduce general frameworks for a single ablation.

Preserve backward compatibility with existing checkpoints and configs when
reasonably possible.

Manuscript intent overrides convenience refactors.

Do not silently treat unused scientific configuration keys as implemented
behavior.

Comments and docstrings must not claim scientific capability that the runtime
does not consume.

## Verification

Changes to a scientific component require tests for the changed behavior.

Where possible, verify both:
- canonical behavior remains unchanged,
- the new ablation behavior differs in exactly the intended way.

Do not weaken an invariant merely to make a test pass.
