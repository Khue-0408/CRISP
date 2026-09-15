# Retained Baseline Source Contract

This subtree contains retained upstream or reference implementations used for
CRISP adapters and checkpoint compatibility.

## Preservation

Do not mass-refactor, reformat, rename, modernize, or otherwise clean up these
sources. Preserve architecture semantics, checkpoint compatibility, licenses,
attribution, and any preprocessing required by the retained implementation.

Minimal changes are allowed only for adapter compatibility, runtime
portability, security, public-surface hygiene, removal of private absolute
paths, or explicit scientific integration. Do not alter architecture or
preprocessing to improve results.

## Separation

Implement CRISP behavior in `src/` and configuration, and prefer a wrapper or
adapter over modifying retained baseline internals.

## Verification

Test changes that can affect outputs. Do not claim reproduction or verified
baseline performance unless the relevant evaluation was actually run.
