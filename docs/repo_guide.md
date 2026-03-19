# Repository Guide

Directory responsibilities:

- `configs/`: experiment, dataset, model, and evaluation defaults.
- `src/crisp/modules/`: CRISP-specific math (`w_b`, `p_T`, `t_star`, solver, losses).
- `src/crisp/models/`: host backbones and the amortized projector head.
- `src/crisp/engine/`: trainer/evaluator orchestration.
- `src/crisp/scripts/`: runnable CLI entry points.
- `tests/`: invariants and regression tests for solver, metrics, projector, evaluator, and exports.
