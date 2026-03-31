# CRISP

Research-grade replication repository for:

CRISP: Amortized Boundary Posterior Projection for Robust and Calibrated Polyp Segmentation under Domain Shift

## Goals

This repository provides:
- A config-driven CRISP implementation with explicit detached `alpha_star`.
- Source-only training and target-only evaluation under domain shift.
- Modular support for teacher barycenters, boundary posterior targets, stabilized local projection, amortized projector training, and post-hoc baselines.
- Reproducible metric export for projector-on/projector-off evaluation.

## Planned reproduction targets

- Task A: Kvasir-SEG -> ColonDB / ETIS / PolypGen.
- Initial hosts: PraNet and U-Net.
- Main outputs: Dice, Boundary-F1, HD95, ECE, bECE, BA-ECE, TACE, Brier, NLL.
- Main comparisons: baseline, CRISP, soft-label controls, spatial-alpha baseline, boundary-posterior CE.

## Environment

Set dataset and teacher checkpoint roots before running the paper-faithful CRISP config:

```bash
export CRISP_DATA_ROOT=/path/to/data
export CRISP_PRANET_TEACHER_CKPT=/path/to/pranet_teacher.pt
export CRISP_POLYP_PVT_TEACHER_CKPT=/path/to/polyp_pvt_teacher.pt
export CRISP_RABBIT_TEACHER_CKPT=/path/to/rabbit_teacher.pt

export CRISP_UNET_TEACHER_PRANET_CKPT=/path/to/pranet_teacher.pt
export CRISP_UNET_TEACHER_POLYP_PVT_CKPT=/path/to/polyp_pvt_teacher.pt
export CRISP_UNET_CRISP_INIT_CKPT=/path/to/unet_baseline_or_init.pt

# Optional third teacher for the U-Net path:
export CRISP_UNET_ENABLE_UACANET_TEACHER=true
export CRISP_UNET_TEACHER_UACANET_CLASS=package.module.UACANetL
export CRISP_UNET_TEACHER_UACANET_CKPT=/path/to/uacanet_l_teacher.pt
```

`taskA_pranet_crisp.yaml` expects a frozen three-teacher pool. If those checkpoints are missing, training now fails loudly instead of silently falling back to self-ensembling.

`taskA_unet_crisp.yaml` uses a practical two-teacher default pool:
- PraNet
- Polyp-PVT

The optional UACANet-L third teacher is config-driven and only activated when
`CRISP_UNET_ENABLE_UACANET_TEACHER=true` plus a valid importable class path and
checkpoint are provided.

### Local Mode Environment

For the local U-Net workflow, the repository supports a separate TrainDataset/TestDataset
mode without changing the CRISP math:

```bash
export CRISP_LOCAL_WORKSPACE_ROOT=/path/to/repo_root
export CRISP_LOCAL_DATA_ROOT=/path/to/repo_root/data
# The loader also accepts /path/to/repo_root as the local data root.

export CRISP_LOCAL_VAL_FRACTION=0.1
export CRISP_AUTO_DOWNLOAD_WEIGHTS=false

export CRISP_LOCAL_UNET_TEACHER_PRANET_CKPT=checkpoints/teachers/pranet/pranet_teacher.pt
export CRISP_LOCAL_UNET_TEACHER_POLYP_PVT_CKPT=checkpoints/teachers/polyp_pvt/polyp_pvt_teacher.pt
export CRISP_LOCAL_UNET_TEACHER_UACANET_CKPT=checkpoints/teachers/uacanet/uacanet_l_teacher.pt

export CRISP_LOCAL_UNET_TEACHER_PRANET_URL=
export CRISP_LOCAL_UNET_TEACHER_POLYP_PVT_URL=
export CRISP_LOCAL_UNET_TEACHER_UACANET_URL=

export CRISP_LOCAL_UNET_BASELINE_INIT_CKPT=
export CRISP_LOCAL_UNET_BASELINE_INIT_URL=
export CRISP_LOCAL_UNET_CRISP_INIT_CKPT=
export CRISP_LOCAL_UNET_CRISP_INIT_URL=

# Optional third teacher:
export CRISP_LOCAL_ENABLE_UACANET_TEACHER=false
export CRISP_LOCAL_UNET_TEACHER_UACANET_CLASS=package.module.UACANetL
```

### Data Root Layout

`CRISP_DATA_ROOT` should contain:

```text
$CRISP_DATA_ROOT/
├── Kvasir-SEG/
│   ├── images/
│   ├── masks/
│   └── splits/
│       ├── train.txt
│       └── val.txt
├── ColonDB/
│   ├── images/
│   └── masks/
├── ETIS-LaribPolypDB/
│   ├── images/
│   └── masks/
└── PolypGen/
    ├── images/
    ├── masks/
    └── splits/
        └── test.txt
```

### Local Mode Layout

The local U-Net mode accepts either:

- `repo_root/TrainDataset` and `repo_root/TestDataset`
- `repo_root/data/TrainDataset` and `repo_root/data/TestDataset`

with the following structure:

```text
<local_data_root>/
├── TrainDataset/
│   ├── image/   or images/
│   └── mask/    or masks/
└── TestDataset/
    ├── CVC-300/
    │   ├── image/ or images/
    │   └── mask/  or masks/
    ├── CVC-ClinicDB/
    ├── CVC-ColonDB/
    ├── ETIS-LaribPolypDB/
    └── Kvasir/
```

Local-mode behavior:

- Train on the entire `TrainDataset`.
- Build a deterministic train/val split from `TrainDataset` and cache it under `metadata/splits/`.
- Auto-discover immediate child datasets under `TestDataset/*` for evaluation unless `eval_datasets` is overridden.
- Preserve CRISP inference invariants: no teachers, no local solver, projector-off means `alpha_hat = 1`.

### Local Smoke Test

Bootstrap the local workspace first:

```bash
bash scripts/bootstrap_local_workspace.sh
```

For a quick CPU smoke test on macOS or laptop hardware, prefer small overrides:

```bash
bash scripts/train_unet_baseline_local.sh \
  training.epochs=1 \
  training.batch_size=2 \
  training.mixed_precision=false \
  source_data.num_workers=0 \
  source_data.pin_memory=false \
  model.base_channels=8
```

This run now logs `Starting training...` and `Starting epoch ...` immediately, so an initially quiet terminal is no longer ambiguous.

To smoke-test local CRISP without waiting for a full baseline run:

```bash
unset CRISP_LOCAL_UNET_CRISP_INIT_CKPT

bash scripts/train_unet_crisp_local.sh \
  training.epochs=1 \
  training.batch_size=2 \
  training.mixed_precision=false \
  source_data.num_workers=0 \
  source_data.pin_memory=false \
  model.base_channels=8
```

If `CRISP_LOCAL_UNET_CRISP_INIT_CKPT` points to a missing baseline checkpoint, `scripts/train_unet_crisp_local.sh` falls back to `checkpoints/students/unet/init/unet_init.pt` and loads it in non-strict mode for smoke testing only.

For a true baseline-initialized CRISP run, let the baseline finish at least one epoch and then point CRISP to the produced checkpoint, for example:

```bash
export CRISP_LOCAL_UNET_CRISP_INIT_CKPT=/path/to/repo_root/checkpoints/students/unet/baseline/seed_0/epoch_1.pt
```

For baseline evaluation, pass the config explicitly because `scripts/eval_unet_local.sh` defaults to `experiment/task_local_unet_crisp`:

```bash
bash scripts/eval_unet_local.sh \
  /path/to/repo_root/checkpoints/students/unet/baseline/seed_0/epoch_1.pt \
  experiment/task_local_unet_baseline \
  eval.batch_size=1 \
  source_data.num_workers=0 \
  source_data.pin_memory=false
```

## Quick Start

```bash
conda env create -f environment.yml
conda activate crisp-replication
pip install -e .
bash scripts/train_pranet_crisp.sh
bash scripts/eval_pranet_crossdomain.sh /path/to/checkpoint.pt
bash scripts/train_unet_baseline.sh
bash scripts/train_unet_crisp.sh
bash scripts/eval_unet_crossdomain.sh /path/to/checkpoint.pt
bash scripts/bootstrap_local_workspace.sh
bash scripts/train_unet_baseline_local.sh
bash scripts/train_unet_crisp_local.sh
bash scripts/eval_unet_local.sh /path/to/checkpoint.pt experiment/task_local_unet_baseline
bash scripts/eval_unet_local.sh /path/to/checkpoint.pt experiment/task_local_unet_crisp
bash scripts/export_tables.sh
```

## Paper-Faithful Now

- Task A / PraNet core path preserves CRISP as amortized boundary posterior projection, not generic soft-label learning or generic spatial temperature scaling.
- Task A / U-Net baseline and U-Net + CRISP paths are now first-class local run targets with explicit host configs and smoke coverage.
- Training keeps explicit detached `alpha_star`, a stabilized inner solver used only for supervision, and raw student logits `z` in the forward path.
- The projector predicts bounded per-pixel inverse temperatures in `[alpha_min, alpha_max]` and inference uses `p_tilde = sigmoid(alpha_hat * z)`.
- Boundary-posterior targets use the frozen teacher barycenter and soft Gaussian boundary weighting field.
- Evaluation exports projector-on and projector-off results, including Dice, Boundary-F1, HD95, ECE, bECE, BA-ECE, TACE, Brier, and NLL.
- Post-hoc calibrators are fit on source validation only.
- Paper-faithful experiment configs now require source validation and fail loudly when teacher checkpoints or requested target datasets are missing.
- U-Net teacher pools are config-driven, with a practical two-teacher default and an optional third external teacher slot.
- Local U-Net mode now supports `TrainDataset`/`TestDataset` workflows with deterministic split caching, auto-discovered test datasets, local checkpoint workspace bootstrapping, and optional checkpoint auto-download.

## Partial

- The CRISP method path is faithful for Task A / PraNet, but exact paper-number reproduction still depends on using the official host architectures and checkpoints.
- The U-Net milestone is ready for local smoke tests and Task A training/evaluation wiring, but exact cross-repo teacher reproduction still depends on external checkpoints.
- Local mode is ready for TrainDataset/TestDataset workflows, but the exact teacher quality and optional third-teacher integration still depend on external weights and, for UACANet-L, an external implementation.
- `src/crisp/models/pranet.py` is a clean CRISP-compatible PraNet-style wrapper, not the original official Res2Net PraNet implementation.
- `src/crisp/models/polyp_pvt.py` and `src/crisp/models/rabbit.py` are lightweight compatible wrappers for teacher/host interfacing, not official paper backbones.
- `src/crisp/scripts/benchmark.py` is a lightweight local benchmark utility, not a full paper-accurate compute reproduction pipeline.

## Not Yet Implemented

- Bundled official teacher checkpoints and exact upstream host-model weights.
- A repo-local UACANet-L wrapper. The optional third teacher slot currently expects an importable external class path.
- Final repository license text in `LICENSE`.

## Local Verification

The current local smoke path has been exercised with:

```bash
conda run -n crisp-replication python -m pytest tests -q
conda run -n crisp-replication python -m crisp.scripts.train --config-path /abs/path/to/configs --config-name experiment/taskA_pranet_crisp --cfg job
conda run -n crisp-replication bash scripts/train_pranet_crisp.sh --cfg job
conda run -n crisp-replication python -m crisp.scripts.train --config-path /abs/path/to/configs --config-name experiment/taskA_unet_crisp --cfg job
conda run -n crisp-replication python -m crisp.scripts.evaluate --config-path /abs/path/to/configs --config-name experiment/taskA_unet_crisp --cfg job +checkpoint=/tmp/fake.pt
conda run -n crisp-replication bash scripts/bootstrap_local_workspace.sh
conda run -n crisp-replication python -m crisp.scripts.train --config-path /abs/path/to/configs --config-name experiment/task_local_unet_baseline --cfg job
conda run -n crisp-replication python -m crisp.scripts.evaluate --config-path /abs/path/to/configs --config-name experiment/task_local_unet_crisp --cfg job +checkpoint=/tmp/fake.pt
```

See [`docs/audit_report.md`](/Users/minhnguyen/Desktop/CRISP_MIDL/docs/audit_report.md) for the structured audit status.
See [`docs/unet_taskA_report.md`](/Users/minhnguyen/Desktop/CRISP_MIDL/docs/unet_taskA_report.md) for the U-Net milestone report.
See [`docs/local_mode_report.md`](/Users/minhnguyen/Desktop/CRISP_MIDL/docs/local_mode_report.md) for the TrainDataset/TestDataset local-mode report.
