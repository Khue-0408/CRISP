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

## Partial

- The CRISP method path is faithful for Task A / PraNet, but exact paper-number reproduction still depends on using the official host architectures and checkpoints.
- The U-Net milestone is ready for local smoke tests and Task A training/evaluation wiring, but exact cross-repo teacher reproduction still depends on external checkpoints.
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
```

See [`docs/audit_report.md`](/Users/minhnguyen/Desktop/CRISP_MIDL/docs/audit_report.md) for the structured audit status.
See [`docs/unet_taskA_report.md`](/Users/minhnguyen/Desktop/CRISP_MIDL/docs/unet_taskA_report.md) for the U-Net milestone report.
