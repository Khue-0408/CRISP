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
- Initial host: PraNet.
- Main outputs: Dice, Boundary-F1, HD95, ECE, bECE, BA-ECE, TACE, Brier, NLL.
- Main comparisons: baseline, CRISP, soft-label controls, spatial-alpha baseline, boundary-posterior CE.

## Environment

Set dataset and teacher checkpoint roots before running the paper-faithful CRISP config:

```bash
export CRISP_DATA_ROOT=/path/to/data
export CRISP_PRANET_TEACHER_CKPT=/path/to/pranet_teacher.pt
export CRISP_POLYP_PVT_TEACHER_CKPT=/path/to/polyp_pvt_teacher.pt
export CRISP_RABBIT_TEACHER_CKPT=/path/to/rabbit_teacher.pt
```

`taskA_pranet_crisp.yaml` expects a frozen three-teacher pool. If those checkpoints are missing, training now fails loudly instead of silently falling back to self-ensembling.

## Quick Start

```bash
conda env create -f environment.yml
conda activate crisp-replication
pip install -e .
bash scripts/train_pranet_crisp.sh
bash scripts/eval_pranet_crossdomain.sh /path/to/checkpoint.pt
bash scripts/export_tables.sh
```
