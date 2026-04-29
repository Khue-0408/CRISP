# Experiments

This document records the public experiment protocol and result tables for CRISP.

## Experiment Protocol

Retained thesis students:

- U-Net
- UNet++
- PraNet

Default CRISP teacher pool:

- UACANet-L
- Polyp-PVT

Default input size is 352x352. The thesis schedule uses 120 epochs:

- Phase I: 25 epochs baseline student warmup
- Phase II: 65 epochs full CRISP objective
- Phase III: 30 epochs joint fine-tuning

Baseline configs:

- `configs/experiment/thesis_unet_baseline.yaml`
- `configs/experiment/thesis_unetpp_baseline.yaml`
- `configs/experiment/thesis_pranet_baseline.yaml`

CRISP configs:

- `configs/experiment/thesis_unet_crisp.yaml`
- `configs/experiment/thesis_unetpp_crisp.yaml`
- `configs/experiment/thesis_pranet_crisp.yaml`

Training scripts:

```bash
bash scripts/train_thesis_unet_baseline.sh
bash scripts/train_thesis_unet_crisp.sh
bash scripts/train_thesis_unetpp_baseline.sh
bash scripts/train_thesis_unetpp_crisp.sh
bash scripts/train_thesis_pranet_baseline.sh
bash scripts/train_thesis_pranet_crisp.sh
```

Evaluation scripts:

```bash
bash scripts/eval_thesis_unet.sh /path/to/checkpoint.pt
bash scripts/eval_thesis_unetpp.sh /path/to/checkpoint.pt
bash scripts/eval_thesis_pranet.sh /path/to/checkpoint.pt
```

Default reporting metrics include `mDice`, `mIoU`, `B-F1`, `HD95`, `bECE`, and `off-bECE`.

## Full Result Tables

The tables below report CRISP gains over each corresponding student baseline. Higher is better for mDice, mIoU, and B-F1; lower is better for HD95 and bECE.

<details>
<summary>Seen-domain gains</summary>

| Student | Dataset | ΔmDice ↑ | ΔmIoU ↑ | ΔB-F1 ↑ | ΔHD95 ↓ | ΔbECE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| U-Net + CRISP | Kvasir-SEG | +2.8 | +3.0 | +3.2 | -2.3 | -0.028 |
| U-Net + CRISP | CVC-ClinicDB | +6.8 | +7.4 | +6.1 | -3.9 | -0.029 |
| UNet++ + CRISP | Kvasir-SEG | +2.8 | +3.8 | +3.1 | -2.2 | -0.027 |
| UNet++ + CRISP | CVC-ClinicDB | +8.2 | +8.3 | +7.6 | -4.0 | -0.028 |
| PraNet + CRISP | Kvasir-SEG | +1.4 | +0.7 | +1.8 | -1.5 | -0.016 |
| PraNet + CRISP | CVC-ClinicDB | +5.4 | +6.3 | +5.9 | -3.2 | -0.016 |

</details>

<details>
<summary>Unseen-domain gains</summary>

| Student | Dataset | ΔmDice ↑ | ΔmIoU ↑ | ΔB-F1 ↑ | ΔHD95 ↓ | ΔbECE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| U-Net + CRISP | CVC-300 | +4.8 | +4.9 | +5.8 | -4.4 | -0.039 |
| U-Net + CRISP | CVC-ColonDB | +3.7 | +4.7 | +5.6 | -3.7 | -0.056 |
| U-Net + CRISP | ETIS | +4.6 | +5.3 | +6.9 | -4.9 | -0.068 |
| UNet++ + CRISP | CVC-300 | +4.6 | +4.7 | +5.5 | -4.3 | -0.038 |
| UNet++ + CRISP | CVC-ColonDB | +4.6 | +5.5 | +5.9 | -4.2 | -0.054 |
| UNet++ + CRISP | ETIS | +4.4 | +4.7 | +6.7 | -4.5 | -0.062 |
| PraNet + CRISP | CVC-300 | +2.9 | +3.3 | +3.7 | -2.1 | -0.020 |
| PraNet + CRISP | CVC-ColonDB | +3.3 | +2.9 | +4.4 | -2.9 | -0.048 |
| PraNet + CRISP | ETIS | +4.1 | +5.5 | +5.3 | -4.3 | -0.058 |

</details>

## Qualitative Figures

The README displays:

- Architecture overview: `docs/ch3_arc.png`
- Seen-domain qualitative results: `docs/ch4_qual_seen.png`
- Unseen-domain qualitative results: `docs/ch4_qual_unseen.png`
