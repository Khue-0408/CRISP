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

Higher is better for `mDice`, `mIoU`, and `B-F1`; lower is better for `HD95` and `bECE`.

### Cross-dataset robustness on polyp segmentation: seen-domain results. Means over five seeds.

| Group | Method | Params (M) | FLOPs (G) | Kvasir-SEG mDice ↑ | Kvasir-SEG mIoU ↑ | Kvasir-SEG B-F1 ↑ | Kvasir-SEG HD95 ↓ | Kvasir-SEG bECE ↓ | CVC-ClinicDB mDice ↑ | CVC-ClinicDB mIoU ↑ | CVC-ClinicDB B-F1 ↑ | CVC-ClinicDB HD95 ↓ | CVC-ClinicDB bECE ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prominent State-of-the-art Methods | SANet | 23.8 | 11.3 | 0.904 | 0.847 | 0.868 | 10.1 | 0.030 | 0.916 | 0.859 | 0.881 | 8.8 | 0.026 |
| Prominent State-of-the-art Methods | MSNet | 27.6 | 17.0 | 0.907 | 0.862 | 0.875 | 9.7 | 0.028 | 0.921 | 0.879 | 0.892 | 8.1 | 0.024 |
| Prominent State-of-the-art Methods | Polyp-PVT | 25.1 | 10.1 | 0.917 | 0.864 | 0.886 | 8.9 | 0.025 | 0.937 | 0.889 | 0.914 | 6.6 | 0.020 |
| Prominent State-of-the-art Methods | CTNet | 44.2 | 32.6 | 0.917 | 0.863 | 0.889 | 8.7 | 0.024 | 0.936 | 0.887 | 0.913 | 6.4 | 0.019 |
| Prominent State-of-the-art Methods | CFA-Net | 25.2 | 55.3 | 0.915 | 0.861 | 0.884 | 8.9 | 0.024 | 0.933 | 0.883 | 0.911 | 6.9 | 0.019 |
| Prominent State-of-the-art Methods | SAM-Mamba | 103.0 | 423.0 | 0.924 | 0.873 | 0.899 | 8.1 | 0.021 | 0.942 | 0.887 | 0.922 | 6.0 | 0.017 |
| Lightweight Baseline Methods | U-Net | 16.7 | 73.9 | 0.818 | 0.746 | 0.782 | 16.9 | 0.067 | 0.823 | 0.755 | 0.791 | 15.1 | 0.061 |
| Lightweight Baseline Methods | UNet++ | 9.1 | 65.9 | 0.821 | 0.743 | 0.789 | 16.4 | 0.064 | 0.794 | 0.729 | 0.768 | 15.8 | 0.063 |
| Lightweight Baseline Methods | PraNet | 30.4 | 13.1 | 0.898 | 0.840 | 0.861 | 11.3 | 0.041 | 0.899 | 0.849 | 0.872 | 9.4 | 0.035 |
| Lightweight Baselines Enhanced with CRISP (Ours) | U-Net + CRISP | 16.9 | 74.7 | 0.846 | 0.776 | 0.814 | 14.6 | 0.039 | 0.891 | 0.829 | 0.852 | 11.2 | 0.032 |
|  | Gain over U-Net |  |  | +2.8 | +3.0 | +3.2 | -2.3 | -0.028 | +6.8 | +7.4 | +6.1 | -3.9 | -0.029 |
| Lightweight Baselines Enhanced with CRISP (Ours) | UNet++ + CRISP | 9.4 | 66.9 | 0.849 | 0.781 | 0.820 | 14.2 | 0.037 | 0.876 | 0.812 | 0.844 | 11.8 | 0.035 |
|  | Gain over UNet++ |  |  | +2.8 | +3.8 | +3.1 | -2.2 | -0.027 | +8.2 | +8.3 | +7.6 | -4.0 | -0.028 |
| Lightweight Baselines Enhanced with CRISP (Ours) | PraNet + CRISP | 31.2 | 13.8 | 0.912 | 0.847 | 0.879 | 9.8 | 0.025 | 0.953 | 0.912 | 0.931 | 6.2 | 0.019 |
|  | Gain over PraNet |  |  | +1.4 | +0.7 | +1.8 | -1.5 | -0.016 | +5.4 | +6.3 | +5.9 | -3.2 | -0.016 |

### Cross-dataset robustness on polyp segmentation: unseen-domain results. Means over five seeds.

| Group | Method | Params (M) | FLOPs (G) | CVC-300 mDice ↑ | CVC-300 mIoU ↑ | CVC-300 B-F1 ↑ | CVC-300 HD95 ↓ | CVC-300 bECE ↓ | CVC-ColonDB mDice ↑ | CVC-ColonDB mIoU ↑ | CVC-ColonDB B-F1 ↑ | CVC-ColonDB HD95 ↓ | CVC-ColonDB bECE ↓ | ETIS mDice ↑ | ETIS mIoU ↑ | ETIS B-F1 ↑ | ETIS HD95 ↓ | ETIS bECE ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prominent State-of-the-art Methods | SANet | 23.8 | 11.3 | 0.888 | 0.815 | 0.843 | 12.7 | 0.036 | 0.753 | 0.670 | 0.714 | 17.8 | 0.058 | 0.750 | 0.654 | 0.707 | 18.6 | 0.062 |
| Prominent State-of-the-art Methods | MSNet | 27.6 | 17.0 | 0.869 | 0.807 | 0.835 | 13.2 | 0.038 | 0.755 | 0.678 | 0.719 | 17.1 | 0.056 | 0.719 | 0.664 | 0.688 | 19.2 | 0.065 |
| Prominent State-of-the-art Methods | Polyp-PVT | 25.1 | 10.1 | 0.900 | 0.833 | 0.872 | 11.4 | 0.031 | 0.808 | 0.727 | 0.779 | 14.3 | 0.045 | 0.787 | 0.706 | 0.752 | 15.2 | 0.049 |
| Prominent State-of-the-art Methods | CTNet | 44.2 | 32.6 | 0.908 | 0.844 | 0.881 | 10.8 | 0.029 | 0.813 | 0.734 | 0.786 | 13.7 | 0.043 | 0.810 | 0.734 | 0.773 | 14.2 | 0.046 |
| Prominent State-of-the-art Methods | CFA-Net | 25.2 | 55.3 | 0.893 | 0.827 | 0.865 | 11.7 | 0.031 | 0.743 | 0.665 | 0.716 | 17.5 | 0.053 | 0.732 | 0.655 | 0.701 | 18.4 | 0.058 |
| Prominent State-of-the-art Methods | SAM-Mamba | 103.0 | 423.0 | 0.920 | 0.861 | 0.892 | 9.9 | 0.025 | 0.853 | 0.771 | 0.829 | 11.9 | 0.038 | 0.848 | 0.782 | 0.814 | 11.4 | 0.040 |
| Lightweight Baseline Methods | U-Net | 16.7 | 73.9 | 0.710 | 0.627 | 0.661 | 21.8 | 0.095 | 0.744 | 0.661 | 0.691 | 18.4 | 0.094 | 0.689 | 0.538 | 0.623 | 23.7 | 0.118 |
| Lightweight Baseline Methods | UNet++ | 9.1 | 65.9 | 0.707 | 0.624 | 0.668 | 22.1 | 0.091 | 0.731 | 0.648 | 0.684 | 19.2 | 0.090 | 0.704 | 0.556 | 0.636 | 22.4 | 0.110 |
| Lightweight Baseline Methods | PraNet | 30.4 | 13.1 | 0.871 | 0.797 | 0.834 | 13.0 | 0.051 | 0.779 | 0.704 | 0.728 | 15.8 | 0.079 | 0.727 | 0.571 | 0.671 | 20.4 | 0.101 |
| Lightweight Baselines Enhanced with CRISP (Ours) | U-Net + CRISP | 16.9 | 74.7 | 0.758 | 0.676 | 0.719 | 17.4 | 0.056 | 0.781 | 0.708 | 0.747 | 14.7 | 0.038 | 0.735 | 0.591 | 0.692 | 18.8 | 0.050 |
|  | Gain over U-Net |  |  | +4.8 | +4.9 | +5.8 | -4.4 | -0.039 | +3.7 | +4.7 | +5.6 | -3.7 | -0.056 | +4.6 | +5.3 | +6.9 | -4.9 | -0.068 |
| Lightweight Baselines Enhanced with CRISP (Ours) | UNet++ + CRISP | 9.4 | 66.9 | 0.753 | 0.671 | 0.723 | 17.8 | 0.053 | 0.777 | 0.703 | 0.743 | 15.0 | 0.036 | 0.748 | 0.603 | 0.703 | 17.9 | 0.048 |
|  | Gain over UNet++ |  |  | +4.6 | +4.7 | +5.5 | -4.3 | -0.038 | +4.6 | +5.5 | +5.9 | -4.2 | -0.054 | +4.4 | +4.7 | +6.7 | -4.5 | -0.062 |
| Lightweight Baselines Enhanced with CRISP (Ours) | PraNet + CRISP | 31.2 | 13.8 | 0.900 | 0.830 | 0.871 | 10.9 | 0.031 | 0.812 | 0.733 | 0.772 | 12.9 | 0.031 | 0.768 | 0.626 | 0.724 | 16.1 | 0.043 |
|  | Gain over PraNet |  |  | +2.9 | +3.3 | +3.7 | -2.1 | -0.020 | +3.3 | +2.9 | +4.4 | -2.9 | -0.048 | +4.1 | +5.5 | +5.3 | -4.3 | -0.058 |

## Qualitative Figures

The README displays:

- Architecture overview: `docs/ch3_arc.png`
- Seen-domain qualitative results: `docs/ch4_qual_seen.png`
- Unseen-domain qualitative results: `docs/ch4_qual_unseen.png`
