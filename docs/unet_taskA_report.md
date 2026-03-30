# U-Net Task A Milestone Report

## Ready For U-Net Baseline

- `configs/experiment/taskA_unet_baseline.yaml` provides a first-class Task A U-Net baseline.
- U-Net now uses the same explicit host contract as PraNet:
  - raw logits `z`
  - decoder features `F_theta(x)`
  - explicit `decoder_channels`
- Source-validation enforcement is enabled by default for the main U-Net baseline path.
- Optional student initialization is supported through `student_init.checkpoint`.

## Ready For U-Net + CRISP

- `configs/experiment/taskA_unet_crisp.yaml` provides a first-class Task A U-Net + CRISP path.
- CRISP method identity is unchanged:
  - soft boundary weighting field
  - frozen teacher barycenter
  - boundary posterior target
  - detached stabilized `alpha_star`
  - amortized projector
  - task loss + amortization loss
  - projector-on / projector-off evaluation
- The default practical U-Net teacher pool is:
  - PraNet
  - Polyp-PVT
- An optional third teacher slot is config-driven for UACANet-L or another external teacher implementation when an importable class path is available.

## External Prerequisites Still Needed

- Kvasir-SEG with `splits/train.txt` and `splits/val.txt`
- ColonDB and ETIS target datasets
- Optional PolypGen target dataset
- External teacher checkpoints for:
  - `CRISP_UNET_TEACHER_PRANET_CKPT`
  - `CRISP_UNET_TEACHER_POLYP_PVT_CKPT`
- Optional external UACANet-L teacher:
  - `CRISP_UNET_ENABLE_UACANET_TEACHER=true`
  - `CRISP_UNET_TEACHER_UACANET_CLASS`
  - `CRISP_UNET_TEACHER_UACANET_CKPT`
- Optional student initialization checkpoint:
  - `CRISP_UNET_BASELINE_INIT_CKPT`
  - `CRISP_UNET_CRISP_INIT_CKPT`

## Local Smoke Tests

Run the local U-Net path checks with:

```bash
conda run -n crisp-replication python -m pytest tests/test_model_loading.py tests/test_unet_path.py -q
conda run -n crisp-replication python -m crisp.scripts.train --config-path /Users/minhnguyen/Desktop/CRISP_MIDL/configs --config-name experiment/taskA_unet_baseline --cfg job
conda run -n crisp-replication python -m crisp.scripts.train --config-path /Users/minhnguyen/Desktop/CRISP_MIDL/configs --config-name experiment/taskA_unet_crisp --cfg job
conda run -n crisp-replication python -m crisp.scripts.evaluate --config-path /Users/minhnguyen/Desktop/CRISP_MIDL/configs --config-name experiment/taskA_unet_crisp --cfg job +checkpoint=/tmp/fake.pt
```

Convenience shell scripts:

```bash
bash scripts/train_unet_baseline.sh
bash scripts/train_unet_crisp.sh
bash scripts/eval_unet_crossdomain.sh /path/to/checkpoint.pt
```
