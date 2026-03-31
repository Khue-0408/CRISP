# Local Mode Report

## Ready Now

- Local U-Net baseline training on `TrainDataset`.
- Local U-Net + CRISP training on `TrainDataset` with the existing paper-faithful CRISP math intact:
  - raw student logits `z` in the forward path,
  - detached stabilized local `alpha_star`,
  - bounded projector output in `[alpha_min, alpha_max]`,
  - no teachers or solver at inference,
  - projector-off evaluation implemented as `alpha_hat = 1`.
- Deterministic train/val split creation from `TrainDataset` with cached split files under `metadata/splits/`.
- Automatic discovery of each valid dataset folder under `TestDataset/*`.
- Organized per-dataset evaluation export plus `summary.json` and `summary.csv`.
- Auto-created local workspace folders for checkpoints, outputs, metadata, logs, and cache.
- Optional checkpoint auto-download when explicit URLs are configured.

## External Prerequisites

- Local data folders under either `repo_root/TrainDataset` + `repo_root/TestDataset` or `repo_root/data/TrainDataset` + `repo_root/data/TestDataset`.
- Teacher checkpoints for the local practical pool:
  - PraNet
  - Polyp-PVT
  - optional UACANet-L
- Optional student initialization checkpoint for:
  - local baseline init
  - local CRISP init from a baseline checkpoint
- Optional download URLs if you want missing checkpoints to be fetched automatically.

## Recommended Folder Layout

```text
repo_root/
├── checkpoints/
│   ├── students/
│   │   └── unet/
│   │       ├── init/
│   │       ├── baseline/
│   │       └── crisp/
│   └── teachers/
│       ├── pranet/
│       ├── polyp_pvt/
│       └── uacanet/
├── outputs/
│   ├── metrics/
│   ├── predictions/
│   ├── posthoc/
│   └── runs/
├── metadata/
│   └── splits/
├── logs/
├── cache/
└── data/
    ├── TrainDataset/
    └── TestDataset/
```

## Commands

Bootstrap workspace:

```bash
bash scripts/bootstrap_local_workspace.sh
```

Train local U-Net baseline:

```bash
bash scripts/train_unet_baseline_local.sh
```

Train local U-Net + CRISP:

```bash
bash scripts/train_unet_crisp_local.sh
```

Evaluate a checkpoint over all discovered `TestDataset/*` folders:

```bash
bash scripts/eval_unet_local.sh /path/to/checkpoint.pt
```

## Local Smoke Tests

```bash
conda run -n crisp-replication python -m pytest tests/test_model_loading.py tests/test_train_script.py tests/test_local_data_mode.py -q
conda run -n crisp-replication python -m crisp.scripts.train --config-path /abs/path/to/configs --config-name experiment/task_local_unet_baseline --cfg job
conda run -n crisp-replication python -m crisp.scripts.evaluate --config-path /abs/path/to/configs --config-name experiment/task_local_unet_crisp --cfg job +checkpoint=/tmp/fake.pt
```

## Remaining Risk

- Exact teacher quality still depends on external checkpoints.
- The optional UACANet-L third teacher still requires an importable external implementation.
- The current local environment still emits a `torchvision` image-extension warning about `libjpeg.9.dylib`; it does not block the tested code paths, but the env should be rebuilt if you want that warning gone.
