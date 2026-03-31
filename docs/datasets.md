# Dataset Setup

The repository expects `CRISP_DATA_ROOT` to contain:

- `Kvasir-SEG/`
- `ColonDB/`
- `ETIS-LaribPolypDB/`
- `PolypGen/`

Expected minimal layout:

```text
<dataset>/
  images/
  masks/
  splits/
    train.txt
    val.txt
    test.txt
```

Protocol notes:

- `Kvasir-SEG` uses `splits/train.txt` and `splits/val.txt` for source-only fitting and validation.
- `ColonDB` and `ETIS` are treated as target-only evaluation sets.
- `PolypGen` should use a precomputed patient-/center-aware `splits/test.txt`.

## Local Mode

The repository also supports a separate local TrainDataset/TestDataset workflow.

Accepted local data roots:

- `repo_root/`
- `repo_root/data/`

Expected layout:

```text
<local_data_root>/
├── TrainDataset/
│   ├── image/   or images/
│   └── mask/    or masks/
└── TestDataset/
    ├── <dataset_name_1>/
    │   ├── image/ or images/
    │   └── mask/  or masks/
    └── <dataset_name_2>/
```

Local-mode protocol notes:

- Training uses all matched pairs in `TrainDataset`.
- Validation is created deterministically from `TrainDataset` using the configured seed and `val_fraction`.
- Split files are cached under `metadata/splits/`.
- Evaluation auto-discovers valid immediate child datasets under `TestDataset/*`.
- Image/mask mismatches fail loudly in local mode instead of being silently intersected.
