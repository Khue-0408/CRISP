# Dataset Setup

CRISP uses a simple TrainDataset/TestDataset layout for local and thesis
experiments.

```text
data/
├── TrainDataset/
│   ├── image/ or images/
│   └── mask/  or masks/
└── TestDataset/
    ├── Kvasir/
    ├── CVC-ClinicDB/
    ├── CVC-300/
    ├── CVC-ColonDB/
    └── ETIS-LaribPolypDB/
```

Each test dataset folder must contain an image folder and a mask folder. Folder
names may be singular or plural.

Protocol:

- Training uses all matched image/mask pairs in `TrainDataset`.
- Validation is created deterministically from `TrainDataset` using
  `val_fraction=0.1` by default.
- Evaluation auto-discovers valid immediate child folders under `TestDataset`.
- Missing folders, empty datasets, and image/mask pairing mismatches fail loudly.
- Exact benchmark counts are not enforced by default.

Check the data tree with:

```bash
bash scripts/verify_data.sh --root ./data --non-strict
```
