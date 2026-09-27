# Dataset Setup

CRISP uses a TrainDataset/TestDataset layout for current and local-debug
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

Current protocol:

- Source training and validation membership must be supplied by explicit manifests.
- Evaluation membership must be supplied by one explicit manifest for each of the
  five canonical datasets.
- Missing folders, empty datasets, and image/mask pairing mismatches fail loudly.
- Current-protocol runs enforce canonical dataset identities and exact counts.

Fraction-derived source splits and automatically discovered evaluation membership
remain available only to configs that do not opt into `protocol_profile: current_crisp`.

Check the data tree with:

```bash
bash scripts/verify_data.sh --root ./data --non-strict
```
