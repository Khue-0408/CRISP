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

Exact historical membership manifests underlying the reported experiments are
not included in this release. Current-protocol execution requires explicit
user-supplied manifests; membership must not be inferred from seeds, fractions,
or directory order.

Check the data tree with:

```bash
bash scripts/verify_data.sh --root ./data --non-strict
```
