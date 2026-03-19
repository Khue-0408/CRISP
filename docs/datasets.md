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
