#!/usr/bin/env bash

crisp_resolve_python() {
  if [[ -n "${CRISP_PYTHON:-}" && -x "${CRISP_PYTHON:-}" ]]; then
    printf '%s\n' "$CRISP_PYTHON"
    return
  fi

  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi

  echo "Could not find Python. Set CRISP_PYTHON=/path/to/python." >&2
  return 1
}

crisp_arg_present() {
  local needle="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "$arg" == "$needle"* ]]; then
      return 0
    fi
  done
  return 1
}

crisp_should_use_tiny_smoke_data() {
  if [[ -n "${CRISP_DATA_ROOT:-}" ]]; then
    return 1
  fi
  if crisp_arg_present "source_data.root=" "$@"; then
    return 1
  fi
  crisp_arg_present "training.total_epochs=1" "$@"
}

crisp_ensure_tiny_smoke_data() {
  local python_bin="$1"
  "$python_bin" - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw

root = Path("/tmp/crisp_thesis_smoke_data")
folders = [
    root / "TrainDataset" / "image",
    root / "TrainDataset" / "mask",
    root / "TestDataset" / "Kvasir" / "images",
    root / "TestDataset" / "Kvasir" / "masks",
]
for folder in folders:
    folder.mkdir(parents=True, exist_ok=True)

for image_rel, mask_rel, count in [
    ("TrainDataset/image", "TrainDataset/mask", 4),
    ("TestDataset/Kvasir/images", "TestDataset/Kvasir/masks", 2),
]:
    for idx in range(count):
        image_path = root / image_rel / f"{idx}.png"
        mask_path = root / mask_rel / f"{idx}.png"
        if image_path.exists() and mask_path.exists():
            continue
        image = Image.new("RGB", (64, 64), (40 + idx * 30, 80, 120))
        mask = Image.new("L", (64, 64), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((16, 16, 48, 48), fill=255)
        image.save(image_path)
        mask.save(mask_path)

print(root)
PY
}
