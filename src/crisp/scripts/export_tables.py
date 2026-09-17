"""Export traceable per-artifact evaluator records without cross-run statistics."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from crisp.utils.serialization import save_csv


_MODES = {"projector_on", "projector_off"}
_SEED_DIR = re.compile(r"seed_([0-9]+)\Z")


def _parse_metric_file(path: Path, root: Path) -> dict[str, Any]:
    """Parse one JSON file in the evaluator's nested output layout."""
    parts = path.relative_to(root).parts
    if len(parts) != 4:
        raise ValueError(
            f"Evaluator result '{path}' must be under "
            "<root>/<experiment>/seed_<integer>/<dataset>/<mode>.json."
        )
    experiment, seed_dir, dataset, filename = parts
    seed_match = _SEED_DIR.fullmatch(seed_dir)
    if seed_match is None:
        raise ValueError(f"Evaluator result '{path}' has invalid seed directory '{seed_dir}'.")
    mode = Path(filename).stem
    if mode not in _MODES:
        raise ValueError(
            f"Evaluator result '{path}' has unsupported mode '{mode}'; "
            "expected projector_on.json or projector_off.json. Legacy flat filenames are unsupported."
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Evaluator result '{path}' is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Evaluator result '{path}' must contain a nonempty metric mapping.")
    for key, value in data.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Evaluator result '{path}' has non-numeric or non-finite metric '{key}'.")

    return {
        "source_file": str(path),
        "experiment": experiment,
        "seed": int(seed_match.group(1)),
        "dataset": dataset,
        "mode": mode,
        "metrics": data,
    }


def _collect_metric_files(root: Path) -> list[dict[str, Any]]:
    """Discover per-dataset results; exclude evaluator summaries to avoid double counting."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Evaluator result root does not exist: {root}")
    records = []
    for path in sorted(root.rglob("*.json")):
        if path.name == "summary.json":
            continue
        records.append(_parse_metric_file(path, root))
    return records


def main() -> None:
    """Export one row per evaluator artifact, without seed aggregation."""
    parser = argparse.ArgumentParser(description="Export normalized per-artifact evaluator records")
    parser.add_argument("--input-dir", type=str, required=True, help="Evaluator metrics root")
    parser.add_argument("--output-dir", type=str, default="outputs/tables", help="Parsed-record output directory")
    args = parser.parse_args()

    records = _collect_metric_files(Path(args.input_dir))
    if not records:
        print("No per-dataset evaluator records found.")
        return

    metric_keys = sorted({key for record in records for key in record["metrics"]})
    headers = ["source_file", "experiment", "seed", "dataset", "mode", *metric_keys]
    rows = [
        {**{key: record[key] for key in headers[:5]},
         **{key: record["metrics"].get(key, "") for key in metric_keys}}
        for record in records
    ]
    output_dir = Path(args.output_dir)
    save_csv(output_dir / "parsed_results.csv", rows)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(str(row[key]) for key in headers) + " |" for row in rows),
    ]
    (output_dir / "parsed_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Parsed evaluator records saved to {output_dir}")


if __name__ == "__main__":
    main()
