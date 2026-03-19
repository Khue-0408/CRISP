"""
CLI entry point for regenerating paper-style tables from exported metrics.

This script should:
- scan run directories,
- merge per-run JSON or CSV metric files,
- aggregate by method / dataset / seed,
- export clean table-ready CSV and Markdown files.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from crisp.metrics.aggregation import average_metric_dicts
from crisp.utils.serialization import save_csv


def _infer_run_metadata(path: Path) -> Dict[str, Any]:
    """
    Infer basic run metadata from a metric file path.

    Expected directory structure (recommended)
    ----------------------------------------
    <root>/<experiment_name>/seed_<seed>/.../*.json
    """
    meta: Dict[str, Any] = {}
    parts = list(path.parts)
    # Heuristic: look for "seed_<n>" component.
    seed = None
    exp = None
    for i, p in enumerate(parts):
        if p.startswith("seed_"):
            seed = p.replace("seed_", "")
            exp = parts[i - 1] if i - 1 >= 0 else None
            break
    if exp is not None:
        meta["experiment"] = exp
    if seed is not None:
        try:
            meta["seed"] = int(seed)
        except ValueError:
            meta["seed"] = seed
    return meta


def _collect_metric_files(root: Path) -> List[Dict[str, Any]]:
    """Scan *root* for JSON metric files and return normalized rows."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rows: List[Dict[str, Any]] = []
    for p in sorted(root.rglob("*.json")):
        try:
            with open(p) as f:
                data = json.load(f)
            # Infer dataset name from filename, e.g. "colondb_projector_on.json"
            ds_name = p.stem.rsplit("_projector", 1)[0]
            mode = "projector_on" if "projector_on" in p.stem else "projector_off"
            meta = _infer_run_metadata(p)
            row: Dict[str, Any] = {
                "dataset": ds_name,
                "mode": mode,
                "file": str(p),
            }
            row.update(meta)
            # Copy scalar metrics.
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    row[k] = v
            rows.append(row)
        except (json.JSONDecodeError, KeyError):
            continue
    return rows


def main() -> None:
    """
    Main table export entry point.
    """
    parser = argparse.ArgumentParser(description="Export paper-style metric tables")
    parser.add_argument("--input-dir", type=str, required=True, help="Root output directory to scan")
    parser.add_argument("--output-dir", type=str, default="outputs/tables", help="Directory for exported tables")
    args = parser.parse_args()

    root = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = _collect_metric_files(root)

    # Aggregate by (experiment, seed, dataset, mode) where available, otherwise by (dataset, mode).
    key_fields = ["experiment", "seed", "dataset", "mode"]
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in metric_rows:
        key = tuple(r.get(k, None) for k in key_fields)
        grouped[key].append(r)

    agg_rows: List[Dict[str, Any]] = []
    for key, entries in sorted(grouped.items(), key=lambda kv: str(kv[0])):
        # Average scalar metrics across files (typically multiple evaluation runs).
        metrics_only = [
            {
                k: v
                for k, v in e.items()
                if isinstance(v, (int, float)) and k not in key_fields
            }
            for e in entries
        ]
        avg = average_metric_dicts(metrics_only)

        row: Dict[str, Any] = {k: entries[0].get(k, None) for k in key_fields}
        row.update({k: round(v, 6) for k, v in avg.items()})
        agg_rows.append(row)

    if agg_rows:
        save_csv(out_dir / "aggregated_results.csv", agg_rows)

        # Also emit a Markdown table.
        md_path = out_dir / "aggregated_results.md"
        headers = list(agg_rows[0].keys())
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for r in agg_rows:
            lines.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
        md_path.write_text("\n".join(lines) + "\n")
        print(f"Tables saved to {out_dir}")
    else:
        print("No metric files found.")


if __name__ == "__main__":
    main()
