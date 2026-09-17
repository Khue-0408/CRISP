"""Export traceable per-artifact evaluator records without cross-run statistics."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from crisp.utils.serialization import save_csv
from crisp.utils.provenance import content_sha256, file_sha256, read_provenance


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

    record = {
        "source_file": str(path),
        "experiment": experiment,
        "seed": int(seed_match.group(1)),
        "dataset": dataset,
        "mode": mode,
        "metrics": data,
    }
    sidecar = path.with_name(f"{mode}.provenance.json")
    if sidecar.exists():
        provenance = read_provenance(sidecar, "evaluation")
        for key in ("experiment", "seed", "dataset", "mode"):
            if provenance.get(key) != record[key]:
                raise ValueError(f"Evaluator path {key} conflicts with provenance: {sidecar}")
        if Path(provenance.get("source_file", "")).resolve() != path.resolve():
            raise ValueError(f"Evaluator source_file conflicts with provenance: {sidecar}")
        checkpoint_path = provenance.get("checkpoint_path")
        checkpoint_sha = provenance.get("checkpoint_sha256")
        if not checkpoint_path:
            raise ValueError(f"Missing checkpoint path in provenance: {sidecar}")
        if not isinstance(checkpoint_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha):
            raise ValueError(f"Invalid checkpoint SHA-256 in provenance: {sidecar}")
        identity = {
            "run_id": provenance.get("run_id"),
            "checkpoint_sha256": checkpoint_sha,
            "evaluation_config_sha256": provenance.get("evaluation_config_sha256"),
            "dataset": provenance["dataset"],
            "mode": provenance["mode"],
        }
        if provenance.get("evaluation_id") != content_sha256(identity):
            raise ValueError(f"Evaluation ID conflicts with provenance fields: {sidecar}")
        if checkpoint_path and Path(checkpoint_path).is_file() and file_sha256(checkpoint_path) != checkpoint_sha:
            raise ValueError(f"Checkpoint SHA-256 conflicts with provenance: {sidecar}")
        record.update({
            "provenance_file": str(sidecar),
            "run_id": provenance.get("run_id"),
            "checkpoint_sha256": checkpoint_sha,
            "config_sha256": provenance.get("config_sha256"),
            "git_sha": provenance.get("git_sha"),
        })
    return record


def _collect_metric_files(root: Path) -> list[dict[str, Any]]:
    """Discover per-dataset results; exclude evaluator summaries to avoid double counting."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Evaluator result root does not exist: {root}")
    records = []
    run_identities: dict[str, tuple[str, int, str]] = {}
    for path in sorted(root.rglob("*.json")):
        if path.name == "summary.json":
            continue
        if path.name.endswith(".provenance.json"):
            metric_path = path.with_name(path.name.replace(".provenance.json", ".json"))
            if not metric_path.is_file():
                raise ValueError(f"Orphan evaluator provenance sidecar: {path}")
            continue
        record = _parse_metric_file(path, root)
        run_id = record.get("run_id")
        if run_id is not None:
            config_hash = record.get("config_sha256")
            if not isinstance(config_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", config_hash):
                raise ValueError(f"Missing config hash for run ID {run_id}.")
            identity = (record["experiment"], record["seed"], config_hash)
            if run_id in run_identities and run_identities[run_id] != identity:
                raise ValueError(f"Conflicting experiment, seed, or config hashes for run ID {run_id}.")
            run_identities[run_id] = identity
        records.append(record)
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
    identity_headers = ["source_file", "experiment", "seed", "dataset", "mode"]
    trace_headers = [
        key for key in ("provenance_file", "run_id", "checkpoint_sha256", "config_sha256", "git_sha")
        if any(key in record for record in records)
    ]
    headers = [*identity_headers, *trace_headers, *metric_keys]
    rows = [
        {**{key: record.get(key, "") for key in (*identity_headers, *trace_headers)},
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
