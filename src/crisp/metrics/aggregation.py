"""
Metric aggregation helpers.

This module standardizes:
- per-batch accumulation,
- per-dataset reduction,
- multi-seed aggregation,
- export-friendly formatting.

It is especially useful when regenerating paper tables from many runs.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List


def average_metric_dicts(metric_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Average a list of metric dictionaries with matching keys.

    Parameters
    ----------
    metric_dicts:
        List of dictionaries where each key maps to a numeric metric value.

    Returns
    -------
    Dict[str, float]
        Dictionary of averaged metrics.
    """
    if not metric_dicts:
        return {}

    accum: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)

    for d in metric_dicts:
        for k, v in d.items():
            accum[k] += float(v)
            counts[k] += 1

    return {k: accum[k] / counts[k] for k in accum}
