"""
Top-level package for the CRISP replication scaffold.

This package contains:
- data loading and preprocessing utilities,
- backbone and projector model definitions,
- CRISP-specific mathematical modules,
- training and evaluation engines,
- metric computation and result export utilities.

The package is structured to support:
1. minimal faithful replication of the binary segmentation setting,
2. clean ablations of individual CRISP components,
3. future extension to additional backbones and tasks.
"""

__all__ = [
    "registry",
]

