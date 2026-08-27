"""Scoring metrics for kernel quality evaluation.

Every scorer follows the same contract::

    scorer(K, target) → float | None

where *K* is a precomputed (optionally normalized) kernel matrix and
*target* is an optional reference array (e.g. energies).  Return
``None`` to signal that the score could not be computed.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["Scorer"]


@runtime_checkable
class Scorer(Protocol):
    """Minimal interface every metric must satisfy."""

    name: str

    def __call__(
        self, K: np.ndarray, target: np.ndarray | None = None,
    ) -> float | None: ...