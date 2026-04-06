"""Kernel dissimilarity scorer: 1 − K."""

from __future__ import annotations

import numpy as np


class DissimilarityScorer:
    """Score a normalised kernel by its off-diagonal dissimilarity.

    Assumes *K* has been row/column-normalised so that ``K[i, i] ≈ 1``.

    * **Two structures** (K is 2×2): returns ``1 − K[0, 1]``.
    * **N structures** (K is N×N): returns the mean ``1 − K[i, j]``
      over all off-diagonal pairs.

    Higher values → more discriminative descriptor/kernel.

    The *target* argument is accepted for protocol compatibility but
    ignored.
    """

    name: str = "dissimilarity"

    def __call__(
        self, K: np.ndarray, target: np.ndarray | None = None,
    ) -> float | None:
        n = K.shape[0]
        if n < 2:
            return None

        if n == 2:
            sim = K[0, 1]
        else:
            mask = ~np.eye(n, dtype=bool)
            sim = float(K[mask].mean())

        if not np.isfinite(sim):
            return None
        return 1.0 - sim

    def __repr__(self) -> str:
        return "DissimilarityScorer()"