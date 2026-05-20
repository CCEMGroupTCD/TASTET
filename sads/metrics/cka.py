"""Centred Kernel Alignment scorer."""

from __future__ import annotations

import numpy as np

from sads.cka import cka_score as _cka_score


class CKAScorer:
    """Score a kernel matrix against a target via CKA.

    :param str target_kernel: Kernel applied to the target vector before
        alignment. Must be ``"linear"`` or ``"rbf"``.
    """

    name: str = "cka"

    def __init__(self, target_kernel: str = "linear") -> None:
        if target_kernel not in ("linear", "rbf"):
            raise ValueError(
                f"target_kernel must be 'linear' or 'rbf', got {target_kernel!r}"
            )
        self.target_kernel = target_kernel

    def __call__(
        self, K: np.ndarray, target: np.ndarray | None = None,
    ) -> float | None:
        if target is None:
            raise ValueError("CKAScorer requires a target array (e.g. energies).")
        try:
            score = _cka_score(K, target, target_kernel=self.target_kernel)
            return float(score) if np.isfinite(score) else None
        except Exception:
            return None

    def __repr__(self) -> str:
        return f"CKAScorer(target_kernel={self.target_kernel!r})"