"""Centred Kernel Alignment (CKA) between a kernel and a target.

CKA measures the similarity between two kernels after centring, here
between a structure-space kernel ``K`` and a kernel built from a target
property vector ``y`` (e.g. energies). The score lies in ``[0, 1]``;
higher means the structure kernel's geometry aligns better with the
target's.
"""

from __future__ import annotations

import numpy as np


def _center_kernel(K: np.ndarray) -> np.ndarray:
    """Double-centre a kernel matrix: ``H K H`` with ``H = I − 11ᵀ/n``.

    :param K: Square kernel matrix.
    :returns: The centred kernel ``H @ K @ H``.
    """
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def cka_score(K: np.ndarray, y: np.ndarray, target_kernel: str = "rbf") -> float:
    """Centred Kernel Alignment between kernel *K* and target vector *y*.

    Builds a target kernel ``T`` from *y* (Gaussian/``"rbf"`` or
    linear), centres both kernels, and returns the normalised
    Frobenius inner product

    .. math::

        \\mathrm{CKA}(K, T) =
        \\frac{\\langle K_c, T_c \\rangle_F}
             {\\|K_c\\|_F \\, \\|T_c\\|_F}.

    For ``"rbf"`` the target bandwidth is set by the median heuristic
    over the off-diagonal target differences, with fallbacks to avoid
    division by zero.

    :param K: Structure-space kernel matrix, shape ``(N, N)``.
    :param y: Target property per structure, shape ``(N,)``.
    :param target_kernel: ``"rbf"`` (Gaussian on targets) or
        ``"linear"`` (outer product ``y yᵀ``).
    :returns: CKA score in ``[0, 1]``.
    :raises ValueError: If *target_kernel* is not ``"rbf"`` or
        ``"linear"``.
    """
    Kc = _center_kernel(K)

    if target_kernel == "rbf":
        # Gaussian kernel on targets
        diffs = y[:, None] - y[None, :]
        n = K.shape[0]
        mask = ~np.eye(n, dtype=bool)
        sigma = np.median(np.abs(diffs[mask]))
        if sigma == 0:  # fallback to avoid divide-by-zero
            sigma = np.std(y) + 1e-12
        T = np.exp(-(diffs**2) / (2 * sigma**2 + 1e-12))

    elif target_kernel == "linear":
        # Linear kernel: yy^T
        T = np.outer(y, y)

    else:
        raise ValueError("target_kernel must be 'rbf' or 'linear'")

    Tc = _center_kernel(T)
    similarity = (Kc * Tc).sum()
    normalization = (
        np.linalg.norm(Kc, ord="fro") * np.linalg.norm(Tc, ord="fro") + 1e-12
    )
    return float(similarity / normalization)