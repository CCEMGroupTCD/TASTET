"""Kernel-induced distance distribution analysis.

For a normalized kernel (K[i,i] ≈ 1), the squared kernel distance between
structures *i* and *j* is::

    d²(i, j) = K(i,i) + K(j,j) − 2K(i,j) = 2(1 − K(i,j))

so the kernel-induced distance is::

    d(i, j) = √(2(1 − K(i,j)))

Analyzing the distribution of *d* values reveals whether a SOAP + kernel
representation is:

* **too coarse** — all distances ≈ 0 (Dirac delta at 0),
* **too sharp**  — all distances ≈ √2 (Dirac delta at √2),
* **well-tuned** — broad or multimodal distribution.

The primary diagnostic is the *histogram shape*, not any scalar summary.
Use :func:`pairwise_distances` to extract the raw distribution and
:func:`pairwise_dataframe` to build a per-pair CSV for inspection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


D_MAX: float = float(np.sqrt(2.0))
"""Maximum kernel-induced distance for a normalized kernel: ``√2``."""


def pairwise_distances(K: np.ndarray) -> np.ndarray:
    """Compute kernel-induced distances for every unique pair.

    :param K: Normalized kernel matrix with diagonal approximately equal to 1.
    :type K: ndarray, shape (N, N)
    :returns: Distances for each unique pair, where
        :math:`d(i, j) = \sqrt{2(1 - K[i, j])}`.
    :rtype: ndarray, shape (N(N - 1)/2,)
    """
    n = K.shape[0]
    if n < 2:
        return np.array([])
    iu = np.triu_indices(n, k=1)
    one_minus_k = 1.0 - K[iu]
    np.clip(one_minus_k, 0.0, None, out=one_minus_k)
    return np.sqrt(2.0 * one_minus_k)


def pairwise_dataframe(
    K: np.ndarray,
    ids: np.ndarray | list,
) -> pd.DataFrame:
    """Build a DataFrame of all unique pairwise distances with structure IDs.

    :param K: Normalized kernel matrix.
    :type K: ndarray, shape (N, N)
    :param ids: Identifier for each structure, for example ``configuration_id``
        from the database. Row order must match ``K``.
    :type ids: array-like, shape (N,)
    :returns: DataFrame with columns ``id_i``, ``id_j``, and ``d``, sorted by
        ``d`` descending so the most dissimilar pairs appear first.
    :rtype: DataFrame
    """
    n = K.shape[0]
    if n < 2:
        return pd.DataFrame(columns=["id_i", "id_j", "d"])

    ids = np.asarray(ids)
    iu_i, iu_j = np.triu_indices(n, k=1)

    d = pairwise_distances(K)

    df = pd.DataFrame(
        {
            "id_i": ids[iu_i],
            "id_j": ids[iu_j],
            "d": d,
        }
    )
    return df.sort_values("d", ascending=False, ignore_index=True)
