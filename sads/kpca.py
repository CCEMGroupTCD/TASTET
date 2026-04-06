"""Kernel PCA projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import KernelPCA


@dataclass
class KPCAResult:
    """Container for kPCA output.

    Attributes
    ----------
    projections : ndarray, shape (N, n_components)
        Low-dimensional coordinates.
    eigenvalues : ndarray, shape (n_components,)
        Eigenvalues of the centred kernel matrix.
    explained_variance : ndarray, shape (n_components,)
        Fraction of total variance per component.
    """

    projections: np.ndarray
    eigenvalues: np.ndarray
    explained_variance: np.ndarray


def fit_kpca(K: np.ndarray, *, n_components: int = 2) -> KPCAResult:
    """Run Kernel PCA on a precomputed kernel matrix.

    Parameters
    ----------
    K : ndarray, shape (N, N)
        Precomputed kernel matrix.
    n_components : int
        Number of principal components to keep.

    Returns
    -------
    KPCAResult
    """
    # 1. Fit the standard kPCA
    kpca = KernelPCA(n_components=n_components, kernel="precomputed")
    projections = kpca.fit_transform(K)
    eigenvalues = kpca.eigenvalues_

    # 2. Calculate the true total variance
    N = K.shape[0]
    # The trace of the centered kernel matrix equals the sum of ALL its eigenvalues.
    # Mathematically: Tr(K_centered) = Tr(K) - (1/N) * Sum(K)
    true_total_variance = np.trace(K) - (1.0 / N) * np.sum(K)

    # 3. Calculate the accurate explained variance
    if true_total_variance > 0:
        explained = eigenvalues / true_total_variance
    else:
        explained = np.zeros_like(eigenvalues)

    return KPCAResult(
        projections=projections,
        eigenvalues=eigenvalues,
        explained_variance=explained,
    )