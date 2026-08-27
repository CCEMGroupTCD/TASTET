"""Kernel PCA projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import KernelPCA


@dataclass
class KPCAResult:
    """Container for kPCA output.

    :ivar projections: Low-dimensional coordinates.
    :vartype projections: ndarray, shape (N, n_components)
    :ivar eigenvalues: Eigenvalues of the centered kernel matrix.
    :vartype eigenvalues: ndarray, shape (n_components,)
    :ivar explained_variance: Fraction of total variance per component.
    :vartype explained_variance: ndarray, shape (n_components,)
    """

    projections: np.ndarray
    eigenvalues: np.ndarray
    explained_variance: np.ndarray


def fit_kpca(K: np.ndarray, *, n_components: int = 2) -> KPCAResult:
    """Run Kernel PCA on a precomputed kernel matrix.

    :param K: Precomputed kernel matrix of shape ``(N, N)``.
    :param n_components: Number of principal components to keep.
    :returns: kPCA output containing projections, eigenvalues, and explained
        variance.

    .. note::

        scikit-learn's ``KernelPCA.eigenvalues_`` are the eigenvalues of the
        centered kernel matrix :math:`H K H`, where
        :math:`H = I - \\frac{1}{N}\\mathbf{1}\\mathbf{1}^T`.

        The explained variance returned here divides each retained eigenvalue
        by :math:`\\operatorname{Tr}(H K H)`, the total centered-kernel
        variance across all components. Since

        :math:`\\operatorname{Tr}(H K H) = \\operatorname{Tr}(K)
        - \\frac{1}{N}\\sum_{ij} K_{ij}`,

        this denominator gives the fraction of total centered-kernel variance,
        matching the explained-variance label used in plots. Normalizing by
        only the retained eigenvalues is correct only when all components are
        kept; with truncated ``n_components`` it can overstate the explained
        fraction.
    """
    kpca = KernelPCA(n_components=n_components, kernel="precomputed")
    projections = kpca.fit_transform(K)
    eigenvalues = kpca.eigenvalues_

    N = K.shape[0]

    # Total variance of the centered kernel, Tr(H K H).
    total_centered_variance = np.trace(K) - (1.0 / N) * np.sum(K)

    if total_centered_variance > 0:
        explained = eigenvalues / total_centered_variance
    else:
        explained = np.zeros_like(eigenvalues)

    return KPCAResult(
        projections=projections,
        eigenvalues=eigenvalues,
        explained_variance=explained,
    )
