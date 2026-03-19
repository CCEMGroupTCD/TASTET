"""Global similarity kernels from per-structure SOAP features."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from dscribe.kernels import AverageKernel, REMatchKernel
from tqdm import tqdm


def compute_kernel(
    soap_list: Sequence[np.ndarray],
    *,
    method: Literal["average", "rematch"] = "rematch",
    metric: str = "linear",
    gamma: float | None = None,
    alpha: float = 0.5,
    normalize: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """Build a global similarity kernel from per-structure SOAP features.

    Parameters
    ----------
    soap_list : sequence of ndarray
        One feature matrix per structure (from :func:`sads.soap.compute_soap`).
    method : ``"average"`` or ``"rematch"``
        Global kernel type.
    metric : str
        Pairwise local-environment metric (``"linear"``, ``"rbf"``, …).
    gamma : float, optional
        Metric width (passed to dscribe when not *None*).
    alpha : float
        REMatch regularisation (only used when *method* = ``"rematch"``).
    normalize : bool
        Row/column-normalise the kernel so that ``K[i,i] == 1``.
    verbose : bool
        Show a ``tqdm`` progress bar.

    Returns
    -------
    ndarray, shape (N, N)
        Symmetric positive-semidefinite kernel matrix.
    """
    metric_kwargs: dict = {}
    if gamma is not None:
        metric_kwargs["gamma"] = gamma

    if method == "average":
        kern = AverageKernel(metric=metric, **metric_kwargs)
    elif method == "rematch":
        kern = REMatchKernel(metric=metric, alpha=alpha, **metric_kwargs)
    else:
        raise ValueError(f"Unknown kernel method: {method!r}")

    N = len(soap_list)
    K = np.zeros((N, N))

    iterator = tqdm(range(N), desc="Kernel rows") if verbose else range(N)
    for i in iterator:
        for j in range(i, N):
            y_j = None if j == i else soap_list[j]
            C_ij = kern.get_pairwise_matrix(soap_list[i], y_j)
            k_ij = kern.get_global_similarity(C_ij)
            K[i, j] = k_ij
            if j != i:
                K[j, i] = k_ij

    if normalize:
        diag_sqrt = np.sqrt(np.diag(K))
        K /= np.outer(diag_sqrt, diag_sqrt)

    return K
