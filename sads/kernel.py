"""Global similarity kernels from per-structure SOAP features."""

from __future__ import annotations

from typing import Literal, Sequence, Any

import numpy as np
from dscribe.kernels import AverageKernel, REMatchKernel
from tqdm import tqdm


def median_heuristic_gamma(
    soap_list: Sequence[np.ndarray],
    max_envs: int = 5000,
    rng_seed: int = 42,
) -> float:
    """Compute the RBF gamma via the median heuristic over local environments.

    Pools all per-atom SOAP vectors, (sub)samples if the total exceeds
    *max_envs*, computes pairwise squared Euclidean distances, and returns

        gamma = 1 / (2 * median_distance²)

    which is the standard median heuristic for
    ``K(x, y) = exp(-gamma ||x - y||²)``.

    :param soap_list: One feature matrix per structure.
    :param max_envs: Cap on pooled environments (random subsample).
    :param rng_seed: Seed for reproducible subsampling.
    :returns: Scalar gamma value.
    """
    from sklearn.metrics import pairwise_distances

    pooled = np.vstack(soap_list)

    if len(pooled) > max_envs:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(pooled), size=max_envs, replace=False)
        pooled = pooled[idx]

    D2 = pairwise_distances(pooled, metric="sqeuclidean")
    med = np.median(D2[np.triu_indices_from(D2, k=1)])

    if med == 0.0:
        raise ValueError(
            "Median squared distance is 0 — all environments are identical."
        )

    gamma = 1.0 / (2.0 * med)
    return float(gamma)


def resolve_kernel_params(
    soap_list: Sequence[np.ndarray],
    params: dict,
    verbose: bool = True,
) -> dict:
    """Return a copy of *params* with ``gamma="median"`` resolved to a float.

    :param soap_list: Per-structure SOAP feature matrices (needed only when
        ``gamma="median"``).
    :param params: Kernel parameter dict, e.g. ``cfg.KERNEL_PARAMS`` or a
        single row from ``KERNEL_GRID``.
    :param verbose: Print the resolved value.
    :returns: A *new* dict with ``gamma`` replaced by its numeric value
        when it was ``"median"``, or an unchanged copy otherwise.
    """
    params = dict(params)
    if params.get("gamma") == "median":
        gamma = median_heuristic_gamma(soap_list)
        params["gamma"] = gamma
        if verbose:
            print(f"  Median-heuristic γ = {gamma:.6g}")
    return params


def compute_kernel(
    soap_list: Sequence[np.ndarray],
    *,
    method: Literal["average", "rematch"] = "rematch",
    metric: str = "linear",
    alpha: float = 0.5,
    normalize: bool = True,
    verbose: bool = True,
    **metric_kwargs: Any,
) -> np.ndarray:
    """
    Build a global similarity kernel from per-structure SOAP features.

    :param soap_list: One feature matrix per structure, typically produced by
        :func:`sads.soap.compute_soap`.
    :param method: Global kernel type. Must be ``"average"`` or ``"rematch"``.
    :param metric: Pairwise local-environment metric understood by DScribe /
        scikit-learn, for example ``"linear"``, ``"rbf"``, or ``"polynomial"``.
    :param alpha: REMatch regularisation parameter. Only used when
        ``method == "rematch"``.
    :param normalize: Whether to normalise the kernel so that diagonal elements
        satisfy ``K[i, i] == 1``. The default should normally be used.
    :param verbose: Whether to display a progress bar over kernel rows.
    :param metric_kwargs: Additional keyword arguments forwarded to the DScribe
        kernel constructor for the selected local metric. Examples include
        ``gamma`` for ``"rbf"``, and ``degree``, ``gamma``, ``coef0`` for
        ``"polynomial"``.
    :returns: Symmetric kernel matrix of shape ``(N, N)``.
    :rtype: numpy.ndarray
    :raises ValueError: If ``method`` is not one of ``"average"`` or
        ``"rematch"``.

    .. warning::

        Downstream parts of the pipeline assume a normalised kernel with unit
        diagonal. In particular, kernel-induced distances, kernel PCA, and
        structure selection rely on ``K[i, i] == 1``. Set ``normalize=False`` only
        for low-level diagnostics or custom workflows that do not use those
        downstream steps.

    .. note::

        ``gamma="median"`` must be resolved *before* calling this function.
        Use :func:`resolve_kernel_params` to do so.
    """
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


def combine_kernels(
    kernels: Sequence[np.ndarray],
    mode: Literal["product", "sum", "weighted_sum"] = "product",
    *,
    weights: Sequence[float] | None = None,
    tol: float = 1e-8,
) -> np.ndarray:
    """Combine pre-normalised kernel matrices into a single kernel.

    Each input kernel is assumed to be normalised (diagonal ≈ 1).
    The combined kernel preserves this property when *weights* sum
    to 1 in the ``"weighted_sum"`` case:

    * ``"product"``      — element-wise (Hadamard) product. Because
      every input has unit diagonal, the product diagonal is also
      exactly 1.
    * ``"sum"``          — element-wise mean
      ``(K₁ + K₂ + … + Kₙ) / n``, so the diagonal remains 1.
    * ``"weighted_sum"`` — linear combination ``Σᵢ wᵢ Kᵢ``. With
      ``Σᵢ wᵢ = 1`` and unit-diagonal inputs the result has unit
      diagonal too; otherwise the diagonal scales with the weight
      sum and the diagnostic warning will fire.

    A diagnostic check verifies that the output diagonal is within
    *tol* of 1 and emits a warning otherwise.

    :param kernels: Two or more normalised kernel matrices of the same
        shape ``(N, N)``.
    :param mode: ``"product"`` for the Hadamard product, ``"sum"`` for
        the element-wise mean, or ``"weighted_sum"`` for a linear
        combination with explicit per-channel weights.
    :param weights: Per-channel weights, required iff
        ``mode == "weighted_sum"``. Must be the same length as
        *kernels*. Ignored for the other modes.
    :param tol: Tolerance for the diagonal-unity check.
    :returns: Combined kernel matrix, shape ``(N, N)``.
    :raises ValueError: If fewer than one kernel is provided, shapes
        are inconsistent, *mode* is unrecognised, or weights are
        missing/mis-sized for ``"weighted_sum"``.
    """
    import warnings

    if len(kernels) < 1:
        raise ValueError("Need at least one kernel matrix.")

    shape = kernels[0].shape
    for i, K in enumerate(kernels):
        if K.shape != shape:
            raise ValueError(
                f"Shape mismatch: kernel 0 is {shape}, kernel {i} is {K.shape}."
            )

    if len(kernels) == 1:
        return kernels[0].copy()

    if mode == "product":
        K_out = kernels[0].copy()
        for K in kernels[1:]:
            K_out *= K
    elif mode == "sum":
        K_out = sum(kernels) / len(kernels)
    elif mode == "weighted_sum":
        if weights is None:
            raise ValueError("weighted_sum requires explicit weights.")
        if len(weights) != len(kernels):
            raise ValueError(
                f"weighted_sum needs one weight per kernel; "
                f"got {len(weights)} for {len(kernels)} kernels."
            )
        K_out = np.zeros_like(kernels[0])
        for w, K in zip(weights, kernels):
            K_out = K_out + w * K
    else:
        raise ValueError(f"Unknown combine mode: {mode!r}")

    diag = np.diag(K_out)
    max_dev = float(np.max(np.abs(diag - 1.0)))
    if max_dev > tol:
        warnings.warn(
            f"Combined kernel diagonal deviates from 1 by up to {max_dev:.2e} "
            f"(tol={tol:.0e}).  Check that input kernels are normalised.",
            stacklevel=2,
        )

    return K_out