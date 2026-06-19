"""Structure selection via diverse sampling in kernel space.

Supports k-medoids (cluster-based) and furthest point sampling
(greedy maximal diversity).  Handles optional energy filtering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import kmedoids as _kmedoids

from tastet.plotting.style import (
    set_mpl_style,
    apply_axis_style,
    savefig,
    cmap,
    palette,
)


# ── Kernel helpers ────────────────────────────────────────────────────


def kernel_to_distance(K: np.ndarray) -> np.ndarray:
    """Convert a normalised kernel to a distance matrix.

    :param K: Kernel matrix, shape (N, N).
    :returns: Distance matrix, ``d(i,j) = sqrt(K[i,i] - 2K[i,j] + K[j,j])``.
    """
    diag = np.diag(K)
    D2 = diag[:, None] + diag[None, :] - 2 * K
    np.clip(D2, 0.0, None, out=D2)
    return np.sqrt(D2)


# ── Sampling backends ─────────────────────────────────────────────────


def _select_kmedoids(K_sub: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Select representative points using k-medoids clustering.

    Converts the input kernel matrix to a distance matrix and applies the
    FasterPAM k-medoids algorithm to select ``k`` medoids.

    :param np.ndarray K_sub: Square kernel matrix for the candidate subset.
    :param int k: Number of medoids to select.
    :param int seed: Random seed used to initialize the k-medoids algorithm.
    :returns: Indices of the selected medoids, relative to ``K_sub``.
    """
    D = kernel_to_distance(K_sub)
    result = _kmedoids.fasterpam(D, k, random_state=seed)
    return np.array(result.medoids)


def _select_fps(K_sub: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Select representative points using farthest point sampling.

    Starts from a randomly selected point and iteratively adds the point whose
    minimum squared distance to the selected set is largest. Squared distances
    are computed directly from the kernel matrix as

    .. math::

        d^2(i, j) = K_{ii} + K_{jj} - 2K_{ij}.

    :param np.ndarray K_sub: Square kernel matrix for the candidate subset.
    :param int k: Number of points to select.
    :param int seed: Random seed used to choose the initial point.
    :returns: Indices of the selected points, relative to ``K_sub``.
    """
    n = K_sub.shape[0]
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(n))]
    diag = np.diag(K_sub).copy()

    first = selected[0]
    min_d2 = diag + diag[first] - 2.0 * K_sub[:, first]
    np.maximum(min_d2, 0.0, out=min_d2)

    for _ in range(k - 1):
        best = int(np.argmax(min_d2))
        selected.append(best)
        d2_new = diag + diag[best] - 2.0 * K_sub[:, best]
        np.maximum(d2_new, 0.0, out=d2_new)
        np.minimum(min_d2, d2_new, out=min_d2)
        min_d2[best] = 0.0

    return np.array(selected)


_METHODS = {"kmedoids": _select_kmedoids, "fps": _select_fps}


# ── Selection ─────────────────────────────────────────────────────────


def select_structures(
    K: np.ndarray,
    meta: pd.DataFrame,
    *,
    energy_max: float | None = None,
    energy_col: str | None = None,
    energy_relative: bool = False,
    k: int = 10,
    method: Literal["kmedoids", "fps"] = "kmedoids",
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Filter (optionally) by energy, then pick *k* diverse structures.

    :param K: Kernel matrix, shape (N, N).  Row order matches *meta*.
    :param meta: Metadata DataFrame.
    :param energy_max: Keep structures with ``energy <= energy_max``.
        *None* = no filtering (all structures are candidates).
    :param energy_col: Column name for the energy filter.  Required when
        *energy_max* is set.
    :param energy_relative: When *True*, the threshold is applied to the
        column shifted to its own minimum, i.e. keep structures with
        ``energy - energy.min() <= energy_max``. Lets *energy_max* be
        expressed on an ``E - E_gm`` scale while the column stays raw.
    :param k: Number of representatives.
    :param method: ``"kmedoids"`` or ``"fps"``.
    :param seed: Random state.
    :returns: ``(selected, idx_pool, selected_indices)``.

        ``selected``
            DataFrame of the chosen rows (every column from *meta*),
            in selection order, with the index reset to 0..k-1. No
            row-position column is added — callers that need the
            kernel-row positions get them from *selected_indices*.

        ``idx_pool``
            Indices (into *meta*) of all structures that passed the
            energy filter; the candidate pool the picker drew from.

        ``selected_indices``
            Indices (into *meta* / kernel rows) of the chosen
            structures, in selection order. Aligned row-wise with
            *selected*.
    """
    if energy_max is not None:
        if not energy_col:
            raise ValueError("energy_col is required when energy_max is set.")
        if energy_col not in meta.columns:
            raise KeyError(f"Column {energy_col!r} not found in metadata.")
        values = meta[energy_col].values
        label = energy_col
        if energy_relative:
            values = values - values.min()
            label = f"{energy_col} - E_gm"
        mask = values <= energy_max
        idx_pool = np.where(mask)[0]
        print(
            f"  Energy filter: {mask.sum()}/{len(meta)} structures "
            f"with {label} ≤ {energy_max}"
        )
    else:
        idx_pool = np.arange(len(meta))

    if len(idx_pool) < k:
        print(f"  Warning: pool ({len(idx_pool)}) < k ({k}).  Selecting all.")
        k = len(idx_pool)

    if method not in _METHODS:
        raise ValueError(f"Unknown method {method!r}.  Choose from: {list(_METHODS)}")

    print(f"  Method: {method},  k={k},  pool={len(idx_pool)}")
    K_sub = K[np.ix_(idx_pool, idx_pool)]
    local_idx = _METHODS[method](K_sub, k, seed)
    global_idx = idx_pool[local_idx]

    selected = meta.iloc[global_idx].copy().reset_index(drop=True)
    return selected, idx_pool, global_idx


def _select_fps_seeded(
    K: np.ndarray,
    candidate_idx: np.ndarray,
    preselected_idx: np.ndarray,
    k: int,
) -> np.ndarray:
    """Farthest-point sampling warm-started from an existing selection.

    Picks *k* new points from ``candidate_idx`` that are maximally far
    from the union of ``preselected_idx`` and the points chosen so far.
    Distances are kernel-induced,
    :math:`d^2(i, j) = K_{ii} + K_{jj} - 2 K_{ij}`. There is no random
    initial point — the preselected set provides the warm start, so the
    result is deterministic. Operates on global indices into the full
    kernel ``K``.

    :param K: Full kernel matrix, shape (N, N).
    :param candidate_idx: Global indices eligible for selection.
    :param preselected_idx: Global indices already chosen (the seed set).
    :param k: Number of new points to select.
    :returns: Global indices of the *k* newly selected points, in
        selection order.
    """
    diag = np.diag(K)
    cand = np.asarray(candidate_idx, dtype=int)

    # Min squared distance from each candidate to the seed set.
    min_d2 = np.full(cand.shape, np.inf)
    for p in np.asarray(preselected_idx, dtype=int):
        d2 = diag[cand] + diag[p] - 2.0 * K[cand, p]
        np.minimum(min_d2, np.maximum(d2, 0.0), out=min_d2)

    chosen: list[int] = []
    for _ in range(k):
        best_local = int(np.argmax(min_d2))
        chosen.append(int(cand[best_local]))
        min_d2[best_local] = -1.0  # never pick this candidate again
        d2_new = diag[cand] + diag[cand[best_local]] - 2.0 * K[cand, cand[best_local]]
        np.minimum(min_d2, np.maximum(d2_new, 0.0), out=min_d2)

    return np.array(chosen, dtype=int)


def select_additional(
    K: np.ndarray,
    meta: pd.DataFrame,
    *,
    preselected_indices: np.ndarray | list,
    k: int,
    energy_max: float | None = None,
    energy_col: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Extend an existing selection by *k* more via seeded FPS.

    Picks *k* new structures that are maximally diverse with respect to
    the already-selected ``preselected_indices`` (and each other), using
    farthest-point sampling warm-started from that set. The preselected
    structures are excluded from the candidate pool and are not returned.
    Use this for incremental DFT campaigns: feed back the structures
    already computed and draw the next batch from what remains.

    :param K: Full kernel matrix, shape (N, N); row order matches *meta*.
    :param meta: Metadata DataFrame.
    :param preselected_indices: Row positions (into *meta* / kernel rows)
        of the already-selected structures. For the universal schema
        these are ``configuration_id - 1``.
    :param k: Number of additional structures to pick.
    :param energy_max: Optional upper bound for an energy filter applied
        to the candidate pool. *None* = no filtering.
    :param energy_col: Column name for the energy filter. Required when
        *energy_max* is set.
    :returns: ``(selected, idx_pool, selected_indices)`` — same shape as
        :func:`select_structures`, for the newly chosen rows. ``selected``
        is in selection order with the index reset; ``selected_indices``
        are the kernel-row positions of the new picks.
    """
    preselected = np.asarray(preselected_indices, dtype=int)

    if energy_max is not None:
        if not energy_col:
            raise ValueError("energy_col is required when energy_max is set.")
        if energy_col not in meta.columns:
            raise KeyError(f"Column {energy_col!r} not found in metadata.")
        mask = meta[energy_col].values <= energy_max
        print(
            f"  Energy filter: {int(mask.sum())}/{len(meta)} structures "
            f"with {energy_col} ≤ {energy_max}"
        )
    else:
        mask = np.ones(len(meta), dtype=bool)

    mask[preselected] = False  # never re-pick the seed set
    idx_pool = np.where(mask)[0]

    if len(idx_pool) < k:
        print(f"  Warning: pool ({len(idx_pool)}) < k ({k}).  Selecting all.")
        k = len(idx_pool)

    print(
        f"  Incremental FPS: k={k}, pool={len(idx_pool)}, "
        f"seeded by {len(preselected)} preselected"
    )
    global_idx = _select_fps_seeded(K, idx_pool, preselected, k)
    selected = meta.iloc[global_idx].reset_index(drop=True)
    return selected, idx_pool, global_idx


# ── Plotting ──────────────────────────────────────────────────────────


def plot_selection(
    proj_df: pd.DataFrame,
    idx_pool: np.ndarray,
    selected_indices: np.ndarray,
    explained_variance_pct: list[float],
    *,
    color_values: np.ndarray | None = None,
    color_label: str = "",
    save_path: Path | str | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """kPCA scatter showing the filtered pool with selections highlighted.

    Visually identical to :func:`tastet.plotting.kpca.plot_kpca` for the
    base scatter (same figsize, same marker size and alpha, same
    palette colour), except that selected structures are overlaid in
    :data:`palette["magenta"]` at twice the marker size. Reading
    ``selection.png`` next to ``kpca.png`` shows the same point cloud
    with the picks lit up — no other visual difference.

    When *color_values* is provided, the pool is coloured by the
    project gradient (anchored to the full dataset range); the
    overlaid selections stay magenta so they remain visible against any
    point colour.

    :param proj_df: Full projections DataFrame (all structures).
    :param idx_pool: Indices of structures that passed the filter.
    :param selected_indices: Indices of selected structures.
    :param explained_variance_pct: Explained variance per component (%).
    :param color_values: Per-point scalar for the full dataset (not
        subsetted). ``None`` = solid-colour pool (:data:`palette["blue"]`).
    :param color_label: Colorbar label.
    :param save_path: Save figure here. ``None`` skips saving.
    :param show: Call ``plt.show()``.
    :returns: ``(fig, ax)``.
    """
    set_mpl_style()
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    kpc1 = proj_df["kpc1"].values
    kpc2 = proj_df["kpc2"].values

    if color_values is not None:
        # Colourbar anchored to full dataset range
        vmin, vmax = float(color_values.min()), float(color_values.max())
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        scatter = ax.scatter(
            kpc1[idx_pool],
            kpc2[idx_pool],
            c=color_values[idx_pool],
            s=60,
            alpha=0.7,
            edgecolors="none",
            cmap=cmap,
            norm=norm,
            zorder=1,
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(color_label)
    else:
        ax.scatter(
            kpc1[idx_pool],
            kpc2[idx_pool],
            c=palette["blue"],
            s=60,
            alpha=0.7,
            edgecolors="none",
            zorder=1,
        )

    # Selected — magenta, larger, on top
    ax.scatter(
        kpc1[selected_indices],
        kpc2[selected_indices],
        c=palette["magenta"],
        s=120,
        alpha=0.9,
        edgecolors="none",
        zorder=5,
    )

    ev = explained_variance_pct
    ax.set_xlabel(rf"kPC#1 ({ev[0]:.1f}%)")
    ax.set_ylabel(rf"kPC#2 ({ev[1]:.1f}%)")
    apply_axis_style(ax)

    if save_path:
        savefig(fig, save_path)
    if show:
        plt.show()
    return fig, ax


def plot_selection_3d(
    proj_df: pd.DataFrame,
    idx_pool: np.ndarray,
    selected_indices: np.ndarray,
    explained_variance_pct: list[float],
    *,
    color_values: np.ndarray | None = None,
    color_label: str = "",
    save_path: Path | str | None = None,
    show: bool = False,
    elev: float = 30.0,
    azim: float = -60.0,
) -> tuple[Figure, Axes]:
    """3-D companion to :func:`plot_selection`.

    Visually identical to :func:`tastet.plotting.kpca.plot_kpca_3d` for
    the base scatter; selections overlaid in :data:`palette["magenta"]`
    at twice the marker size. ``depthshade=False`` on the selected
    layer keeps the magenta markers visible even when they sit behind
    dense regions of the cloud.

    :param proj_df: Full projections DataFrame (all structures).
        Must contain ``kpc1``, ``kpc2``, ``kpc3``.
    :param idx_pool: Indices of structures that passed the filter.
    :param selected_indices: Indices of selected structures.
    :param explained_variance_pct: Explained variance per component
        (%); must have at least three entries.
    :param color_values: Per-point scalar for the full dataset (not
        subsetted). ``None`` = solid-colour pool (:data:`palette["blue"]`).
    :param color_label: Colorbar label.
    :param save_path: Save figure here. ``None`` skips saving.
    :param show: Call ``plt.show()``.
    :param elev: 3-D view elevation, degrees.
    :param azim: 3-D view azimuth, degrees.
    :returns: ``(fig, ax)``.
    :raises KeyError: If ``proj_df`` does not contain a ``kpc3``
        column.
    :raises IndexError: If ``explained_variance_pct`` has fewer than
        three entries.
    """
    if "kpc3" not in proj_df.columns:
        raise KeyError(
            "plot_selection_3d needs a 'kpc3' column in proj_df. "
            "Rerun the kpca step (it now writes kpc1/kpc2/kpc3)."
        )
    if len(explained_variance_pct) < 3:
        raise IndexError("explained_variance_pct must have at least 3 entries.")

    set_mpl_style()
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    kpc1 = proj_df["kpc1"].values
    kpc2 = proj_df["kpc2"].values
    kpc3 = proj_df["kpc3"].values

    if color_values is not None:
        vmin, vmax = float(color_values.min()), float(color_values.max())
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        scatter = ax.scatter(
            kpc1[idx_pool],
            kpc2[idx_pool],
            kpc3[idx_pool],
            c=color_values[idx_pool],
            s=60,
            alpha=0.7,
            edgecolors="none",
            cmap=cmap,
            norm=norm,
            depthshade=True,
        )
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.1)
        cbar.set_label(color_label)
    else:
        ax.scatter(
            kpc1[idx_pool],
            kpc2[idx_pool],
            kpc3[idx_pool],
            c=palette["blue"],
            s=60,
            alpha=0.7,
            edgecolors="none",
            depthshade=True,
        )

    # Selected — magenta, larger; depthshade=False keeps them visible
    # even when they sit behind dense regions of the cloud.
    ax.scatter(
        kpc1[selected_indices],
        kpc2[selected_indices],
        kpc3[selected_indices],
        c=palette["magenta"],
        s=120,
        alpha=0.9,
        edgecolors="none",
        depthshade=False,
    )

    ev = explained_variance_pct
    ax.set_xlabel(rf"kPC#1 ({ev[0]:.1f}%)")
    ax.set_ylabel(rf"kPC#2 ({ev[1]:.1f}%)")
    ax.set_zlabel(rf"kPC#3 ({ev[2]:.1f}%)")

    if save_path:
        savefig(fig, save_path)
    if show:
        plt.show()
    return fig, ax
