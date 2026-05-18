"""Structure selection via diverse sampling in kernel space.

Supports k-medoids (cluster-based) and furthest point sampling
(greedy maximal diversity).  Handles optional energy filtering.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import kmedoids as _kmedoids

from sads.plotting.style import set_mpl_style, apply_axis_style, savefig, cmap, palette


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
    D = kernel_to_distance(K_sub)
    result = _kmedoids.fasterpam(D, k, random_state=seed)
    return np.array(result.medoids)


def _select_fps(K_sub: np.ndarray, k: int, seed: int) -> np.ndarray:
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
        mask = meta[energy_col].values <= energy_max
        idx_pool = np.where(mask)[0]
        print(f"  Energy filter: {mask.sum()}/{len(meta)} structures "
              f"with {energy_col} ≤ {energy_max}")
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


# ── Plotting ──────────────────────────────────────────────────────────

def plot_selection(
    proj_df: pd.DataFrame,
    idx_pool: np.ndarray,
    selected_indices: np.ndarray,
    explained_variance_pct: list[float],
    *,
    color_values: np.ndarray | None = None,
    color_label: str = "",
    save_path=None,
    show: bool = True,
) -> plt.Figure:
    """kPCA scatter showing the filtered pool with selections highlighted.

    Visually identical to :func:`sads.plotting.kpca.plot_kpca` for the
    base scatter (same figsize, same marker size and alpha, same
    palette colour), except that selected structures are overlaid in
    :data:`palette["pink"]` at twice the marker size. Reading
    ``selection.png`` next to ``kpca.png`` shows the same point cloud
    with the picks lit up — no other visual difference.

    When *color_values* is provided, the pool is coloured by the
    project gradient (anchored to the full dataset range); the
    overlaid selections stay pink so they remain visible against any
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
    :returns: The figure.
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
            kpc1[idx_pool], kpc2[idx_pool],
            c=color_values[idx_pool],
            s=60, alpha=0.7, edgecolors="none",
            cmap=cmap, norm=norm,
            zorder=1,
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(color_label)
    else:
        ax.scatter(
            kpc1[idx_pool], kpc2[idx_pool],
            c=palette["blue"],
            s=60, alpha=0.7, edgecolors="none",
            zorder=1,
        )

    # Selected — pink, larger, on top
    ax.scatter(
        kpc1[selected_indices], kpc2[selected_indices],
        c=palette["pink"],
        s=120, alpha=0.9, edgecolors="none",
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
    return fig


def plot_selection_3d(
    proj_df: pd.DataFrame,
    idx_pool: np.ndarray,
    selected_indices: np.ndarray,
    explained_variance_pct: list[float],
    *,
    color_values: np.ndarray | None = None,
    color_label: str = "",
    save_path=None,
    show: bool = True,
    elev: float = 30.0,
    azim: float = -60.0,
) -> plt.Figure:
    """3-D companion to :func:`plot_selection`.

    Visually identical to :func:`sads.plotting.kpca.plot_kpca_3d` for
    the base scatter; selections overlaid in :data:`palette["pink"]`
    at twice the marker size. ``depthshade=False`` on the selected
    layer keeps the pink markers visible even when they sit behind
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
    :returns: The figure.
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
        raise IndexError(
            "explained_variance_pct must have at least 3 entries."
        )

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
            kpc1[idx_pool], kpc2[idx_pool], kpc3[idx_pool],
            c=color_values[idx_pool],
            s=60, alpha=0.7, edgecolors="none",
            cmap=cmap, norm=norm, depthshade=True,
        )
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.1)
        cbar.set_label(color_label)
    else:
        ax.scatter(
            kpc1[idx_pool], kpc2[idx_pool], kpc3[idx_pool],
            c=palette["blue"],
            s=60, alpha=0.7, edgecolors="none",
            depthshade=True,
        )

    # Selected — pink, larger; depthshade=False keeps them visible
    # even when they sit behind dense regions of the cloud.
    ax.scatter(
        kpc1[selected_indices], kpc2[selected_indices], kpc3[selected_indices],
        c=palette["pink"],
        s=120, alpha=0.9, edgecolors="none",
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
    return fig