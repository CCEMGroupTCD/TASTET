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
) -> tuple[pd.DataFrame, np.ndarray]:
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
    :returns: ``(selected, idx_pool)`` where *selected* is a DataFrame
        of chosen rows with an ``array_index`` column, and *idx_pool*
        contains indices of all structures that passed the filter.
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

    selected = meta.iloc[global_idx].copy()
    selected["array_index"] = global_idx
    return selected, idx_pool


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

    When *color_values* is provided, the pool is coloured (with the
    colourbar anchored to the full dataset range) and selected points
    are overlaid in magenta.  When *None*, the pool is grey and
    selections are magenta — contrast from size alone.

    :param proj_df: Full projections DataFrame (all structures).
    :param idx_pool: Indices of structures that passed the filter.
    :param selected_indices: Indices of selected structures.
    :param explained_variance_pct: Explained variance per component (%).
    :param color_values: Per-point scalar for the full dataset (not subsetted).
        *None* = grey scatter.
    :param color_label: Colorbar label.
    :param save_path: Save figure here.
    :param show: Call ``plt.show()``.
    """
    set_mpl_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    kpc1 = proj_df["kpc1"].values
    kpc2 = proj_df["kpc2"].values

    if color_values is not None:
        # Colourbar anchored to full dataset range
        vmin, vmax = float(color_values.min()), float(color_values.max())
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        scatter = ax.scatter(
            kpc1[idx_pool], kpc2[idx_pool],
            c=color_values[idx_pool],
            s=60, alpha=0.7, edgecolors="none", cmap=cmap, norm=norm,
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(color_label)
    else:
        ax.scatter(
            kpc1[idx_pool], kpc2[idx_pool],
            c="#999999", s=20, edgecolors="none", zorder=1,
        )

    # Selected — magenta, larger
    s_size = 120 if color_values is not None else 60
    ax.scatter(
        kpc1[selected_indices], kpc2[selected_indices],
        c=palette["magenta"], s=s_size, alpha=0.9,
        edgecolors="none", zorder=5,
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