"""Distance-distribution histograms for kernel representations.

Panel titles use compact kernel formulas (e.g. ``$(p_i·q_j)^5$``)
rather than verbose key=value strings. The formatting helpers live in
:mod:`sads.plotting._panel` and are shared with
:mod:`sads.plotting.heatmap`.

* :func:`plot_distance_histogram`     — single histogram (kernel step).
* :func:`plot_distance_histogram_kde` — same plus a Gaussian-KDE overlay
  and the diagnostic stats (n_peaks, IQR/√2) used by
  ``analyze_distances.py``.
* :func:`plot_grid_histograms`        — multi-panel figure (grid search).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sads.distance import pairwise_distances, D_MAX
from sads.plotting._panel import panel_title
from sads.plotting.style import set_mpl_style, apply_axis_style, savefig, palette


# ------------------------------------------------------------------
# Single histogram (kernel step)
# ------------------------------------------------------------------

def plot_distance_histogram(
    K: np.ndarray,
    *,
    bins: int = 50,
    title: str = "",
    out_path: Path | str | None = None,
    figsize: tuple[float, float] = (6, 4),
    show: bool = False,
    dpi: int = 200,
) -> plt.Figure:
    r"""Histogram of pairwise kernel distances ``d(i,j) = √(2(1−K))``.

    The x-range is fixed to ``[0, √2]`` — the theoretical bound for a
    normalised kernel — so that the position of the mass immediately
    conveys whether the representation is too coarse (near 0), too
    sharp (near √2), or well-tuned (spread / multimodal).

    :param K: Normalised kernel matrix, shape *(N, N)*.
    :param bins: Number of histogram bins.
    :param title: Figure title.
    :param out_path: Save path.
    :param figsize: Figure size in inches.
    :param show: Call ``plt.show()``.
    :param dpi: Resolution when saving.
    :returns: The figure.
    """
    set_mpl_style()
    d = pairwise_distances(K)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.hist(
        d, bins=bins, range=(0, D_MAX),
        color=palette["dark blue"], alpha=0.85,
        edgecolor="white", linewidth=0.4,
    )

    n_pairs = len(d)
    ax.text(
        0.97, 0.95,
        f"$N_{{\\mathrm{{pairs}}}}$ = {n_pairs}",
        transform=ax.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85),
    )

    ax.set_xlabel(r"$d(i,\,j)$")
    ax.set_ylabel("Count")
    ax.set_xlim(0, D_MAX)
    if title:
        ax.set_title(title, fontweight="bold")

    apply_axis_style(ax, xfmt=".1f", yfmt=".0f")

    if out_path is not None:
        savefig(fig, Path(out_path), dpi=dpi)
    if show:
        plt.show()
    return fig


def plot_distance_histogram_kde(
    K: np.ndarray,
    *,
    bins: int = 50,
    bandwidth: float = 0.02,
    title: str = "",
    out_path: Path | str | None = None,
    figsize: tuple[float, float] = (6, 4),
    show: bool = False,
    dpi: int = 200,
    n_grid: int = 500,
) -> plt.Figure:
    r"""Density-normalised histogram with a Gaussian-KDE overlay.

    Companion to :func:`plot_distance_histogram`. The histogram is
    rendered ``density=True`` so the KDE curve and the bars share a
    vertical axis. The KDE bandwidth is an absolute value in distance
    units; the default of ``0.02`` matches the bandwidth used in
    ``analyze_distances.py``.

    The annotation box shows the same diagnostic stats produced by
    ``analyze_distances.py``: ``n_peaks`` (KDE peaks above 10 % of the
    maximum density, via :func:`scipy.signal.find_peaks`) and
    ``IQR/√2`` (inter-quartile range normalised by the theoretical
    maximum distance).

    :param K: Normalised kernel matrix, shape *(N, N)*.
    :param bins: Number of histogram bins.
    :param bandwidth: Gaussian-KDE bandwidth in distance units.
    :param title: Figure title.
    :param out_path: Save path.
    :param figsize: Figure size in inches.
    :param show: Call ``plt.show()``.
    :param dpi: Resolution when saving.
    :param n_grid: Number of points on the KDE evaluation grid.
    :returns: The figure.
    """
    from scipy.signal import find_peaks
    from sklearn.neighbors import KernelDensity

    set_mpl_style()
    d = pairwise_distances(K)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.hist(
        d, bins=bins, range=(0, D_MAX), density=True,
        color=palette["dark blue"], alpha=0.40,
        edgecolor="white", linewidth=0.4,
    )

    # KDE overlay (constant bandwidth, evaluated on a shared grid).
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(d[:, None])
    x_grid = np.linspace(0.0, D_MAX, n_grid)
    density = np.exp(kde.score_samples(x_grid[:, None]))
    ax.plot(x_grid, density, color=palette["dark orange"], linewidth=1.6)
    ax.fill_between(x_grid, density, color=palette["dark orange"], alpha=0.15)

    # ── Diagnostic stats (same convention as analyze_distances.py) ──
    n_pairs = len(d)
    peaks, _ = find_peaks(density, prominence=0.1 * density.max())
    n_peaks = int(len(peaks))
    q75, q25 = np.percentile(d, [75, 25])
    iqr_norm = float((q75 - q25) / np.sqrt(2.0))

    annotation = (
        f"$N_{{\\mathrm{{pairs}}}}$ = {n_pairs}\n"
        f"bw = {bandwidth:g}\n"
        f"$n_{{\\mathrm{{peaks}}}}$ = {n_peaks}\n"
        f"$\\mathrm{{IQR}}/\\sqrt{{2}}$ = {iqr_norm:.3g}"
    )
    ax.text(
        0.97, 0.95, annotation,
        transform=ax.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85),
    )

    ax.set_xlabel(r"$d(i,\,j)$")
    ax.set_ylabel("Density")
    ax.set_xlim(0, D_MAX)
    if title:
        ax.set_title(title, fontweight="bold")

    apply_axis_style(ax, xfmt=".1f", yfmt=".1f")

    if out_path is not None:
        savefig(fig, Path(out_path), dpi=dpi)
    if show:
        plt.show()
    return fig


# ------------------------------------------------------------------
# Multi-panel grid search
# ------------------------------------------------------------------

def plot_grid_histograms(
    kernels: list[dict],
    *,
    bins: int = 40,
    figsize_per_panel: tuple[float, float] = (4.0, 3.0),
    title_fontsize: float | None = None,
    n_cols: int | None = None,
    out_path: Path | str | None = None,
    suptitle: str | None = None,
    show: bool = False,
    dpi: int = 200,
) -> plt.Figure:
    r"""One histogram per grid-search combination.

    All panels share the fixed ``[0, √2]`` x-range so that distributions
    are visually comparable across parameter combinations. Panel titles
    are built by :func:`sads.plotting._panel.panel_title`.

    :param kernels: List of dicts, each containing:

        * ``"K"``      — normalised kernel matrix,
        * ``"params"`` — dict of parameter-name → value for the title.

    :param bins: Bins per histogram.
    :param figsize_per_panel: Per-panel size ``(width, height)`` in inches.
    :param title_fontsize: Font size for panel titles.  *None* = auto
        (7.5 for single-kernel, 6 for multi-channel).
    :param n_cols: Max subplot columns (default ``min(n, 4)``).
    :param out_path: Save path.
    :param suptitle: Super-title.
    :param show: Call ``plt.show()``.
    :param dpi: Resolution when saving.
    :returns: The figure.
    :raises ValueError: If *kernels* is empty.
    """
    set_mpl_style()
    n = len(kernels)
    if n == 0:
        raise ValueError("No kernels to plot.")

    if n_cols is None:
        # Pick the column count (up to 4) that (1) wastes the fewest
        # cells, (2) is closest to square, (3) prefers wider when tied.
        max_cols = min(n, 4)
        min_cols = min(n, 2)
        n_cols = min(
            range(min_cols, max_cols + 1),
            key=lambda c: ((-n) % c, abs(math.ceil(n / c) - c), -c),
        )
    n_rows = math.ceil(n / n_cols)

    fig_w = figsize_per_panel[0] * n_cols
    fig_h = figsize_per_panel[1] * n_rows
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        constrained_layout=True,
    )

    for idx, entry in enumerate(kernels):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]

        d = pairwise_distances(entry["K"])
        ax.hist(
            d, bins=bins, range=(0, D_MAX),
            color=palette["dark blue"], alpha=0.85,
            edgecolor="white", linewidth=0.3,
        )

        title = panel_title(entry["params"])
        n_lines = title.count("\n") + 1
        fontsize = title_fontsize or (6 if n_lines > 1 else 7.5)
        pad = 4 + 8 * (n_lines - 1)
        ax.set_title(title, fontsize=fontsize, pad=pad)
        ax.set_xlim(0, D_MAX)

        ax.tick_params(
            axis="both", which="both",
            direction="out", top=False, right=False,
            labelsize=7,
        )

        if r == n_rows - 1:
            ax.set_xlabel(r"$d(i,\,j)$", fontsize=9)
        else:
            ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel("Count", fontsize=9)
        else:
            ax.set_yticklabels([])

    # ── Hide unused panels ────────────────────────────────────────
    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontweight="bold")

    if out_path is not None:
        savefig(fig, Path(out_path), dpi=dpi)
    if show:
        plt.show()
    return fig