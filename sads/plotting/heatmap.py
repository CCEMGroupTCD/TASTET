"""Heatmap visualisation for SOAP × kernel sweep results.

Produces multi-panel figures: one subplot per unique combination of
the ``group_by`` columns, with the chosen ``x`` / ``y`` columns on
the axes and cell colour mapped to the score.

When *x*, *y*, and *group_by* are omitted, :func:`infer_heatmap_layout`
inspects the DataFrame to choose automatically:

* Columns with a single unique value are dropped (fixed parameters).
* Categorical columns (strings) go to *group_by*.
* Among the remaining numeric columns, the two with the most unique
  values become *x* and *y*; ties are broken by a domain-aware
  priority order.
* Everything left goes to *group_by*.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FixedLocator, FixedFormatter

from sads.plotting.style import set_mpl_style, savefig, cmap as project_cmap


# ------------------------------------------------------------------
# Axis-label prettifiers
# ------------------------------------------------------------------

DEFAULT_LABEL_MAP: dict[str, str] = {
    "n_max":          r"$n_{\mathrm{max}}$",
    "l_max":          r"$l_{\mathrm{max}}$",
    "r_cut":          r"$r_{\mathrm{cut}}$ (Å)",
    "sigma":          r"$\sigma$ (Å)",
    "alpha":          r"$\alpha$",
    "gamma":          r"$\gamma$",
    "cka":            "CKA",
    "dissimilarity":  r"$1 - K$",
}

# Preferred order when two numeric columns have the same number of
# unique values.  Earlier = more likely to land on an axis.
_AXIS_PRIORITY: list[str] = [
    "r_cut", "sigma", "n_max", "l_max", "gamma", "alpha",
]


def _pretty_label(col: str, label_map: Mapping[str, str] | None) -> str:
    if label_map and col in label_map:
        return label_map[col]
    return DEFAULT_LABEL_MAP.get(col, col)


def _pretty_value(col: str, val) -> str:
    label = DEFAULT_LABEL_MAP.get(col, col)
    if isinstance(val, float):
        return f"{label} = {val:g}"
    return f"{label} = {val}"


# ------------------------------------------------------------------
# Layout inference
# ------------------------------------------------------------------

def _is_numeric_column(series: pd.Series) -> bool:
    """True if all non-null values in *series* are numeric."""
    vals = series.dropna().unique()
    if len(vals) == 0:
        return False
    return all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals)


def infer_heatmap_layout(
    df: pd.DataFrame,
    value: str,
    *,
    exclude: list[str] | None = None,
) -> tuple[str, str, list[str]]:
    """Choose *x*, *y*, and *group_by* from the DataFrame columns.

    Parameters
    ----------
    df : DataFrame
        Sweep results.
    value : str
        Score column (excluded from consideration).
    exclude : list of str, optional
        Additional columns to ignore (e.g. ``["status"]``).

    Returns
    -------
    x, y : str
        Columns for the heatmap axes.
    group_by : list of str
        Columns for subplots (may be empty).

    Raises
    ------
    ValueError
        If fewer than two numeric columns vary.
    """
    skip = {value, "status", *(exclude or [])}
    candidates = [c for c in df.columns if c not in skip]

    # Separate varying columns into numeric vs categorical
    numeric_varying: list[tuple[str, int]] = []   # (col, n_unique)
    categorical_varying: list[str] = []

    for col in candidates:
        n_unique = df[col].nunique(dropna=True)
        if n_unique <= 1:
            continue  # fixed parameter → ignore
        if _is_numeric_column(df[col]):
            numeric_varying.append((col, n_unique))
        else:
            categorical_varying.append(col)

    if len(numeric_varying) < 2:
        have = [c for c, _ in numeric_varying]
        raise ValueError(
            f"Need ≥ 2 varying numeric columns for a heatmap, "
            f"found {len(numeric_varying)}: {have}.  "
            f"Vary more parameters or pass x/y explicitly."
        )

    # Sort: most unique values first, then by domain priority for ties
    def _sort_key(item: tuple[str, int]) -> tuple[int, int]:
        col, n_unique = item
        priority = _AXIS_PRIORITY.index(col) if col in _AXIS_PRIORITY else len(_AXIS_PRIORITY)
        return (-n_unique, priority)

    numeric_varying.sort(key=_sort_key)

    x_col = numeric_varying[0][0]
    y_col = numeric_varying[1][0]
    group_by = [col for col, _ in numeric_varying[2:]] + categorical_varying

    return x_col, y_col, group_by


# ------------------------------------------------------------------
# Tick styling
# ------------------------------------------------------------------

def _apply_heatmap_ticks(ax, x_vals, y_vals) -> None:
    x_pos = np.arange(len(x_vals))
    y_pos = np.arange(len(y_vals))

    ax.xaxis.set_major_locator(FixedLocator(x_pos))
    ax.xaxis.set_major_formatter(
        FixedFormatter([f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in x_vals]),
    )
    ax.yaxis.set_major_locator(FixedLocator(y_pos))
    ax.yaxis.set_major_formatter(
        FixedFormatter([f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in y_vals]),
    )
    ax.tick_params(
        axis="both", which="both",
        direction="out", top=False, right=False,
    )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def plot_grid_heatmaps(
    df: pd.DataFrame,
    value: str,
    *,
    x: str | None = None,
    y: str | None = None,
    group_by: list[str] | None = None,
    out_path: Path | str | None = None,
    annotate: bool = False,
    fmt: str = ".3f",
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | mcolors.Colormap | None = None,
    label_map: dict[str, str] | None = None,
    colorbar_label: str | None = None,
    figsize_per_panel: tuple[float, float] = (5.0, 3.8),
    n_cols: int | None = None,
    suptitle: str | None = None,
    show: bool = False,
    dpi: int = 200,
) -> plt.Figure:
    """Plot sweep results as heatmaps.

    :param df: Sweep output from :func:`~sads.sweep.engine.run_sweep` or a CSV.
    :type df: pd.DataFrame
    :param value: Column name used for the cell color, typically the score.
    :type value: str
    :param x: Column for the horizontal axis. If ``None``,
        :func:`infer_heatmap_layout` selects a numeric column with high variation.
    :type x: str | None
    :param y: Column for the vertical axis. If ``None``,
        :func:`infer_heatmap_layout` selects a numeric column with high variation.
    :type y: str | None
    :param group_by: Columns whose unique combinations define subplots. If ``None``,
        these are inferred automatically from categorical columns and remaining
        varying numeric columns.
    :type group_by: list[str] | None
    :param out_path: Path where the figure should be saved.
    :type out_path: Path | str | None
    :param annotate: Whether to print numeric values inside each cell.
    :type annotate: bool
    :param fmt: Format specification used for cell annotations.
    :type fmt: str
    :param vmin: Lower bound for the color scale. If omitted, it is inferred from
        ``value``.
    :type vmin: float | None
    :param vmax: Upper bound for the color scale. If omitted, it is inferred from
        ``value``.
    :type vmax: float | None
    :param cmap: Colormap to use. If ``None``, the project default gradient is used.
    :type cmap: str | matplotlib.colors.Colormap | None
    :param label_map: Mapping from column names to display labels.
    :type label_map: dict[str, str] | None
    :param colorbar_label: Label for the colorbar.
    :type colorbar_label: str | None
    :param figsize_per_panel: Size in inches for each subplot as
        ``(width, height)``.
    :type figsize_per_panel: tuple[float, float]
    :param n_cols: Maximum number of subplot columns. Defaults to
        ``min(n_panels, 4)``.
    :type n_cols: int | None
    :param suptitle: Figure-level title.
    :type suptitle: str | None
    :param show: Whether to call :func:`matplotlib.pyplot.show`.
    :type show: bool
    :param dpi: Resolution used when saving the figure.
    :type dpi: int

    :returns: The created matplotlib figure.
    :rtype: matplotlib.figure.Figure
    """
    set_mpl_style()

    # ── Auto-infer layout if needed ───────────────────────────────
    if x is None or y is None or group_by is None:
        auto_x, auto_y, auto_gb = infer_heatmap_layout(df, value)
        if x is None:
            x = auto_x
        if y is None:
            y = auto_y
        if group_by is None:
            group_by = auto_gb

    # Report what was chosen (useful during interactive use)
    print(f"  Heatmap layout: x={x}, y={y}, group_by={group_by}")

    # ── Filter to varying group_by columns only ───────────────────
    group_by = [c for c in group_by if df[c].nunique(dropna=True) > 1]

    # ── Resolve colourmap ─────────────────────────────────────────
    if cmap is None:
        use_cmap = project_cmap
    elif isinstance(cmap, str):
        use_cmap = plt.get_cmap(cmap)
    else:
        use_cmap = cmap

    # ── Group data ────────────────────────────────────────────────
    if group_by:
        grouped = df.groupby(group_by, dropna=False)
        group_keys = list(grouped.groups.keys())
    else:
        grouped = None
        group_keys = [None]

    n_panels = len(group_keys)
    if n_cols is None:
        n_cols = min(n_panels, 4)
    n_rows = math.ceil(n_panels / n_cols)

    # ── Axis tick values (shared across panels) ───────────────────
    x_vals = sorted(df[x].dropna().unique())
    y_vals = sorted(df[y].dropna().unique())

    # ── Colour limits ─────────────────────────────────────────────
    valid = pd.to_numeric(df[value], errors="coerce").dropna()
    _vmin = float(valid.min()) if len(valid) else 0.0
    _vmax = float(valid.max()) if len(valid) else 1.0
    if vmin is None:
        vmin = _vmin
    if vmax is None:
        vmax = _vmax
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # ── Figure ────────────────────────────────────────────────────
    fig_w = figsize_per_panel[0] * n_cols
    fig_h = figsize_per_panel[1] * n_rows
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        constrained_layout=True,
    )

    for idx, key in enumerate(group_keys):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]

        sub = grouped.get_group(key if isinstance(key, tuple) else (key,)) if grouped is not None else df

        pivot = sub.pivot_table(index=y, columns=x, values=value, aggfunc="max")
        pivot = pivot.reindex(index=y_vals, columns=x_vals)

        ax.imshow(
            pivot.values,
            origin="lower",
            aspect="auto",
            cmap=use_cmap,
            norm=norm,
            interpolation="nearest",
        )

        if annotate:
            _annotate_cells(ax, pivot.values, norm, fmt)

        _apply_heatmap_ticks(ax, x_vals, y_vals)

        if r == n_rows - 1:
            ax.set_xlabel(_pretty_label(x, label_map))
        else:
            ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel(_pretty_label(y, label_map))
        else:
            ax.set_yticklabels([])

        if group_by and key is not None:
            ax.set_title(_panel_title(group_by, key), pad=6)

    # ── Hide unused panels ────────────────────────────────────────
    for idx in range(n_panels, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    # ── Shared colorbar ───────────────────────────────────────────
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=use_cmap),
        ax=axes.ravel().tolist(),
        fraction=0.02,
        pad=0.03,
    )
    cbar.set_label(colorbar_label or _pretty_label(value, label_map))

    if suptitle:
        fig.suptitle(suptitle, fontweight="bold")

    if out_path is not None:
        savefig(fig, Path(out_path), dpi=dpi)
    if show:
        plt.show()

    return fig


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _annotate_cells(ax, grid: np.ndarray, norm, fmt: str) -> None:
    n_y, n_x = grid.shape
    for yi in range(n_y):
        for xi in range(n_x):
            val = grid[yi, xi]
            if np.isnan(val):
                ax.text(xi, yi, "N/A", ha="center", va="center",
                        fontsize=7, color="red", fontweight="bold")
            else:
                luminance = norm(val)
                ax.text(xi, yi, f"{val:{fmt}}", ha="center", va="center",
                        fontsize=7, color="white" if luminance < 0.45 else "black")


def _panel_title(group_by: list[str], key) -> str:
    if isinstance(key, tuple):
        parts = [_pretty_value(k, v) for k, v in zip(group_by, key)]
    else:
        parts = [_pretty_value(group_by[0], key)]
    return ",  ".join(parts)