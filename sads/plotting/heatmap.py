"""Heatmap visualisation for SOAP × kernel sweep results.

Produces multi-panel figures: one subplot per unique combination of
the ``group_by`` columns, with the chosen ``x`` / ``y`` columns on
the axes and cell colour mapped to the score.

When *x*, *y*, and *group_by* are omitted, :func:`infer_heatmap_layout`
inspects the DataFrame to choose automatically:

* Columns with a single unique value are dropped (fixed parameters).
* Columns that are fully determined by other varying columns are
  treated as *derived* (e.g. ``gamma`` after the median heuristic
  resolves it from SOAP parameters) and excluded from axis candidates;
  they would not produce a meaningful heatmap.
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

from sads.plotting._panel import panel_title as _panel_title
from sads.plotting.style import set_mpl_style, savefig, cmap as project_cmap


# ------------------------------------------------------------------
# Axis-label prettifiers
# ------------------------------------------------------------------

# Axis-label rendering matches the SOAP-knob notation in
# sads.plotting._panel.soap_label, so heatmap axes and histogram
# panel titles use the same symbols (no trailing units; units belong
# in colorbar labels, not in axis labels for parameter sweeps).
DEFAULT_LABEL_MAP: dict[str, str] = {
    "n_max":  r"$n_{\max}$",
    "l_max":  r"$l_{\max}$",
    "r_cut":  r"$r_{\mathrm{cut}}$",
    "sigma":  r"$\sigma$",
    "alpha":  r"$\alpha$",
    "gamma":  r"$\gamma$",
    "degree": r"$d$",
    "cka":    "CKA",
}

# Preferred order when two numeric columns have the same number of
# unique values.  Earlier = more likely to land on an axis.
_AXIS_PRIORITY: list[str] = [
    "r_cut", "sigma", "n_max", "l_max", "gamma", "alpha",
]


def _pretty_label(col: str, label_map: Mapping[str, str] | None) -> str:
    """Return a display label for a parameter column.

    Strips any ``channel_name__`` prefix that multi-channel sweeps
    insert, so axis labels read ``$r_{\\mathrm{cut}}$`` regardless of
    which channel the swept knob belongs to.

    :param col: Column name, optionally ``channel_name__knob``.
    :param label_map: Optional overrides taking precedence over
        :data:`DEFAULT_LABEL_MAP`.
    :returns: Display label for the column.
    """
    if label_map and col in label_map:
        return label_map[col]
    bare = col.split("__", 1)[-1]
    return DEFAULT_LABEL_MAP.get(bare, bare)


# ------------------------------------------------------------------
# Layout inference
# ------------------------------------------------------------------

def _is_numeric_column(series: pd.Series) -> bool:
    """True if *series* holds numeric values.

    Uses :func:`pandas.api.types.is_numeric_dtype` first (the fast,
    canonical check), then falls back to coercing object-dtype columns
    with :func:`pandas.to_numeric`: sweep results assembled via
    ``pd.concat`` of mixed rows can land as object dtype even when every
    value is a clean float, and a per-value ``isinstance`` test is
    fragile in that case.

    :param series: Column to inspect.
    :returns: ``True`` when the column is numeric (natively or after
        coercion of all non-null values).
    """
    vals = series.dropna()
    if vals.empty:
        return False
    if pd.api.types.is_numeric_dtype(vals):
        return True
    coerced = pd.to_numeric(vals, errors="coerce")
    return bool(coerced.notna().all())


def _is_derived_column(
    df: pd.DataFrame, col: str, predictors: list[str],
) -> bool:
    """True if *col* is a deterministic function of *predictors*.

    A column is "derived" when grouping by the predictors yields at
    most one distinct value of *col* per group, ignoring NaNs. This
    catches cases like ``gamma`` resolved by the median heuristic from
    SOAP parameters: once the SOAP knobs are fixed, ``gamma`` is
    fixed too, so it has no independent variation to display on an
    axis.

    :param df: Sweep DataFrame.
    :param col: Candidate column name.
    :param predictors: Other varying columns to test against.
    :returns: ``True`` when *col* is determined by *predictors*.
    """
    if not predictors:
        return False
    sub = df[[*predictors, col]].dropna(subset=[col])
    if sub.empty:
        return False
    grouped = sub.groupby(predictors, dropna=False)[col].nunique(dropna=True)
    return bool((grouped <= 1).all())


def infer_heatmap_layout(
    df: pd.DataFrame,
    value: str,
    *,
    exclude: list[str] | None = None,
) -> tuple[str, str, list[str]]:
    """Choose *x*, *y*, and *group_by* from the DataFrame columns.

    :param df: Sweep results.
    :param value: Score column (excluded from consideration).
    :param exclude: Additional columns to ignore (e.g. ``["status"]``).
    :returns: Tuple ``(x, y, group_by)`` where *x* and *y* are column
        names for the heatmap axes and *group_by* is the (possibly
        empty) list of columns whose unique combinations define
        subplots.
    :raises ValueError: If fewer than two independently-varying
        numeric columns are available.
    """
    skip = {value, "status", *(exclude or [])}
    candidates = [c for c in df.columns if c not in skip]

    numeric_varying: list[tuple[str, int]] = []
    categorical_varying: list[str] = []

    for col in candidates:
        n_unique = df[col].nunique(dropna=True)
        if n_unique <= 1:
            continue
        if _is_numeric_column(df[col]):
            numeric_varying.append((col, n_unique))
        else:
            categorical_varying.append(col)

    # Drop numeric columns that are fully determined by other varying
    # columns (e.g. gamma resolved by median heuristic from SOAP
    # parameters). They can't be a meaningful axis: each combination
    # of the other knobs picks exactly one value, so the heatmap
    # would have one filled cell per row of the panel grid.
    other_varying = (
        [c for c, _ in numeric_varying] + categorical_varying
    )
    independent_numeric: list[tuple[str, int]] = []
    derived_numeric: list[str] = []
    for col, n_unique in numeric_varying:
        predictors = [c for c in other_varying if c != col]
        if _is_derived_column(df, col, predictors):
            derived_numeric.append(col)
        else:
            independent_numeric.append((col, n_unique))

    if derived_numeric:
        print(f"  Heatmap layout: dropping derived columns {derived_numeric}")

    if len(independent_numeric) < 2:
        have = [c for c, _ in independent_numeric]
        raise ValueError(
            f"Need ≥ 2 independently-varying numeric columns for a "
            f"heatmap, found {len(independent_numeric)}: {have}.  "
            f"Vary more parameters or pass x/y explicitly."
        )

    def _sort_key(item: tuple[str, int]) -> tuple[int, int]:
        col, n_unique = item
        priority = (
            _AXIS_PRIORITY.index(col)
            if col in _AXIS_PRIORITY else len(_AXIS_PRIORITY)
        )
        return (-n_unique, priority)

    independent_numeric.sort(key=_sort_key)

    x_col = independent_numeric[0][0]
    y_col = independent_numeric[1][0]
    group_by = (
        [col for col, _ in independent_numeric[2:]] + categorical_varying
    )

    return x_col, y_col, group_by


# ------------------------------------------------------------------
# Tick styling
# ------------------------------------------------------------------

def _apply_heatmap_ticks(ax, x_vals, y_vals) -> None:
    """Place fixed major ticks at every cell centre with ``%g`` labels.

    :param ax: Target axes.
    :param x_vals: Ordered x-axis cell values.
    :param y_vals: Ordered y-axis cell values.
    """
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
    :param value: Column name used for the cell colour, typically the score.
    :param x: Column for the horizontal axis. If ``None``,
        :func:`infer_heatmap_layout` selects a numeric column with high
        variation.
    :param y: Column for the vertical axis. If ``None``,
        :func:`infer_heatmap_layout` selects a numeric column with high
        variation.
    :param group_by: Columns whose unique combinations define subplots. If
        ``None``, these are inferred automatically from categorical columns and
        remaining varying numeric columns.
    :param out_path: Path where the figure should be saved.
    :param annotate: Whether to print numeric values inside each cell.
    :param fmt: Format specification used for cell annotations.
    :param vmin: Lower bound for the colour scale. If omitted, inferred from
        ``value``.
    :param vmax: Upper bound for the colour scale. If omitted, inferred from
        ``value``.
    :param cmap: Colormap to use. If ``None``, the project default gradient is
        used.
    :param label_map: Mapping from column names to display labels.
    :param colorbar_label: Label for the colorbar.
    :param figsize_per_panel: Size in inches for each subplot as
        ``(width, height)``.
    :param n_cols: Maximum number of subplot columns. Defaults to
        ``min(n_panels, 4)``.
    :param suptitle: Figure-level title.
    :param show: Whether to call :func:`matplotlib.pyplot.show`.
    :param dpi: Resolution used when saving the figure.
    :returns: The created matplotlib figure.
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
            if isinstance(key, tuple):
                title_params = dict(zip(group_by, key))
            else:
                title_params = {group_by[0]: key}
            title = _panel_title(title_params)
            n_lines = title.count("\n") + 1
            fontsize = 6 if n_lines > 1 else 8
            pad = 4 + 8 * (n_lines - 1)
            ax.set_title(title, fontsize=fontsize, pad=pad)

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
    """Write each cell's value at its centre, ``"N/A"`` for NaN cells.

    Text colour flips between white and black on the cell luminance so
    annotations stay legible across the colormap.

    :param ax: Target axes.
    :param grid: 2-D array of cell values (NaN where a combination is
        missing).
    :param norm: Normalizer mapping values to ``[0, 1]`` for the
        luminance test.
    :param fmt: Format spec applied to each numeric value.
    """
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