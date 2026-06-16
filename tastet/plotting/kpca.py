"""kPCA scatter-plot helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from tastet.plotting.style import (
    set_mpl_style, apply_axis_style, savefig, cmap, palette, palette_2,
)

if TYPE_CHECKING:
    from tastet.kpca import KPCAResult


# Ordered list of distinct hex colors for the categorical legend path
# (facet, site_type, stability, site_label, …). Drawn from the main
# colour-blind-safe palette first, then the extended palette_2, so the
# most distinguishable colours are used before the lighter ones. Built
# from hex values and de-duplicated, so a colour shared by both
# palettes (e.g. a "blue") is only listed once. With ~17 distinct
# colours this covers the full site-label set without cycling.
def _build_categorical_colors() -> list[str]:
    """Order palette colors by mutual contrast, de-duplicated to hex.

    The lead colors form a maximally-distinguishable sequence — dark
    blue then orange, the same two hues that anchor the continuous
    :data:`cmap` extremes — so a *binary* categorical column (e.g.
    ``facet`` 100/111) reads as a clear blue-vs-orange split rather than
    the two near-identical oranges the raw palette order produced. The
    remaining ``palette`` and ``palette_2`` colors follow for higher
    class counts, de-duplicated so a hue shared by both palettes is
    listed once.

    :returns: Hex colour strings ordered for categorical legends.
    """
    # High-contrast lead (mirrors the cmap's blue->orange extremes);
    # the trailing palette.values() is a no-op for the current palette
    # but keeps any future palette key from being dropped.
    lead = (
        palette["dark blue"],   # #0072B2  (cmap low extreme)
        palette["orange"],      # #E69F00  (cmap high extreme)
        palette["green"],       # #019E74
        palette["magenta"],     # #CC79A7
        palette["dark orange"], # #D55E00
        palette["blue"],        # #57B4E9
        palette["yellow"],      # #F0E442
        palette["black"],       # #000000
    )
    seen: set[str] = set()
    ordered: list[str] = []
    for hex_color in (*lead, *palette.values(), *palette_2.values()):
        if hex_color not in seen:
            seen.add(hex_color)
            ordered.append(hex_color)
    return ordered


_CATEGORICAL_COLORS: list[str] = _build_categorical_colors()


def _legend_ncol(n_entries: int, max_per_row: int = 6) -> int:
    """Columns for a horizontal bottom legend.

    Spreads *n_entries* across at most *max_per_row* columns so a long
    site-label list wraps into a few short rows instead of one tall
    single column. Capped at *max_per_row* and at the entry count.

    :param n_entries: Number of legend entries (distinct classes).
    :param max_per_row: Maximum columns in a single row.
    :returns: Column count for ``ax.legend(ncol=...)``, at least 1.
    """
    return max(1, min(max_per_row, n_entries))


def _categorical_color_map(values: np.ndarray) -> dict:
    """Map each distinct value to a color from the chained palettes.

    Distinct values are sorted for a stable, reproducible legend order
    (so ``facet`` 100 / 111 and ``site_label`` strings always get the
    same colors across runs). Colors are drawn from ``palette`` then
    ``palette_2`` (see :data:`_CATEGORICAL_COLORS`); they only cycle if
    there are more classes than the ~17 available distinct colours, in
    which case a warning is emitted since the scatter becomes
    ambiguous.

    :param values: Per-point categorical labels (strings or ints).
    :returns: ``{value: hex_color}``.
    """
    uniques = sorted(set(values.tolist()), key=lambda v: (str(type(v)), v))
    n_colors = len(_CATEGORICAL_COLORS)
    if len(uniques) > n_colors:
        import warnings
        warnings.warn(
            f"{len(uniques)} categories but only {n_colors} distinct colors; "
            f"colors will repeat and the scatter may be ambiguous. Consider "
            f"grouping categories or adding a marker dimension.",
            stacklevel=2,
        )
    return {
        val: _CATEGORICAL_COLORS[i % n_colors]
        for i, val in enumerate(uniques)
    }


def _plot_categorical_2d(ax, projections, values, color_map) -> None:
    """Scatter one series per class so each gets a legend entry."""
    for val, color in color_map.items():
        mask = values == val
        ax.scatter(
            projections[mask, 0], projections[mask, 1],
            c=color, label=str(val),
            s=60, alpha=0.7, edgecolors="none",
        )


def _plot_categorical_3d(ax, projections, values, color_map) -> None:
    """3-D variant of :func:`_plot_categorical_2d`."""
    for val, color in color_map.items():
        mask = values == val
        ax.scatter(
            projections[mask, 0], projections[mask, 1], projections[mask, 2],
            c=color, label=str(val),
            s=60, alpha=0.7, edgecolors="none",
        )


# Marker shapes for the second categorical channel (e.g. site type
# B/F/H/Q/T), used when color encodes a different attribute (e.g.
# composition). Distinct, filled, easily told apart at small size.
_MARKER_SHAPES: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">")


def _marker_map(values: np.ndarray) -> dict:
    """Map each distinct value to a marker shape (sorted, stable order)."""
    uniques = sorted(set(values.tolist()), key=lambda v: (str(type(v)), v))
    n = len(_MARKER_SHAPES)
    if len(uniques) > n:
        import warnings
        warnings.warn(
            f"{len(uniques)} marker categories but only {n} shapes; "
            f"shapes will repeat.",
            stacklevel=2,
        )
    return {val: _MARKER_SHAPES[i % n] for i, val in enumerate(uniques)}


def _continuous_norm(values: np.ndarray, *, vmin=None, vmax=None):
    """Build a colorbar norm for a continuous color channel.

    Uses explicit *vmin* / *vmax* when given (e.g. to pin ``% Al`` to
    0–100 for cross-subset comparability); otherwise derives the range
    from the finite *values* so an arbitrary channel such as ``delta_g``
    spans its own data range instead of being squashed against a fixed
    0–100 scale. Falls back to a unit range when no finite value is
    present or the range is degenerate.

    :param values: Per-point continuous values (may contain NaNs).
    :param vmin: Lower-bound override, or ``None`` to use the data min.
    :param vmax: Upper-bound override, or ``None`` to use the data max.
    :returns: A :class:`matplotlib.colors.Normalize` instance.
    """
    from matplotlib.colors import Normalize
    finite = np.isfinite(values)
    lo = vmin if vmin is not None else (
        float(np.min(values[finite])) if finite.any() else 0.0
    )
    hi = vmax if vmax is not None else (
        float(np.max(values[finite])) if finite.any() else 1.0
    )
    if hi <= lo:
        hi = lo + 1.0
    return Normalize(vmin=lo, vmax=hi)


def _two_legends(
    fig, ax, color_map: dict, marker_map: dict,
    color_label: str, marker_label: str, *, is_3d: bool,
):
    """Draw two stacked bottom legends: marker→class and color→class.

    The marker legend uses neutral grey glyphs (shape carries the
    meaning, not color); the color legend uses filled circles (color
    carries the meaning, not shape). Both sit below the axes,
    horizontally. Returns the number of legend rows reserved so the
    caller can size the bottom margin.
    """
    from matplotlib.lines import Line2D

    # Marker legend (top row block): grey glyphs, shape = category.
    marker_handles = [
        Line2D([0], [0], marker=m, color="none",
               markerfacecolor=palette.get("grey", "#7F7F7F")
               if "grey" in palette else "#7F7F7F",
               markeredgecolor="none", markersize=8, linestyle="none",
               label=str(val))
        for val, m in marker_map.items()
    ]
    # Color legend (lower block): filled circles, color = category.
    color_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=c, markeredgecolor="none",
               markersize=8, linestyle="none", label=str(val))
        for val, c in color_map.items()
    ]

    m_ncol = _legend_ncol(len(marker_handles))
    c_ncol = _legend_ncol(len(color_handles))
    m_rows = int(np.ceil(len(marker_handles) / m_ncol))
    c_rows = int(np.ceil(len(color_handles) / c_ncol))

    # Anchor BOTH legends in figure coordinates (not axes-fraction), so
    # the later subplots_adjust — which resizes the axes — does not
    # rescale the legend positions or the gap between them. Figure y=0 is
    # the bottom of the canvas; the reserved bottom margin (set by the
    # caller) is where the axes start. Place the marker legend just below
    # that margin and the color legend a fixed gap beneath it.
    fig = ax.figure
    ROW_H = 0.06    # figure-fraction height per legend row at this font
    GAP = 0.05      # extra gap between the two legend boxes
    marker_y = 0.30 if is_3d else 0.34   # top of marker legend (fig frac)
    color_y = marker_y - (m_rows * ROW_H + GAP)

    leg_marker = ax.legend(
        handles=marker_handles,
        frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, marker_y), bbox_transform=fig.transFigure,
        ncol=m_ncol,
    )
    ax.add_artist(leg_marker)  # keep first legend when adding the second

    ax.legend(
        handles=color_handles,
        frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, color_y), bbox_transform=fig.transFigure,
        ncol=c_ncol,
    )
    return m_rows + c_rows + 2  # +2 for the inter-legend gap


def _marker_legend(
    fig, ax, marker_map: dict, marker_label: str, *, is_3d: bool,
) -> int:
    """Draw a single bottom legend mapping marker shape → class.

    Companion to the colorbar in the continuous-color + categorical-
    marker mode: the colorbar carries the (continuous) colour meaning,
    so only the marker channel needs a legend. Glyphs are neutral grey
    because shape, not colour, is what this legend explains. Anchored in
    figure coordinates so the caller's later ``subplots_adjust`` does
    not rescale it. Returns the number of legend rows so the caller can
    size the reserved bottom margin.
    """
    from matplotlib.lines import Line2D

    grey = palette.get("grey", "#7F7F7F")
    handles = [
        Line2D([0], [0], marker=m, color="none",
               markerfacecolor=grey, markeredgecolor="none",
               markersize=8, linestyle="none", label=str(val))
        for val, m in marker_map.items()
    ]
    ncol = _legend_ncol(len(handles))
    n_rows = int(np.ceil(len(handles) / ncol))
    y = 0.02   # bottom of legend in figure coords (near the canvas edge)
    ax.legend(
        handles=handles, frameon=False, loc="lower center",
        bbox_to_anchor=(0.5, y), bbox_transform=fig.transFigure,
        ncol=ncol,
    )
    return n_rows


def _plot_color_marker_2d(ax, projections, color_values, marker_values,
                          color_map, marker_map) -> None:
    """Scatter with color = composition, marker = site type (2-D)."""
    for mval, marker in marker_map.items():
        for cval, color in color_map.items():
            mask = (marker_values == mval) & (color_values == cval)
            if not mask.any():
                continue
            ax.scatter(
                projections[mask, 0], projections[mask, 1],
                c=color, marker=marker,
                s=60, alpha=0.8, edgecolors="none",
            )


def _plot_color_marker_3d(ax, projections, color_values, marker_values,
                          color_map, marker_map) -> None:
    """3-D variant of :func:`_plot_color_marker_2d`."""
    for mval, marker in marker_map.items():
        for cval, color in color_map.items():
            mask = (marker_values == mval) & (color_values == cval)
            if not mask.any():
                continue
            ax.scatter(
                projections[mask, 0], projections[mask, 1], projections[mask, 2],
                c=color, marker=marker,
                s=60, alpha=0.8, edgecolors="none",
            )


def _plot_continuous_marker_2d(ax, projections, color_values, marker_values,
                               marker_map, cmap, norm) -> None:
    """Scatter with continuous color + categorical marker (2-D).

    Colour encodes a continuous value (e.g. percent Al, or delta_g)
    under a shared *norm*, so a single colorbar stays comparable across
    marker groups; marker shape encodes the categorical channel (e.g.
    site type). One scatter call per marker class, masking on the marker
    value.
    """
    for mval, marker in marker_map.items():
        mask = marker_values == mval
        if not mask.any():
            continue
        ax.scatter(
            projections[mask, 0], projections[mask, 1],
            c=color_values[mask], marker=marker, cmap=cmap, norm=norm,
            s=60, alpha=0.8, edgecolors="none",
        )


def _plot_continuous_marker_3d(ax, projections, color_values, marker_values,
                               marker_map, cmap, norm) -> None:
    """3-D variant of :func:`_plot_continuous_marker_2d`."""
    for mval, marker in marker_map.items():
        mask = marker_values == mval
        if not mask.any():
            continue
        ax.scatter(
            projections[mask, 0], projections[mask, 1], projections[mask, 2],
            c=color_values[mask], marker=marker, cmap=cmap, norm=norm,
            s=60, alpha=0.8, edgecolors="none",
        )


def plot_kpca(
    result: KPCAResult,
    *,
    color_values: np.ndarray | None = None,
    color_label: str = "",
    categorical: bool = False,
    marker_values: np.ndarray | None = None,
    marker_label: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    save: Path | str | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Scatter plot of the first two kPCA components.

    Coloring modes:

    * ``color_values is None`` → solid :data:`palette["blue"]` scatter.
    * ``categorical=False`` (default, *continuous*) → points colored by
      the project gradient with a colorbar (e.g. ``delta_g``).
    * ``categorical=True`` → each distinct value gets a discrete color
      and a bottom legend (e.g. ``facet``, ``stability``).
    * ``categorical=False`` **and** ``marker_values`` given →
      *continuous* color (colorbar) + categorical *marker shape* (bottom
      legend). Used by ``site_label`` (color = ``% Al``, marker = site
      type) and, with ``KPCA_MARKER_BY_SITE_TYPE`` set, by continuous
      columns such as ``delta_g`` (color = ``delta_g``, marker = site
      type). The colorbar range follows the data unless *vmin* / *vmax*
      are passed.
    * ``categorical=True`` **and** ``marker_values`` given → two
      categorical channels: *color* encodes ``color_values`` and
      *marker shape* encodes ``marker_values``, with two stacked
      bottom legends. Use this when one channel has too many classes
      for color alone.

    :param result: Output of :func:`tastet.kpca.fit_kpca`.
    :param color_values: Per-point value mapped onto color. ``None``
        produces a solid-color scatter in :data:`palette["blue"]`.
    :param color_label: Colorbar label (continuous) or legend title
        (categorical).
    :param categorical: Treat *color_values* as discrete classes.
    :param marker_values: Optional second channel mapped to marker
        shape. With ``categorical=False`` it pairs with a continuous
        colorbar; with ``categorical=True`` it pairs with a discrete
        color legend. Row-aligned to *color_values*.
    :param marker_label: Title for the marker legend.
    :param vmin: Lower bound for the continuous colorbar. ``None``
        derives it from the data; pass an explicit value (e.g. ``0``
        with ``vmax=100`` for a percentage channel) to pin the scale
        across runs.
    :param vmax: Upper bound for the continuous colorbar. ``None``
        derives it from the data.
    :param save: Save figure to this path. ``None`` skips saving.
    :param show: Call :func:`matplotlib.pyplot.show` after plotting.
    :returns: ``(fig, ax)``.
    """
    set_mpl_style()
    # Any path that places legend(s) below the axes reserves the margin
    # via subplots_adjust, which is incompatible with constrained_layout.
    # That covers every coloured path except the lone continuous colorbar
    # (no marker) and the uncoloured scatter. Enable constrained_layout
    # only for those; the rest set their margin manually.
    use_constrained = not (
        color_values is not None and (categorical or marker_values is not None)
    )
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=use_constrained)

    ev = result.explained_variance

    if color_values is None:
        ax.scatter(
            result.projections[:, 0],
            result.projections[:, 1],
            c=palette["blue"],
            s=60, alpha=0.7, edgecolors="none",
        )
    elif categorical and marker_values is not None:
        cvals = np.asarray(color_values)
        mvals = np.asarray(marker_values)
        color_map = _categorical_color_map(cvals)
        marker_map = _marker_map(mvals)
        _plot_color_marker_2d(
            ax, result.projections, cvals, mvals, color_map, marker_map,
        )
        n_rows = _two_legends(
            fig, ax, color_map, marker_map, color_label, marker_label,
            is_3d=False,
        )
        fig.subplots_adjust(bottom=0.44)
    elif categorical:
        values = np.asarray(color_values)
        color_map = _categorical_color_map(values)
        _plot_categorical_2d(ax, result.projections, values, color_map)
        ncol = _legend_ncol(len(color_map))
        ax.legend(
            frameon=False,
            loc="upper center", bbox_to_anchor=(0.5, -0.16),
            ncol=ncol,
        )
        # Reserve bottom margin for the legend, scaled to its row count.
        n_rows = int(np.ceil(len(color_map) / ncol))
        fig.subplots_adjust(bottom=min(0.5, 0.22 + 0.06 * n_rows))
    elif marker_values is not None:
        # Continuous color (colorbar) + categorical marker (legend).
        # Used by any continuous channel that should also encode site
        # type as marker shape: site_label colors by % Al, delta_g by
        # its own data range, etc.
        from matplotlib.cm import ScalarMappable
        cvals = np.asarray(color_values, dtype=float)
        mvals = np.asarray(marker_values)
        marker_map = _marker_map(mvals)
        norm = _continuous_norm(cvals, vmin=vmin, vmax=vmax)
        _plot_continuous_marker_2d(
            ax, result.projections, cvals, mvals, marker_map, cmap, norm,
        )
        n_rows = _marker_legend(fig, ax, marker_map, marker_label, is_3d=False)
        # Reserve the bottom margin BEFORE adding the colorbar so the
        # colorbar shrinks the already-adjusted axes from the right,
        # rather than being undone by a later subplots_adjust.
        fig.subplots_adjust(bottom=min(0.52, 0.30 + 0.06 * n_rows))
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        if color_label:
            cbar.set_label(color_label)
    else:
        scatter = ax.scatter(
            result.projections[:, 0],
            result.projections[:, 1],
            c=color_values,
            s=60, alpha=0.7, edgecolors="none", cmap=cmap,
        )
        cbar = fig.colorbar(scatter, ax=ax)
        if color_label:
            cbar.set_label(color_label)

    ax.set_xlabel(rf"kPC#1 ({ev[0] * 100:.1f}%)")
    ax.set_ylabel(rf"kPC#2 ({ev[1] * 100:.1f}%)")
    apply_axis_style(ax)

    if save is not None:
        savefig(fig, Path(save), dpi=300)
    if show:
        plt.show()

    return fig, ax


def plot_kpca_3d(
    result: KPCAResult,
    *,
    color_values: np.ndarray | None = None,
    color_label: str = "",
    categorical: bool = False,
    marker_values: np.ndarray | None = None,
    marker_label: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    save: Path | str | None = None,
    show: bool = False,
    elev: float = 30.0,
    azim: float = -60.0,
) -> tuple[Figure, Axes]:
    """Scatter plot of the first three kPCA components.

    Companion to :func:`plot_kpca`. Same call signature plus *elev* /
    *azim* for camera control. Each axis label carries the
    explained-variance percentage from ``result.explained_variance``,
    so the 3-D view immediately shows how much extra structure becomes
    visible by adding the third component.

    Same color modes as :func:`plot_kpca`, including the continuous
    color + categorical marker mode (``categorical=False`` with
    *marker_values*) and the dual categorical mode (``categorical=True``
    with *marker_values*).

    :param result: Output of :func:`tastet.kpca.fit_kpca`. Must have at
        least three components.
    :param color_values: Per-point value mapped onto color. ``None``
        produces a solid-color scatter in :data:`palette["blue"]`.
    :param color_label: Colorbar label (continuous) or legend title
        (categorical).
    :param categorical: Treat *color_values* as discrete classes.
    :param marker_values: Optional second channel mapped to marker
        shape (continuous colorbar when ``categorical=False``, discrete
        color legend when ``categorical=True``).
    :param marker_label: Title for the marker legend.
    :param vmin: Lower bound for the continuous colorbar. ``None``
        derives it from the data; pass an explicit value (e.g. ``0``
        with ``vmax=100`` for a percentage channel) to pin the scale
        across runs.
    :param vmax: Upper bound for the continuous colorbar. ``None``
        derives it from the data.
    :param save: Save figure to this path. ``None`` skips saving.
    :param show: Call :func:`matplotlib.pyplot.show` after plotting.
    :param elev: View elevation in degrees.
    :param azim: View azimuth in degrees.
    :returns: ``(fig, ax)``.
    :raises ValueError: If ``result.projections`` has fewer than three
        components.
    """
    if result.projections.shape[1] < 3:
        raise ValueError(
            "plot_kpca_3d needs at least 3 kPCA components; "
            f"got {result.projections.shape[1]}. Refit with n_components=3."
        )

    set_mpl_style()
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    ev = result.explained_variance

    if color_values is None:
        ax.scatter(
            result.projections[:, 0],
            result.projections[:, 1],
            result.projections[:, 2],
            c=palette["blue"],
            s=60, alpha=0.7, edgecolors="none",
        )
    elif categorical and marker_values is not None:
        cvals = np.asarray(color_values)
        mvals = np.asarray(marker_values)
        color_map = _categorical_color_map(cvals)
        marker_map = _marker_map(mvals)
        _plot_color_marker_3d(
            ax, result.projections, cvals, mvals, color_map, marker_map,
        )
        n_rows = _two_legends(
            fig, ax, color_map, marker_map, color_label, marker_label,
            is_3d=True,
        )
        fig.subplots_adjust(bottom=0.38)
    elif categorical:
        values = np.asarray(color_values)
        color_map = _categorical_color_map(values)
        _plot_categorical_3d(ax, result.projections, values, color_map)
        ncol = _legend_ncol(len(color_map))
        ax.legend(
            frameon=False,
            loc="upper center", bbox_to_anchor=(0.5, -0.06),
            ncol=ncol,
        )
        # 3-D axes have no constrained_layout, so reserve room at the
        # bottom for the horizontal legend (otherwise it clips on save,
        # which omits bbox_inches="tight" by design). Scale to row count.
        n_rows = int(np.ceil(len(color_map) / ncol))
        fig.subplots_adjust(bottom=min(0.4, 0.12 + 0.06 * n_rows))
    elif marker_values is not None:
        # Continuous color (colorbar) + categorical marker (legend).
        from matplotlib.cm import ScalarMappable
        cvals = np.asarray(color_values, dtype=float)
        mvals = np.asarray(marker_values)
        marker_map = _marker_map(mvals)
        norm = _continuous_norm(cvals, vmin=vmin, vmax=vmax)
        _plot_continuous_marker_3d(
            ax, result.projections, cvals, mvals, marker_map, cmap, norm,
        )
        n_rows = _marker_legend(fig, ax, marker_map, marker_label, is_3d=True)
        fig.subplots_adjust(bottom=min(0.45, 0.22 + 0.06 * n_rows))
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.1)
        if color_label:
            cbar.set_label(color_label)
    else:
        scatter = ax.scatter(
            result.projections[:, 0],
            result.projections[:, 1],
            result.projections[:, 2],
            c=color_values,
            s=60, alpha=0.7, edgecolors="none", cmap=cmap,
        )
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.1)
        if color_label:
            cbar.set_label(color_label)

    ax.set_xlabel(rf"kPC#1 ({ev[0] * 100:.1f}%)")
    ax.set_ylabel(rf"kPC#2 ({ev[1] * 100:.1f}%)")
    ax.set_zlabel(rf"kPC#3 ({ev[2] * 100:.1f}%)")

    if save is not None:
        savefig(fig, Path(save), dpi=300)
    if show:
        plt.show()

    return fig, ax