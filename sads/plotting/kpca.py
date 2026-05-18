"""kPCA scatter-plot helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from sads.plotting.style import (
    set_mpl_style, apply_axis_style, savefig, cmap, palette,
)

if TYPE_CHECKING:
    from sads.kpca import KPCAResult


def plot_kpca(
    result: KPCAResult,
    *,
    color_values: np.ndarray | None = None,
    color_label: str = "",
    save: Path | str | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Scatter plot of the first two kPCA components.

    When *color_values* is provided, points are coloured by the project
    gradient and a colorbar is drawn. When ``None``, points are drawn
    in :data:`palette["blue"]` to match the selection plots.

    :param result: Output of :func:`sads.kpca.fit_kpca`.
    :param color_values: Per-point scalar to map onto colour
        (e.g. formation energy). ``None`` produces a solid-colour
        scatter in :data:`palette["blue"]`.
    :param color_label: Colorbar label. Ignored when *color_values* is
        ``None``.
    :param save: Save figure to this path. ``None`` skips saving.
    :param show: Call :func:`matplotlib.pyplot.show` after plotting.
    :returns: ``(fig, ax)``.
    """
    set_mpl_style()
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    ev = result.explained_variance

    if color_values is None:
        ax.scatter(
            result.projections[:, 0],
            result.projections[:, 1],
            c=palette["blue"],
            s=60, alpha=0.7, edgecolors="none",
        )
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

    Same colour rule as :func:`plot_kpca`: solid
    :data:`palette["blue"]` when ``color_values`` is ``None``, project
    gradient otherwise.

    :param result: Output of :func:`sads.kpca.fit_kpca`. Must have at
        least three components.
    :param color_values: Per-point scalar to map onto colour
        (e.g. formation energy). ``None`` produces a solid-colour
        scatter in :data:`palette["blue"]`.
    :param color_label: Colorbar label. Ignored when *color_values* is
        ``None``.
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