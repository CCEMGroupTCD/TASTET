"""kPCA scatter-plot helper.

Moved verbatim from the former ``sads/plotting.py`` single-file module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from sads.plotting.style import set_mpl_style, apply_axis_style, savefig, cmap


def plot_kpca(
    result: "sads.kpca.KPCAResult",
    *,
    color_values: np.ndarray | None = None,
    color_label: str = "",
    ax: Axes | None = None,
    save: Path | str | None = None,
    show: bool = False,
    **scatter_kw,
) -> tuple[Figure, Axes]:
    """Scatter plot of the first two kPCA components.

    Parameters
    ----------
    result : KPCAResult
        Output of :func:`sads.kpca.fit_kpca`.
    color_values : ndarray, optional
        Per-point scalar to map onto color (e.g. formation energy).
    color_label : str
        Colorbar label.
    ax : Axes, optional
        Existing axes to plot on.  A new figure is created when *None*.
    save : path-like, optional
        Save figure to this path.
    show : bool
        Call ``plt.show()`` after plotting.
    **scatter_kw
        Extra keyword arguments forwarded to ``ax.scatter``.

    Returns
    -------
    (Figure, Axes)
    """
    from sads.kpca import KPCAResult  # deferred to avoid circular import

    set_mpl_style()

    defaults = dict(s=60, alpha=0.7, edgecolors="none", cmap=cmap)
    defaults.update(scatter_kw)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    ev = result.explained_variance
    scatter = ax.scatter(
        result.projections[:, 0],
        result.projections[:, 1],
        c=color_values,
        **defaults,
    )

    if color_values is not None:
        cbar = fig.colorbar(scatter, ax=ax)
        if color_label:
            cbar.set_label(color_label)

    ax.set_xlabel(rf"kPC#1 ({ev[0] * 100:.1f}%)")
    ax.set_ylabel(rf"kPC#2 ({ev[1] * 100:.1f}%)")
    apply_axis_style(ax)

    if save is not None:
        savefig(fig, Path(save))
    if show:
        plt.show()

    return fig, ax