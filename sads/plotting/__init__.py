"""Plotting utilities for SADS analyses.

Migration note
--------------
This package replaces the former ``sads/plotting.py`` module and
absorbs ``sads/plot_style.py`` (now ``sads/plotting/style.py``).
Delete the two old files and drop in this ``sads/plotting/`` directory.
All existing imports (``from sads.plotting import plot_kpca``) still work.
"""

from sads.plotting.kpca import plot_kpca
from sads.plotting.style import set_mpl_style, apply_axis_style, savefig, cmap, palette
from sads.plotting.heatmap import plot_grid_heatmaps, infer_heatmap_layout

__all__ = [
    "plot_kpca",
    "plot_grid_heatmaps",
    "infer_heatmap_layout",
    "set_mpl_style",
    "apply_axis_style",
    "savefig",
    "cmap",
    "palette",
]