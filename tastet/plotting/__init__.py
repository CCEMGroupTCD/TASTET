"""Plotting utilities for TASTET analyses."""

from tastet.plotting.kpca import plot_kpca, plot_kpca_3d
from tastet.plotting.distance import (
    plot_distance_histogram,
    plot_distance_histogram_kde,
    plot_grid_histograms,
)
from tastet.plotting.heatmap import plot_grid_heatmaps, infer_heatmap_layout
from tastet.plotting.style import set_mpl_style, apply_axis_style, savefig, cmap, palette

__all__ = [
    "plot_kpca",
    "plot_kpca_3d",
    "plot_distance_histogram",
    "plot_distance_histogram_kde",
    "plot_grid_histograms",
    "plot_grid_heatmaps",
    "infer_heatmap_layout",
    "set_mpl_style",
    "apply_axis_style",
    "savefig",
    "cmap",
    "palette",
]