"""RMSD histograms of the conformer comparison sets.

Reads the committed RMSD table ``input/rmsd_input.csv`` (one row per
reference/conformer pair, RMSD in angstrom). The table holds two
independent ensembles, each compared against its own reference conformer
(the ``set`` column). This script plots the RMSD distribution as a
fixed-width histogram (bin size 0.20 angstrom) four times: once for each
ensemble on its own, once for both pooled together, and once with the two
ensembles stacked in a single figure. All share the same bin edges so they
are directly comparable, and all follow the project palette and axis
conventions.

Run with::

    python analysis/plot_rmsd_histogram.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tastet.plotting.style import (
    apply_axis_style,
    palette,
    savefig,
    set_mpl_style,
    styled_legend,
)

# config.py lives in the example root, one level up from analysis/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg  # noqa: E402

RMSD_LABEL: str = r"RMSD ($\mathrm{\AA}$)"
BIN_WIDTH: float = 0.20


def _plot_histogram(
    values: np.ndarray, bin_edges: np.ndarray, out_path: Path, color: str
) -> None:
    """Plot one RMSD histogram and save it as PNG and PDF.

    :param values: RMSD values to histogram.
    :param bin_edges: Fixed-width bin edges (``BIN_WIDTH`` angstrom apart).
    :param out_path: Save path for the ``.png`` (a ``.pdf`` sibling is also
        written).
    :param color: Bar fill colour.
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    ax.hist(
        values,
        bins=bin_edges,
        color=color,
        edgecolor=palette["black"],
        linewidth=0.8,
        alpha=0.75,
    )

    ax.set_xlabel(RMSD_LABEL)
    ax.set_ylabel("Count")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    apply_axis_style(ax, xfmt="%.1f", yfmt="%.0f")

    fig.tight_layout()
    savefig(fig, out_path, also_pdf=True)
    plt.close(fig)


def _plot_stacked(
    sdf: np.ndarray, xyz: np.ndarray, bin_edges: np.ndarray, out_path: Path
) -> None:
    """Plot both ensembles stacked in one figure and save it as PNG and PDF.

    :param sdf: RMSD values for the SDF ensemble (drawn green, lower stack).
    :param xyz: RMSD values for the XYZ ensemble (drawn orange, upper stack).
    :param bin_edges: Fixed-width bin edges (``BIN_WIDTH`` angstrom apart).
    :param out_path: Save path for the ``.png`` (a ``.pdf`` sibling is also
        written).
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    ax.hist(
        [sdf, xyz],
        bins=bin_edges,
        stacked=True,
        color=[palette["green"], palette["orange"]],
        edgecolor=palette["black"],
        linewidth=0.8,
        alpha=0.75,
        label=["SDF (vs 171)", "XYZ (vs 137)"],
    )

    ax.set_xlabel(RMSD_LABEL)
    ax.set_ylabel("Count")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    apply_axis_style(ax, xfmt="%.1f", yfmt="%.0f")
    styled_legend(ax)

    fig.tight_layout()
    savefig(fig, out_path, also_pdf=True)
    plt.close(fig)


def main() -> None:
    """Load the RMSD table and plot per-ensemble and pooled histograms."""
    csv_path = cfg.USE_CASE_DIR / "input" / "rmsd_input.csv"
    if not csv_path.exists():
        sys.exit(f"RMSD table not found: {csv_path}.")

    df = pd.read_csv(csv_path)
    all_values = df["rmsd"].to_numpy(dtype=float)

    # One histogram per ensemble (keyed by reference) plus one pooled.
    sdf = df.loc[df["reference"] == "sdf_conformer_0171", "rmsd"].to_numpy(dtype=float)
    xyz = df.loc[df["reference"] == "xyz_conformer_0137", "rmsd"].to_numpy(dtype=float)

    set_mpl_style(base_fontsize=12)
    # Fixed 0.20 A bins from 0 up to the first multiple past the global
    # maximum, shared by all three figures so they are directly comparable.
    upper = np.ceil(all_values.max() / BIN_WIDTH) * BIN_WIDTH
    bin_edges = np.arange(0.0, upper + BIN_WIDTH, BIN_WIDTH)

    figures = [
        ("rmsd_histogram_sdf.png", sdf, palette["green"]),
        ("rmsd_histogram_xyz.png", xyz, palette["orange"]),
        ("rmsd_histogram.png", all_values, palette["dark blue"]),
    ]
    for name, values, color in figures:
        out_path = cfg.OUTPUT_ROOT / name
        _plot_histogram(values, bin_edges, out_path, color)
        print(f"  {len(values):4d} values -> {out_path}")

    stacked_path = cfg.OUTPUT_ROOT / "rmsd_histogram_stacked.png"
    _plot_stacked(sdf, xyz, bin_edges, stacked_path)
    print(f"  {len(sdf) + len(xyz):4d} values -> {stacked_path}")


if __name__ == "__main__":
    main()
