"""RMSD histogram of the conformer ensemble.

Reads the committed RMSD table ``input/rmsd_input.csv`` (one row per
conformer, columns ``conformer`` and ``rmsd`` in angstrom) and plots the
RMSD distribution as a fixed-width histogram (bin size 0.20 angstrom),
following the project palette and axis conventions.

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
)

# config.py lives in the example root, one level up from analysis/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg  # noqa: E402

RMSD_LABEL: str = r"RMSD ($\mathrm{\AA}$)"
BIN_WIDTH: float = 0.20


def _plot_histogram(
    values: np.ndarray, bin_edges: np.ndarray, out_path: Path, color: str
) -> None:
    """Plot the RMSD histogram and save it as PNG and PDF.

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


def main() -> None:
    """Load the RMSD table and plot the ensemble histogram."""
    csv_path = cfg.USE_CASE_DIR / "input" / "rmsd_input.csv"
    if not csv_path.exists():
        sys.exit(f"RMSD table not found: {csv_path}.")

    df = pd.read_csv(csv_path)
    values = df["rmsd"].to_numpy(dtype=float)

    set_mpl_style(base_fontsize=12)
    # Fixed 0.20 A bins from 0 up to the first multiple past the maximum.
    upper = np.ceil(values.max() / BIN_WIDTH) * BIN_WIDTH
    bin_edges = np.arange(0.0, upper + BIN_WIDTH, BIN_WIDTH)

    out_path = cfg.OUTPUT_ROOT / "rmsd_histogram.png"
    _plot_histogram(values, bin_edges, out_path, palette["dark blue"])
    print(f"  {len(values):4d} values -> {out_path}")


if __name__ == "__main__":
    main()
