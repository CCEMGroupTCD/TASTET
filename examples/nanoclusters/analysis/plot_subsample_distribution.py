"""Energy histograms of the full set vs the energy-balanced subsample.

The grid search runs on an energy-balanced subset drawn by
``prepare.subsample`` (inverse-density sampling that over-represents rare
energy regions).  This script visualizes the effect: it loads the master
database, reproduces the subset deterministically via
``prepare._subsample_indices_energy`` (same ``SEED`` / ``NUM_BINS`` /
``GRID_SEARCH_N_SAMPLES`` as the pipeline), and plots ``E - E_gm`` histograms for
the full set and the subset on shared bin edges.

Run with::

    python analysis/plot_subsample_distribution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from tastet.io import load_atoms_and_meta
from tastet.plotting.style import apply_axis_style, palette, savefig, set_mpl_style

# config.py / prepare.py live in the example root, one level up from analysis/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg  # noqa: E402
import prepare  # noqa: E402

RELATIVE_ENERGY_LABEL: str = r"$E - E_{\mathrm{gm}}$ (eV)"


def _plot_distribution(
    values: np.ndarray,
    bin_edges: np.ndarray,
    out_path: Path,
    color: str,
) -> None:
    """Plot a single ``E - E_gm`` histogram and save it.

    :param values: Relative energies to histogram.
    :param bin_edges: Shared bin edges (so figures are comparable).
    :param out_path: Save path for the figure.
    :param color: Bar fill color.
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

    ax.set_xlabel(RELATIVE_ENERGY_LABEL)
    ax.set_ylabel("Count")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    apply_axis_style(ax, xfmt="%.0f", yfmt="%.0f")

    fig.tight_layout()
    savefig(fig, out_path)
    plt.close(fig)


def main() -> None:
    """Load the production DB, reproduce the subset, and plot both histograms."""
    db_path = cfg.db_path()
    if not db_path.exists():
        sys.exit(f"Database not found: {db_path}.  Run 'python run.py db' first.")

    _, meta = load_atoms_and_meta(db_path)
    energy = meta["energy_eV"].values
    relative_full = energy - energy.min()

    idx = prepare._subsample_indices_energy(
        energy, cfg.GRID_SEARCH_N_SAMPLES, cfg.SEED, cfg.NUM_BINS
    )
    relative_sub = relative_full[idx]

    print(f"Full set:  {len(relative_full)} structures")
    print(
        f"Subsample: {len(relative_sub)} structures (seed={cfg.SEED}, "
        f"bins={cfg.NUM_BINS})"
    )

    set_mpl_style(base_fontsize=12)
    # Shared bin edges so the two figures are directly comparable.
    _, bin_edges = np.histogram(relative_full, bins=cfg.NUM_BINS)

    out_dir = db_path.parent
    full_path = out_dir / "sampling_distribution_full.png"
    sampled_path = out_dir / "sampling_distribution_sampled.png"

    _plot_distribution(relative_full, bin_edges, full_path, palette["dark blue"])
    _plot_distribution(relative_sub, bin_edges, sampled_path, palette["green"])

    print(f"  Full distribution    -> {full_path}")
    print(f"  Sampled distribution -> {sampled_path}")


if __name__ == "__main__":
    main()
