"""Full-view kPCA of the picks, with the surrogate and DFT minima marked.

Reproduces ``selection_full.png`` — the full kPCA point cloud colored by
the surrogate ``E - E_gm``, with the FPS picks overlaid in magenta — and
additionally highlights the two extremes of the picks:

* the **surrogate minimum**, the pick with the lowest ``energy_eV`` in
  ``selected_structures.csv`` (the surrogate energy that drove selection);
* the **DFT minimum**, the pick with the lowest ``E (eV)`` in
  ``config.ENERGIES_SELECTED_CSV`` (the DFT energy obtained afterwards).

The point is to read off how far apart the two sit in the projected
space of the *initial* structures: the kPCA is built from the selected
snapshots as sampled, not from their DFT-relaxed counterparts.  The
in-plot legend reports their kPC#1/kPC#2 separation, and the console
additionally places that separation within the distribution of all
pairwise distances among the picks.

Run with::

    python analysis/plot_selection_extremes.py
    python analysis/plot_selection_extremes.py --no-show
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from tastet.selection import plot_selection
from tastet.plotting.style import palette, savefig, styled_legend

# config.py lives in the example root, one level up from analysis/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg  # noqa: E402

ENERGY_LABEL: str = r"$E - E_{\mathrm{gm}}$ (eV)"

#: Marker color for the pick with the lowest surrogate energy.
SURROGATE_MIN_COLOR: str = palette["dark orange"]

#: Marker color for the pick with the lowest DFT energy.
DFT_MIN_COLOR: str = palette["black"]


def find_extremes(selection_csv: Path, energies_csv: Path) -> tuple[int, int]:
    """Locate the surrogate- and DFT-minimum picks by row position.

    Both CSVs list the picks in FPS selection order, so row *i* of one
    describes the same structure as row *i* of the other.

    :param selection_csv: ``selected_structures.csv`` from the ``select``
        step (column ``energy_eV``, the surrogate energies).
    :param energies_csv: Committed DFT energies of the picks (column
        ``E (eV)``).
    :returns: ``(surrogate_row, dft_row)`` — row positions of the lowest
        surrogate energy and the lowest DFT energy.
    :raises ValueError: When the two CSVs disagree in length.
    """
    selected = pd.read_csv(selection_csv)
    dft = pd.read_csv(energies_csv)
    if len(selected) != len(dft):
        raise ValueError(
            f"Row-count mismatch: {selection_csv.name} has {len(selected)} rows "
            f"but {energies_csv.name} has {len(dft)}."
        )
    return int(selected["energy_eV"].idxmin()), int(dft["E (eV)"].idxmin())


def plot_extremes(
    *,
    out_path: Path,
    show: bool = True,
) -> plt.Figure:
    """Draw the full-view kPCA with both minima highlighted.

    :param out_path: Save path for the figure (a ``.pdf`` sibling is
        written alongside it).
    :param show: Call :func:`matplotlib.pyplot.show`.
    :returns: The figure.
    :raises FileNotFoundError: When a pipeline input is missing.
    """
    for path in (cfg.kpca_csv_path(), cfg.selection_csv_path(), cfg.csv_path()):
        if not path.exists():
            raise FileNotFoundError(f"Missing pipeline output: {path}")
    if not cfg.ENERGIES_SELECTED_CSV.exists():
        raise FileNotFoundError(f"Missing DFT energies: {cfg.ENERGIES_SELECTED_CSV}")

    proj_df = pd.read_csv(cfg.kpca_csv_path())
    selected = pd.read_csv(cfg.selection_csv_path())
    meta = pd.read_csv(cfg.csv_path())
    ev_pct = json.loads(cfg.kpca_meta_path().read_text())["explained_variance_pct"]

    # Same color scale as run.py: surrogate energies shifted to the
    # global minimum of the whole set.
    energies = meta["energy_eV"].to_numpy(dtype=float)
    color_values = energies - energies.min()

    # configuration_id is 1-based and gap-free; row position = id - 1.
    selected_indices = selected["configuration_id"].to_numpy(dtype=int) - 1

    surrogate_row, dft_row = find_extremes(
        cfg.selection_csv_path(), cfg.ENERGIES_SELECTED_CSV
    )
    surrogate_idx = int(selected_indices[surrogate_row])
    dft_idx = int(selected_indices[dft_row])

    fig, ax = plot_selection(
        proj_df,
        idx_pool=np.arange(len(proj_df)),
        selected_indices=selected_indices,
        explained_variance_pct=ev_pct,
        color_values=color_values,
        color_label=ENERGY_LABEL,
        save_path=None,
        show=False,
    )

    kpc1 = proj_df["kpc1"].to_numpy(dtype=float)
    kpc2 = proj_df["kpc2"].to_numpy(dtype=float)
    separation = float(
        np.hypot(
            kpc1[surrogate_idx] - kpc1[dft_idx], kpc2[surrogate_idx] - kpc2[dft_idx]
        )
    )

    # Connector first, so the two markers sit on top of it.
    ax.plot(
        [kpc1[surrogate_idx], kpc1[dft_idx]],
        [kpc2[surrogate_idx], kpc2[dft_idx]],
        color=palette["black"],
        lw=1.0,
        ls="--",
        zorder=6,
    )

    handles = []
    for idx, color, label in (
        (
            surrogate_idx,
            SURROGATE_MIN_COLOR,
            f"Surrogate min. (id {idx_to_id(surrogate_idx)})",
        ),
        (dft_idx, DFT_MIN_COLOR, f"DFT min. (id {idx_to_id(dft_idx)})"),
    ):
        ax.scatter(
            kpc1[idx],
            kpc2[idx],
            marker="*",
            c=color,
            s=320,
            edgecolors="white",
            linewidths=0.8,
            zorder=7,
        )
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="*",
                markersize=14,
                markerfacecolor=color,
                markeredgecolor="white",
                color=color,
                label=label,
            )
        )

    handles.append(
        Line2D(
            [],
            [],
            linestyle="none",
            marker="",
        )
    )
    styled_legend(ax, handles=handles, loc="upper left", fontsize=8)

    savefig(fig, out_path, also_pdf=True)
    print(f"Saved -> {out_path}")

    _report(proj_df, selected_indices, surrogate_idx, dft_idx, separation)

    if show:
        plt.show()
    return fig


def idx_to_id(index: int) -> int:
    """Convert a 0-based row position to a 1-based ``configuration_id``.

    :param index: Row position in the projections frame.
    :returns: The matching ``configuration_id``.
    """
    return index + 1


def _report(
    proj_df: pd.DataFrame,
    selected_indices: np.ndarray,
    surrogate_idx: int,
    dft_idx: int,
    separation: float,
) -> None:
    """Print where the two minima sit relative to the other picks.

    :param proj_df: Full projections frame.
    :param selected_indices: Row positions of the picks.
    :param surrogate_idx: Row position of the surrogate minimum.
    :param dft_idx: Row position of the DFT minimum.
    :param separation: kPC#1/kPC#2 distance between the two minima.
    """
    pts = proj_df.loc[selected_indices, ["kpc1", "kpc2"]].to_numpy(dtype=float)
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.hypot(diff[:, :, 0], diff[:, :, 1])
    pairwise = dists[np.triu_indices(len(pts), k=1)]
    percentile = float((pairwise < separation).mean() * 100.0)

    print(f"\n{'─' * 58}")
    print(f"  Surrogate min.  configuration_id {idx_to_id(surrogate_idx)}")
    print(f"  DFT min.        configuration_id {idx_to_id(dft_idx)}")
    print(f"  2-D separation  {separation:.4f}")
    print(
        f"  Pairwise distances among the {len(pts)} picks: "
        f"min {pairwise.min():.4f}, median {np.median(pairwise):.4f}, "
        f"max {pairwise.max():.4f}"
    )
    print(f"  The two minima are closer than {100.0 - percentile:.0f}% of pick pairs")
    print(f"{'─' * 58}\n")


def main() -> None:
    """Render the annotated full-view kPCA plot."""
    parser = argparse.ArgumentParser(
        description="Full-view kPCA with the surrogate and DFT minima highlighted",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help="Output figure path (default: selection/selection_full_extremes.png)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot interactively",
    )
    args = parser.parse_args()

    out_path = args.out_path or cfg.selection_dir() / "selection_full_extremes.png"
    plot_extremes(out_path=out_path, show=not args.no_show)


if __name__ == "__main__":
    main()
