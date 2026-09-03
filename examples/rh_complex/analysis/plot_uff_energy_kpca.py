"""The UFF-selected conformers colored by their DFT energies on the kPCA.

Force-field companion to :mod:`plot_energy_kpca`. That script overlays the
round-1 FPS picks on the production kPCA; this one overlays the 25 conformers
a UFF pre-filter would have chosen instead, colored by ΔE on the same zero,
so the two figures can be read side by side: same map, two different ways of
spending the DFT budget.

The colorbar *zero* is shared but its range is not — each figure autoscales to
its own subset (round 1 spans 0.0-18.8 kcal/mol, the UFF picks 6.4-27.6), so
compare positions and absolute values, not hues.

ΔE is referenced to the study-wide found minimum (the lowest absolute energy
across ``config.ENERGIES_CSV`` and ``config.ROUND2_ENERGIES_CSV``), matching
``round2/_common.study_wide_dE_by_cid`` so every energy figure in the example
shares one zero.

Reads ``kpca_projections.csv`` / ``kpca_meta.json`` from the kPCA directory
and ``config.UFF_FILTER_CSV``; writes ``kpca_uff_energy.png`` and
``kpca_3d_uff_energy.png`` to that directory's ``analysis/`` subfolder.

``--kpca-dir`` defaults to the directory ``config.py`` currently resolves to.
Pass it explicitly to plot against a kPCA produced by an earlier kernel
configuration, whose hashed directory name no longer matches the config::

    python analysis/plot_uff_energy_kpca.py
    python analysis/plot_uff_energy_kpca.py --kpca-dir output/production/product_e8cf56ec
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

from tastet.plotting.style import (
    apply_axis_style,
    cmap,
    palette,
    savefig,
    set_mpl_style,
    styled_legend,
)

# config.py lives in the example root, one level up from analysis/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg  # noqa: E402

HARTREE_TO_KCAL: float = 627.509474
ENERGY_LABEL: str = r"$E - E_\mathrm{gm}$ (kcal mol$^{-1}$)"

#: Legend label for the highlighted subset.
SUBSET_LABEL: str = "UFF-selected"


def _shared_zero_hartree() -> float:
    """Absolute energy of the study-wide found minimum, in Hartree.

    Pools the round-1 and round-2 DFT energies and returns the single lowest
    absolute energy across both. This reproduces the zero used by
    ``round2/_common.study_wide_dE_by_cid`` without importing the round-2
    machinery, so every energy figure in the example shares one reference.

    :returns: The lowest ``config.ENERGY_COL`` value across both round CSVs.
    :raises SystemExit: If either energy CSV is missing.
    """
    for path in (cfg.ENERGIES_CSV, cfg.ROUND2_ENERGIES_CSV):
        if not path.exists():
            sys.exit(f"Missing energies CSV: {path}")
    pooled = pd.concat(
        [pd.read_csv(cfg.ENERGIES_CSV), pd.read_csv(cfg.ROUND2_ENERGIES_CSV)],
        ignore_index=True,
    )
    return float(pooled[cfg.ENERGY_COL].min())


def _load_inputs(
    kpca_dir: Path,
) -> tuple[pd.DataFrame, list[float], np.ndarray, np.ndarray]:
    """Load the kPCA projections, variances, and the UFF subset's ΔE.

    :param kpca_dir: Directory holding ``kpca_projections.csv`` and
        ``kpca_meta.json``.
    :returns: ``(proj, ev, dE, mask)`` — the projections frame
        (``configuration_id``, ``kpc1..3``), explained-variance percentages,
        a per-row ΔE array (kcal/mol; ``NaN`` off the UFF-selected set), and
        a boolean mask of the UFF-selected rows.
    :raises SystemExit: If a required input file is missing, or if the CSV
        names a conformer absent from the projections.
    """
    csv_path = kpca_dir / "kpca_projections.csv"
    meta_path = kpca_dir / "kpca_meta.json"
    if not csv_path.exists():
        sys.exit(f"Missing kPCA projections: {csv_path}.  Run 'run.py kpca' first.")
    if not cfg.UFF_FILTER_CSV.exists():
        sys.exit(f"Missing UFF energies CSV: {cfg.UFF_FILTER_CSV}")

    proj = pd.read_csv(csv_path)
    ev = json.loads(meta_path.read_text())["explained_variance_pct"]

    e_df = pd.read_csv(cfg.UFF_FILTER_CSV)
    e_df["configuration_id"] = e_df["file"].apply(lambda s: int(str(s).split("_")[-1]))
    missing = set(e_df["configuration_id"]) - set(proj["configuration_id"])
    if missing:
        sys.exit(
            f"{cfg.UFF_FILTER_CSV.name} names conformers absent from "
            f"{csv_path.name}: {sorted(missing)}"
        )

    e = e_df[cfg.ENERGY_COL].to_numpy(dtype=float)
    dE_by_cid = dict(
        zip(e_df["configuration_id"], (e - _shared_zero_hartree()) * HARTREE_TO_KCAL)
    )

    dE = proj["configuration_id"].map(dE_by_cid).to_numpy(dtype=float)
    mask = ~np.isnan(dE)
    print(
        f"Colored {int(mask.sum())} UFF-selected conformers by ΔE "
        f"out of {len(proj)} total."
    )
    return proj, ev, dE, mask


def _plot_2d(proj, ev, dE, mask, out_path: Path) -> None:
    """All conformers gray; the UFF-selected ones colored by ΔE (2-D).

    :param proj: Projections frame with ``kpc1``/``kpc2`` columns.
    :param ev: Explained-variance percentages per component.
    :param dE: Per-row ΔE in kcal/mol (``NaN`` off the subset).
    :param mask: Boolean mask of the UFF-selected rows.
    :param out_path: Save path; a ``.pdf`` sibling is written too.
    """
    set_mpl_style()
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.scatter(
        proj["kpc1"],
        proj["kpc2"],
        c="lightgray",
        s=60,
        alpha=0.5,
        edgecolors="none",
        label=f"all ({len(proj)})",
    )
    sc = ax.scatter(
        proj["kpc1"][mask],
        proj["kpc2"][mask],
        c=dE[mask],
        cmap=cmap,
        s=70,
        alpha=0.95,
        edgecolors=palette["black"],
        linewidth=0.3,
        zorder=3,
        label=f"{SUBSET_LABEL} ({int(mask.sum())})",
    )
    fig.colorbar(sc, ax=ax).set_label(ENERGY_LABEL)
    ax.set_xlabel(rf"kPC#1 ({ev[0]:.1f}%)")
    ax.set_ylabel(rf"kPC#2 ({ev[1]:.1f}%)")
    apply_axis_style(ax)
    styled_legend(ax, loc="best")
    savefig(fig, out_path, dpi=300, also_pdf=True)
    plt.close(fig)


def _plot_3d(proj, ev, dE, mask, out_path: Path) -> None:
    """All conformers gray; the UFF-selected ones colored by ΔE (3-D).

    :param proj: Projections frame with ``kpc1``/``kpc2``/``kpc3`` columns.
    :param ev: Explained-variance percentages per component.
    :param dE: Per-row ΔE in kcal/mol (``NaN`` off the subset).
    :param mask: Boolean mask of the UFF-selected rows.
    :param out_path: Save path; a ``.pdf`` sibling is written too.
    """
    set_mpl_style()
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        proj["kpc1"],
        proj["kpc2"],
        proj["kpc3"],
        c="lightgray",
        s=15,
        alpha=0.15,
        edgecolors="none",
        depthshade=False,
        label=f"all ({len(proj)})",
    )
    sc = ax.scatter(
        proj["kpc1"][mask],
        proj["kpc2"][mask],
        proj["kpc3"][mask],
        c=dE[mask],
        cmap=cmap,
        s=80,
        alpha=0.95,
        edgecolors=palette["black"],
        linewidth=0.3,
        depthshade=False,
        label=f"{SUBSET_LABEL} ({int(mask.sum())})",
    )
    fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.1).set_label(ENERGY_LABEL)
    ax.set_xlabel(rf"kPC#1 ({ev[0]:.1f}%)")
    ax.set_ylabel(rf"kPC#2 ({ev[1]:.1f}%)")
    ax.set_zlabel(rf"kPC#3 ({ev[2]:.1f}%)")
    styled_legend(ax, loc="upper left")
    savefig(fig, out_path, dpi=300, also_pdf=True)
    plt.close(fig)


def main() -> None:
    """Render the UFF-selected energy-colored kPCA (2-D + 3-D)."""
    parser = argparse.ArgumentParser(
        description="Overlay the UFF-selected conformers on the kPCA, colored by ΔE",
    )
    parser.add_argument(
        "--kpca-dir",
        type=Path,
        default=cfg.kernel_dir(),
        help="Directory holding kpca_projections.csv / kpca_meta.json",
    )
    args = parser.parse_args()

    proj, ev, dE, mask = _load_inputs(args.kpca_dir)
    out_dir = args.kpca_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_2d(proj, ev, dE, mask, out_dir / "kpca_uff_energy.png")
    print(f"  2-D plot -> {out_dir / 'kpca_uff_energy.png'}")
    _plot_3d(proj, ev, dE, mask, out_dir / "kpca_3d_uff_energy.png")
    print(f"  3-D plot -> {out_dir / 'kpca_3d_uff_energy.png'}")


if __name__ == "__main__":
    main()
