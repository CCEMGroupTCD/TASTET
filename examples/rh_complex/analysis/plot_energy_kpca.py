"""Middle step (round 1.5) — production kPCA colored by round-1 DFT energies.

Takes the cached production kPCA projections (the unsupervised round-1
representation) and overlays the round-1 DFT energies on the conformers
that were selected and relaxed. This is the visual bridge between
round 1 (unsupervised selection) and round 2 (energy-supervised CKA
re-optimization): it shows how energy is distributed across the round-1
representation before that signal is used to re-optimize the kernel.

Reads ``kpca_projections.csv`` / ``kpca_meta.json`` from the active
(production) analysis and ``config.ENERGIES_CSV``; writes
``kpca_energy.png`` and ``kpca_3d_energy.png`` to the kPCA ``analysis/``
subfolder. Run with::

    python analysis/plot_energy_kpca.py
"""

from __future__ import annotations

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


def _load_inputs() -> tuple[pd.DataFrame, list[float], np.ndarray, np.ndarray]:
    """Load the production kPCA projections, variances, and round-1 ΔE.

    :returns: ``(proj, ev, dE, round1_mask)`` — the projections frame
        (``configuration_id``, ``kpc1..3``), explained-variance
        percentages, a per-row ΔE array (kcal/mol; ``NaN`` off the
        round-1 set), and a boolean mask of the round-1 rows.
    :raises SystemExit: If a required input file is missing.
    """
    csv_path = cfg.kpca_csv_path()
    meta_path = cfg.kpca_meta_path()
    if not csv_path.exists():
        sys.exit(f"Missing kPCA projections: {csv_path}.  Run 'run.py kpca' first.")
    if not cfg.ENERGIES_CSV.exists():
        sys.exit(f"Missing round-1 energies CSV: {cfg.ENERGIES_CSV}")

    proj = pd.read_csv(csv_path)
    ev = json.loads(meta_path.read_text())["explained_variance_pct"]

    e_df = pd.read_csv(cfg.ENERGIES_CSV)
    e_df["configuration_id"] = e_df["file"].apply(lambda s: int(str(s).split("_")[-1]))
    e = e_df[cfg.ENERGY_COL].to_numpy(dtype=float)
    dE_by_cid = dict(zip(e_df["configuration_id"], (e - e.min()) * HARTREE_TO_KCAL))

    dE = proj["configuration_id"].map(dE_by_cid).to_numpy(dtype=float)
    round1_mask = ~np.isnan(dE)
    print(
        f"Colored {int(round1_mask.sum())} round-1 conformers by ΔE "
        f"out of {len(proj)} total."
    )
    return proj, ev, dE, round1_mask


def _plot_2d(proj, ev, dE, mask, out_path: Path) -> None:
    """All conformers gray; round-1 conformers colored by ΔE (2-D)."""
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
        label=f"round 1 ({int(mask.sum())})",
    )
    fig.colorbar(sc, ax=ax).set_label(ENERGY_LABEL)
    ax.set_xlabel(rf"kPC#1 ({ev[0]:.1f}%)")
    ax.set_ylabel(rf"kPC#2 ({ev[1]:.1f}%)")
    apply_axis_style(ax)
    styled_legend(ax, loc="best")
    savefig(fig, out_path, dpi=300, also_pdf=True)
    plt.close(fig)


def _plot_3d(proj, ev, dE, mask, out_path: Path) -> None:
    """All conformers gray; round-1 conformers colored by ΔE (3-D)."""
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
        label=f"round 1 ({int(mask.sum())})",
    )
    fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.1).set_label(ENERGY_LABEL)
    ax.set_xlabel(rf"kPC#1 ({ev[0]:.1f}%)")
    ax.set_ylabel(rf"kPC#2 ({ev[1]:.1f}%)")
    ax.set_zlabel(rf"kPC#3 ({ev[2]:.1f}%)")
    styled_legend(ax, loc="upper left")
    savefig(fig, out_path, dpi=300, also_pdf=True)
    plt.close(fig)


def main() -> None:
    """Render the energy-colored production kPCA (2-D + 3-D)."""
    proj, ev, dE, mask = _load_inputs()
    out_dir = cfg.kpca_analysis_dir()
    _plot_2d(proj, ev, dE, mask, out_dir / "kpca_energy.png")
    print(f"  2-D plot -> {out_dir / 'kpca_energy.png'}")
    _plot_3d(proj, ev, dE, mask, out_dir / "kpca_3d_energy.png")
    print(f"  3-D plot -> {out_dir / 'kpca_3d_energy.png'}")


if __name__ == "__main__":
    main()
