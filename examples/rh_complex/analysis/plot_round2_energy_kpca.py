"""Round-2 selections coloured by their DFT energies on the round-2 kPCA.

Round-2 companion to :mod:`plot_energy_kpca`. After the energy-supervised
CKA re-optimisation, the three selection strategies (``select`` /
``zoom_select`` / ``nearest_select``) each pick 19 conformers; those were
DFT-relaxed and their energies recorded in ``config.ROUND2_ENERGIES_CSV``.
This overlays each strategy's picks on the round-2 kPCA, coloured by ΔE
referenced to the *study-wide* found minimum (round 1 ∪ round 2), so the
round-1 and round-2 energy plots share a single zero.

Builds the round-2 kPCA from the cached round-2 kernel, reads each
``selected_*.csv`` plus ``config.ENERGIES_CSV`` / ``ROUND2_ENERGIES_CSV``,
and writes ``<strategy>_energy.png`` / ``<strategy>_3d_energy.png`` to the
round-2 ``analysis/`` subfolder. Run with::

    python analysis/plot_round2_energy_kpca.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

from tastet.io import load_atoms_and_meta
from tastet.kpca import fit_kpca
from tastet.plotting.style import (
    apply_axis_style,
    cmap,
    palette,
    savefig,
    set_mpl_style,
)

# config.py lives in the example root; _common.py lives in round2/.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "round2"))
import config as cfg  # noqa: E402
from tastet.io import load_kernel  # noqa: E402
from _common import activate_round2  # noqa: E402

HARTREE_TO_KCAL: float = 627.509474
ENERGY_LABEL: str = "round 2  ΔE (kcal/mol)"

# Each strategy's selection subfolder and CSV filename.
STRATEGIES: dict[str, tuple[str, str]] = {
    "select": ("selection_round2", "selected_round2.csv"),
    "zoom": ("selection_zoom", "selected_zoom.csv"),
    "nearest": ("selection_nearest", "selected_nearest.csv"),
}


def _dE_by_cid() -> dict[int, float]:
    """ΔE (kcal/mol) per conformer, referenced to the study-wide minimum.

    Pools the round-1 and round-2 DFT energies and subtracts the single
    lowest energy across both, so every energy plot in the example shares
    one zero (the found global minimum, ``conformer_780``).

    :returns: Mapping ``configuration_id -> ΔE`` for all relaxed
        conformers (round 1 and round 2).
    :raises SystemExit: If either energy CSV is missing.
    """
    for path in (cfg.ENERGIES_CSV, cfg.ROUND2_ENERGIES_CSV):
        if not path.exists():
            sys.exit(f"Missing energies CSV: {path}")

    e_all = pd.concat(
        [pd.read_csv(cfg.ENERGIES_CSV), pd.read_csv(cfg.ROUND2_ENERGIES_CSV)],
        ignore_index=True,
    )
    e_all["configuration_id"] = e_all["file"].apply(
        lambda s: int(str(s).split("_")[-1])
    )
    e = e_all[cfg.ENERGY_COL].to_numpy(dtype=float)
    dE = (e - e.min()) * HARTREE_TO_KCAL
    return dict(zip(e_all["configuration_id"].astype(int), dE))


def _round2_projections() -> tuple[pd.DataFrame, list[float]]:
    """Build the round-2 kPCA from its cached kernel.

    :returns: ``(proj, ev)`` — a frame with ``configuration_id`` and
        ``kpc1..3`` for all conformers, and explained-variance
        percentages.
    :raises SystemExit: If the round-2 kernel is missing.
    """
    activate_round2()
    if not cfg.kernel_path().exists():
        sys.exit(
            f"Missing round-2 kernel: {cfg.kernel_path()}.  "
            "Run 'round2/reselect.py select' first."
        )
    _, meta = load_atoms_and_meta(cfg.db_path())
    result = fit_kpca(load_kernel(cfg.kernel_path()), n_components=3)
    p = result.projections
    proj = pd.DataFrame(
        {
            "configuration_id": meta["configuration_id"].to_numpy(),
            "kpc1": p[:, 0],
            "kpc2": p[:, 1],
            "kpc3": p[:, 2],
        }
    )
    return proj, list(result.explained_variance * 100)


def _picks_dE(proj, dE_map, sel_ids) -> tuple[np.ndarray, np.ndarray]:
    """ΔE per projection row and a mask of the strategy's picks.

    :param proj: Round-2 projections frame.
    :param dE_map: ``configuration_id -> ΔE`` mapping.
    :param sel_ids: Selected ``configuration_id`` values for the strategy.
    :returns: ``(dE, mask)`` — ΔE per row (``NaN`` off the picks) and a
        boolean mask of the picked rows.
    """
    sel = set(int(c) for c in sel_ids)
    cid = proj["configuration_id"].to_numpy()
    dE = np.array(
        [dE_map.get(int(c), np.nan) if int(c) in sel else np.nan for c in cid]
    )
    return dE, np.isin(cid, list(sel))


def _plot_2d(proj, ev, dE, mask, label: str, out_path: Path) -> None:
    """All conformers grey; the strategy's picks coloured by ΔE (2-D)."""
    set_mpl_style()
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.scatter(
        proj["kpc1"],
        proj["kpc2"],
        c="lightgrey",
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
        label=f"{label} ({int(mask.sum())})",
    )
    fig.colorbar(sc, ax=ax).set_label(ENERGY_LABEL)
    ax.set_xlabel(rf"kPC#1 ({ev[0]:.1f}%)")
    ax.set_ylabel(rf"kPC#2 ({ev[1]:.1f}%)")
    apply_axis_style(ax)
    ax.legend(frameon=False, loc="best")
    savefig(fig, out_path, dpi=300)
    plt.close(fig)


def _plot_3d(proj, ev, dE, mask, label: str, out_path: Path) -> None:
    """All conformers grey; the strategy's picks coloured by ΔE (3-D)."""
    set_mpl_style()
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        proj["kpc1"],
        proj["kpc2"],
        proj["kpc3"],
        c="lightgrey",
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
        label=f"{label} ({int(mask.sum())})",
    )
    fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.1).set_label(ENERGY_LABEL)
    ax.set_xlabel(rf"kPC#1 ({ev[0]:.1f}%)")
    ax.set_ylabel(rf"kPC#2 ({ev[1]:.1f}%)")
    ax.set_zlabel(rf"kPC#3 ({ev[2]:.1f}%)")
    ax.legend(frameon=False, loc="upper left")
    savefig(fig, out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    """Render energy-coloured round-2 kPCA plots for each strategy."""
    proj, ev = _round2_projections()
    dE_map = _dE_by_cid()
    out_dir = cfg.kernel_dir() / "analysis"

    for label, (subdir, csv_name) in STRATEGIES.items():
        sel_csv = cfg.kernel_dir() / subdir / csv_name
        if not sel_csv.exists():
            print(f"  Skipping {label}: missing {sel_csv}")
            continue
        sel_ids = pd.read_csv(sel_csv)["configuration_id"]
        dE, mask = _picks_dE(proj, dE_map, sel_ids)
        if np.isnan(dE[mask]).any():
            missing = sorted(
                int(c)
                for c, m in zip(proj["configuration_id"], mask)
                if m and int(c) not in dE_map
            )
            print(f"  Warning: {label} picks lack energies: {missing}")

        _plot_2d(proj, ev, dE, mask, label, out_dir / f"{label}_energy.png")
        _plot_3d(proj, ev, dE, mask, label, out_dir / f"{label}_3d_energy.png")
        print(f"  {label}: {int(mask.sum())} picks -> {out_dir}/{label}_*energy.png")


if __name__ == "__main__":
    main()
