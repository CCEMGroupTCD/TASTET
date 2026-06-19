"""Validate the surrogate against DFT for the FPS-selected structures.

After the ``select`` step exports POSCARs and they are relaxed with DFT,
this script compares the surrogate (pre-DFT) energies against the DFT
energies of the same structures and produces four plots:

1. DFT energies only.
2. Surrogate (pre-DFT) energies only.
3. Overlay of both — a visual measure of surrogate accuracy.
4. Parity plot — surrogate vs DFT relative energies against ``y = x``.

The DFT energies are read from ``config.ENERGIES_SELECTED_CSV`` (a
committed input, one row per selected structure in selection order).
The surrogate energies are read from ``selected_structures.csv``
(produced by ``select``), whose ``energy_eV`` column is in the same FPS
order, so the two are aligned row-by-row.

Each curve is referenced to its own global minimum ``E_gm``. The DFT
plots (DFT-only and the overlay, whose common reference is the DFT
global-minimum structure) carry an ``E - E_gm^DFT`` axis; the
surrogate-only plot carries ``E - E_gm``.

Run with::

    python analysis/energy_profile.py
    python analysis/energy_profile.py --no-show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tastet.plotting.style import set_mpl_style, apply_axis_style, savefig, palette

# config.py lives in the example root, one level up from analysis/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg  # noqa: E402

DFT_ENERGY_LABEL: str = r"$E - E_{\mathrm{gm}}^{\mathrm{DFT}}$ (eV)"
SURROGATE_ENERGY_LABEL: str = r"$E - E_{\mathrm{gm}}$ (eV)"


# ── Energy loading ────────────────────────────────────────────────────


def load_dft_energies(csv_path: Path) -> np.ndarray:
    """Load DFT energies of the selected structures, in selection order.

    Raw total energies are returned — no shifting is applied so that
    callers can choose their own reference.

    :param csv_path: Path to ``energies_selected.csv`` (column ``E (eV)``).
    :returns: Array of DFT total energies, one per selected structure.
    """
    df = pd.read_csv(csv_path)
    print(f"Read {len(df)} DFT energies from {csv_path.name}")
    return df["E (eV)"].values


def load_surrogate_energies(selection_csv: Path) -> np.ndarray:
    """Load surrogate energies from the selection CSV.

    The CSV is produced by the ``select`` step and lists structures in
    FPS selection order — the same order as the exported POSCARs and the
    DFT energies CSV.  Raw values are returned (no shifting).

    :param selection_csv: Path to ``selected_structures.csv``.
    :returns: Array of surrogate energies, in selection order.
    """
    df = pd.read_csv(selection_csv)
    return df["energy_eV"].values


# ── Plotting ──────────────────────────────────────────────────────────


def plot_energy_profile(
    energies: np.ndarray,
    *,
    color: str = palette["dark orange"],
    ylabel: str = r"$E - E_{\mathrm{gm}}$ (eV)",
    out_path: Path | None = None,
    show: bool = True,
    dpi: int = 300,
) -> plt.Figure:
    """Energy profile scatter plot, shifted to its own global minimum.

    :param energies: Raw energy array.
    :param color: Marker colour.
    :param ylabel: Y-axis label.
    :param out_path: Save path. *None* = don't save.
    :param show: Call ``plt.show()``.
    :param dpi: Resolution when saving.
    :returns: The figure.
    """
    shifted = energies - energies.min()

    set_mpl_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(shifted))

    ax.scatter(
        x, shifted, color=color, s=80, zorder=3, edgecolors="white", linewidths=0.5
    )

    ax.axhline(0, color="0.6", linewidth=0.8, linestyle="--", zorder=1)

    ax.set_xticks([])
    ax.set_xlabel("Selected Structures")
    ax.set_ylabel(ylabel)

    apply_axis_style(ax, use_minor_x=False, yfmt=".1f")

    if out_path is not None:
        savefig(fig, out_path, dpi=dpi)
        print(f"Saved -> {out_path}")
    if show:
        plt.show()
    return fig


def plot_comparison(
    dft_energies: np.ndarray,
    surrogate_energies: np.ndarray,
    *,
    out_path: Path | None = None,
    show: bool = True,
    dpi: int = 300,
) -> plt.Figure:
    """Overlay DFT and surrogate energy profiles with a common reference.

    Both energy sets are shifted relative to the **same structure** —
    whichever has the lowest DFT energy.  This ensures a fair comparison:
    zero on the y-axis corresponds to the same physical structure for
    both curves.

    :param dft_energies: Raw DFT energies (not pre-shifted).
    :param surrogate_energies: Raw surrogate energies (not pre-shifted).
    :param out_path: Save path. *None* = don't save.
    :param show: Call ``plt.show()``.
    :param dpi: Resolution when saving.
    :returns: The figure.
    :raises ValueError: If the two arrays differ in length.
    """
    n = len(dft_energies)
    if len(surrogate_energies) != n:
        raise ValueError(
            f"Length mismatch: {n} DFT structures vs "
            f"{len(surrogate_energies)} surrogate structures."
        )

    # Common reference: the structure with the lowest DFT energy
    ref_idx = int(np.argmin(dft_energies))
    dft_shifted = dft_energies - dft_energies[ref_idx]
    surr_shifted = surrogate_energies - surrogate_energies[ref_idx]

    set_mpl_style()
    fig, ax = plt.subplots(figsize=(max(6, n * 0.15), 4), constrained_layout=True)

    x = np.arange(n)

    ax.scatter(
        x,
        surr_shifted,
        color=palette["dark orange"],
        s=80,
        zorder=2,
        edgecolors="white",
        linewidths=0.5,
        label="Surrogate",
    )
    ax.scatter(
        x,
        dft_shifted,
        color=palette["green"],
        s=80,
        zorder=3,
        edgecolors="white",
        linewidths=0.5,
        label="DFT",
    )

    ax.axhline(0, color="0.6", linewidth=0.8, linestyle="--", zorder=0)

    ax.set_xticks([])
    ax.set_xlabel("Selected Structures")
    # Both curves are shifted to the DFT global-minimum structure.
    ax.set_ylabel(DFT_ENERGY_LABEL)

    apply_axis_style(ax, use_minor_x=False, yfmt=".1f")

    if out_path is not None:
        savefig(fig, out_path, dpi=dpi)
        print(f"Saved -> {out_path}")
    if show:
        plt.show()
    return fig


def plot_parity(
    dft_energies: np.ndarray,
    surrogate_energies: np.ndarray,
    *,
    out_path: Path | None = None,
    show: bool = True,
    dpi: int = 300,
) -> plt.Figure:
    """Parity plot of surrogate vs DFT relative energies.

    Surrogate and DFT live on different absolute scales, so the two are
    comparable only up to one additive constant. Both are therefore
    referenced to a single common structure — the DFT global minimum —
    so each axis is ``E - E_gm^DFT`` (energy relative to the DFT ground
    state): the x-axis from the surrogate, the y-axis from DFT. Points on
    the ``y = x`` diagonal mean the surrogate reproduces the DFT relative
    energetics exactly, and the annotated MAE/RMSE (deviation from the
    diagonal) are the surrogate error relative to DFT — the same values
    reported by :func:`plot_comparison`.

    :param dft_energies: Raw DFT energies (not pre-shifted).
    :param surrogate_energies: Raw surrogate energies (not pre-shifted).
    :param out_path: Save path. *None* = don't save.
    :param show: Call ``plt.show()``.
    :param dpi: Resolution when saving.
    :returns: The figure.
    :raises ValueError: If the two arrays differ in length.
    """
    n = len(dft_energies)
    if len(surrogate_energies) != n:
        raise ValueError(
            f"Length mismatch: {n} DFT structures vs "
            f"{len(surrogate_energies)} surrogate structures."
        )

    # Common reference for both methods: the DFT global-minimum structure.
    ref_idx = int(np.argmin(dft_energies))
    x = surrogate_energies - surrogate_energies[ref_idx]
    y = dft_energies - dft_energies[ref_idx]

    residuals = y - x
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))

    set_mpl_style()
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)

    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = 0.05 * (hi - lo)
    lims = (lo - pad, hi + pad)

    # y = x reference (perfect surrogate–DFT agreement)
    ax.plot(lims, lims, color="0.6", lw=0.8, ls="--", zorder=1)
    ax.scatter(
        x,
        y,
        color=palette["blue"],
        s=80,
        zorder=3,
        edgecolors="white",
        linewidths=0.5,
    )

    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Surrogate $E - E_{\mathrm{gm}}^{\mathrm{DFT}}$ (eV)")
    ax.set_ylabel(r"DFT $E - E_{\mathrm{gm}}^{\mathrm{DFT}}$ (eV)")
    ax.text(
        0.04,
        0.96,
        f"MAE = {mae:.2f} eV\nRMSE = {rmse:.2f} eV",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
    )

    apply_axis_style(ax, xfmt=".1f", yfmt=".1f")

    if out_path is not None:
        savefig(fig, out_path, dpi=dpi)
        print(f"Saved -> {out_path}")
    if show:
        plt.show()
    return fig


# ── CLI ───────────────────────────────────────────────────────────────


def main() -> None:
    """Load DFT + surrogate energies and render the four comparison plots."""
    parser = argparse.ArgumentParser(
        description="Validate the surrogate against DFT for the selected structures",
    )
    parser.add_argument(
        "--energies-csv",
        type=Path,
        default=cfg.ENERGIES_SELECTED_CSV,
        help="DFT energies CSV (default: input/energies_selected.csv)",
    )
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=cfg.selection_csv_path(),
        help="Path to selected_structures.csv (surrogate energies)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=cfg.selection_dir(),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots interactively",
    )
    args = parser.parse_args()

    if not args.energies_csv.exists():
        raise FileNotFoundError(f"DFT energies CSV not found: {args.energies_csv}")
    if not args.selection_csv.exists():
        raise FileNotFoundError(
            f"Selection CSV not found: {args.selection_csv}.  "
            f"Run 'python run.py select' first."
        )

    show = not args.no_show
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load energies ───────────────────────────────────────────────
    dft_raw = load_dft_energies(args.energies_csv)
    surrogate_raw = load_surrogate_energies(args.selection_csv)

    print(f"\n{'─' * 50}")
    print(f"  Structures:       {len(dft_raw)}")
    print(
        f"  DFT E range:      [{(dft_raw - dft_raw.min()).min():.3f}, "
        f"{(dft_raw - dft_raw.min()).max():.3f}] eV  (rel. to gm)"
    )
    print(
        f"  Surrogate E range:[{(surrogate_raw - surrogate_raw.min()).min():.3f}, "
        f"{(surrogate_raw - surrogate_raw.min()).max():.3f}] eV  (rel. to gm)"
    )

    # Errors computed on a common reference (DFT gm structure)
    ref_idx = int(np.argmin(dft_raw))
    dft_ref = dft_raw - dft_raw[ref_idx]
    surr_ref = surrogate_raw - surrogate_raw[ref_idx]
    residuals = surr_ref - dft_ref
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    print(f"  MAE:              {mae:.3f} eV")
    print(f"  RMSE:             {rmse:.3f} eV")
    print(f"{'─' * 50}\n")

    # ── Plot 1: DFT only (shifted to own gm) ────────────────────────
    plot_energy_profile(
        dft_raw,
        color=palette["green"],
        ylabel=DFT_ENERGY_LABEL,
        out_path=out_dir / "energy_profile_dft.png",
        show=show,
    )

    # ── Plot 2: Surrogate only (shifted to own gm) ──────────────────
    plot_energy_profile(
        surrogate_raw,
        color=palette["dark orange"],
        ylabel=SURROGATE_ENERGY_LABEL,
        out_path=out_dir / "energy_profile_surrogate.png",
        show=show,
    )

    # ── Plot 3: Overlay (common reference: DFT gm structure) ────────
    plot_comparison(
        dft_raw,
        surrogate_raw,
        out_path=out_dir / "energy_profile_comparison.png",
        show=show,
    )

    # ── Plot 4: Parity (surrogate vs DFT, common DFT-gm reference) ──
    plot_parity(
        dft_raw,
        surrogate_raw,
        out_path=out_dir / "energy_profile_parity.png",
        show=show,
    )


if __name__ == "__main__":
    main()
