"""Baseline: what a UFF pre-filter would have given, judged against DFT.

The obvious alternative to the SOAP-kernel-FPS selection is to skip the
representation entirely and rank conformers with the force field the
conformer generator already ships -- OpenBabel's UFF -- then spend the DFT
budget on the lowest-UFF-energy ones. This script tests that baseline: the
25 conformers with the lowest UFF energy were relaxed with DFT, and their
DFT energies are compared against the UFF energies that chose them.

Produces one figure, the parity plot of ``uff_parity.png``, in the same
convention as the nanoclusters surrogate validation
(``examples/nanoclusters/analysis/energy_profile.py``): the reference
method on x, the cheap method being judged on y, each axis re-referenced
to its own minimum so the two absolute scales become comparable.

Reads ``config.UFF_FILTER_CSV``; writes to ``uff_filter/`` inside the
active analysis directory. The plot depends on no SOAP or kernel choice,
so it sits beside the kernel-hashed directories rather than inside one.

Run with::

    python analysis/plot_uff_parity.py
    python analysis/plot_uff_parity.py --no-show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from tastet.plotting.style import (
    set_mpl_style,
    apply_axis_style,
    match_major_ticks,
    savefig,
    styled_legend,
    palette,
)

# config.py lives in the example root, one level up from analysis/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg  # noqa: E402

#: Parity axes. Each method is referenced to its own minimum of the filtered
#: set, so ``E_gm`` on the UFF axis is the lowest *UFF* energy — in general a
#: different conformer from the DFT minimum.
DFT_PARITY_LABEL: str = (
    r"DFT, $E^{\mathrm{DFT}} - E_{\mathrm{gm}}^{\mathrm{DFT}}$ (kcal mol$^{-1}$)"
)
UFF_PARITY_LABEL: str = r"UFF, $E - E_{\mathrm{gm}}$ (kcal mol$^{-1}$)"

#: Subdirectory of the active analysis holding this baseline's output.
OUT_DIR_NAME: str = "uff_filter"

#: Hartree to kcal/mol, matching ``analysis/plot_energy_kpca.py``.
HARTREE_TO_KCAL: float = 627.509474


# ── Energy loading ────────────────────────────────────────────────────


def load_energies(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the DFT and UFF energies of the UFF-filtered conformers.

    Both energies live on the same row of the same file, so no join is
    needed and the two cannot be mispaired. Energies are returned as
    stored — no shifting is applied, so that callers choose their own
    reference.

    DFT energies are stored absolute, in Hartree, and converted to kcal/mol
    here so both methods share the unit the rest of this use case reports in.

    :param csv_path: Path to ``energies_uff_filter.csv`` (columns ``file``,
        :data:`config.UFF_ENERGY_COL` and :data:`config.ENERGY_COL`).
    :returns: Tuple of ``(dft, uff, conformer_ids)``, in the row order of the
        file (ascending UFF energy — the order the filter would have ranked
        them).
    :raises ValueError: When a required column is missing.
    """
    frame = pd.read_csv(csv_path)
    required = {"file", cfg.UFF_ENERGY_COL, cfg.ENERGY_COL}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns {sorted(missing)}.")

    print(f"Read {len(frame)} UFF-filtered conformers from {csv_path.name}")
    conformer_ids = frame["file"].str.rsplit("_", n=1).str[-1].astype(int)
    return (
        frame[cfg.ENERGY_COL].to_numpy(dtype=float) * HARTREE_TO_KCAL,
        frame[cfg.UFF_ENERGY_COL].to_numpy(dtype=float),
        conformer_ids.to_numpy(dtype=int),
    )


# ── Plotting ──────────────────────────────────────────────────────────


def plot_parity(
    dft_energies: np.ndarray,
    uff_energies: np.ndarray,
    *,
    out_path: Path | None = None,
    show: bool = True,
    dpi: int = 300,
) -> plt.Figure:
    r"""Parity plot of UFF vs DFT relative energies.

    DFT is the reference and goes on the x-axis; UFF is the quantity being
    judged and goes on the y-axis. Points on the diagonal mean UFF
    reproduces the DFT relative energetics exactly.

    UFF and DFT live on different absolute scales — a force field's zero is
    its own — so each axis is referenced to its own minimum of the filtered
    set:

    .. math::

        x_i = E_i^{\mathrm{DFT}} - \min_j E_j^{\mathrm{DFT}}, \qquad
        y_i = E_i^{\mathrm{UFF}} - \min_j E_j^{\mathrm{UFF}}.

    Both axes then read "energy above the method's own best conformer" — the
    quantity each method would act on — and no single conformer's error is
    built into the gauge (referencing both methods to one conformer would
    add that conformer's UFF error to every residual). Because both axes
    subtract a quantity measured on the same scale as the one they plot,
    whatever reference the incoming DFT column was recorded against cancels.

    The annotated statistics pair a ranking measure with a magnitude
    measure. The Spearman :math:`\rho` rank correlation asks whether UFF
    *orders* the conformers as DFT does — what a pre-filter rests on — and
    is invariant to the reference choice. The MAE is that of the plotted
    points about the diagonal: the mean vertical deviation in the figure's
    own gauge, i.e. the error in energies above each method's own minimum.
    The CLI additionally reports Kendall :math:`\tau`, RMSE, and the
    mean-aligned error variants.

    :param dft_energies: DFT energies, in kcal/mol (not pre-shifted).
    :param uff_energies: UFF energies, in kcal/mol (not pre-shifted).
    :param out_path: Save path. *None* = don't save.
    :param show: Call ``plt.show()``.
    :param dpi: Resolution when saving.
    :returns: The figure.
    :raises ValueError: If the two arrays differ in length.
    """
    n = len(dft_energies)
    if len(uff_energies) != n:
        raise ValueError(
            f"Length mismatch: {n} DFT energies vs {len(uff_energies)} UFF energies."
        )

    # Each method referenced to its own minimum of the set.
    x = dft_energies - dft_energies.min()
    y = uff_energies - uff_energies.min()

    rho = float(pd.Series(y).corr(pd.Series(x), method="spearman"))
    # MAE of the plotted points about the diagonal: the mean vertical
    # deviation in the gauge the axes themselves define.
    mae = float(np.mean(np.abs(y - x)))

    set_mpl_style()
    fig, ax = plt.subplots(figsize=(max(6, n * 0.15), 4), constrained_layout=True)

    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = 0.05 * (hi - lo)
    lims = (lo - pad, hi + pad)

    # y = x reference (perfect UFF–DFT agreement)
    ax.plot(lims, lims, color="0.6", lw=0.8, ls="--", zorder=1)
    ax.scatter(
        x,
        y,
        color=palette["green"],
        s=80,
        zorder=3,
        edgecolors="white",
        linewidths=0.5,
    )

    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect("equal", adjustable="box")
    # Same step on both axes: the two carry the same quantity.
    match_major_ticks(ax)
    ax.set_xlabel(DFT_PARITY_LABEL)
    ax.set_ylabel(UFF_PARITY_LABEL)

    # Text-only legend: one ranking statistic, one magnitude statistic
    # (handles hidden). Lower right: with each axis anchored at its own
    # minimum the cloud gathers in the upper-left half, leaving that
    # corner clear. One decimal on the MAE: the incoming DFT column is
    # recorded to 0.1 kcal/mol.
    stats = [
        Line2D(
            [], [], linestyle="none", marker="", label=rf"Spearman $\rho$ = {rho:.2f}"
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="",
            label=rf"MAE = {mae:.1f} kcal mol$^{{-1}}$",
        ),
    ]
    styled_legend(ax, handles=stats, loc="lower right", handlelength=0, handletextpad=0)

    apply_axis_style(ax, xfmt=".1f", yfmt=".1f")

    if out_path is not None:
        savefig(fig, out_path, dpi=dpi, also_pdf=True)
        print(f"Saved -> {out_path}")
    if show:
        plt.show()
    return fig


# ── CLI ───────────────────────────────────────────────────────────────


def main() -> None:
    """Load the UFF-filter energies and render the parity plot."""
    parser = argparse.ArgumentParser(
        description="Judge a UFF pre-filter of the conformers against DFT",
    )
    parser.add_argument(
        "--energies-csv",
        type=Path,
        default=cfg.UFF_FILTER_CSV,
        help="UFF/DFT energies CSV (default: input/energies_uff_filter.csv)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=cfg.analysis_dir() / OUT_DIR_NAME,
        help="Output directory for the plot",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot interactively",
    )
    args = parser.parse_args()

    if not args.energies_csv.exists():
        raise FileNotFoundError(f"UFF/DFT energies CSV not found: {args.energies_csv}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dft_raw, uff_raw, conformer_ids = load_energies(args.energies_csv)

    ref_idx = int(np.argmin(dft_raw))
    uff_idx = int(np.argmin(uff_raw))
    dft_rank = int(np.sum(dft_raw < dft_raw[uff_idx])) + 1

    # Rank correlations (reference-independent), as annotated on the figure.
    rho = float(pd.Series(uff_raw).corr(pd.Series(dft_raw), method="spearman"))
    tau = float(pd.Series(uff_raw).corr(pd.Series(dft_raw), method="kendall"))

    # Quantitative errors depend on how the UFF-DFT offset is fixed:
    # min-aligned matches the figure's axes; mean-aligned (the least-squares
    # constant) measures pure scatter with no conformer privileged.
    residuals = (uff_raw - uff_raw.min()) - (dft_raw - dft_raw.min())
    centered = residuals - residuals.mean()

    print(f"\n{'─' * 60}")
    print(f"  Conformers:        {len(dft_raw)}")
    print(f"  DFT range:         {(dft_raw - dft_raw[ref_idx]).max():.1f} kcal/mol")
    print(f"  UFF range:         {uff_raw.max() - uff_raw.min():.1f} kcal/mol")
    print(f"  Spearman rho:      {rho:+.3f}")
    print(f"  Kendall tau:       {tau:+.3f}")
    print(
        f"  MAE / RMSE (min-aligned):   {np.mean(np.abs(residuals)):.1f} / "
        f"{np.sqrt(np.mean(residuals**2)):.1f} kcal/mol"
    )
    print(
        f"  MAE / RMSE (mean-aligned):  {np.mean(np.abs(centered)):.1f} / "
        f"{np.sqrt(np.mean(centered**2)):.1f} kcal/mol"
    )
    # Absolute energies at the same level of theory as round 1, so the
    # UFF-filtered set can be placed on the round-1 scale directly.
    round1 = pd.read_csv(cfg.ENERGIES_CSV)[cfg.ENERGY_COL]
    above_round1_gm = dft_raw[ref_idx] - round1.min() * HARTREE_TO_KCAL
    print(
        f"  DFT best:          conformer_{conformer_ids[ref_idx]} "
        f"({above_round1_gm:.1f} kcal/mol above the round-1 FPS global minimum)"
    )
    print(
        f"  UFF would pick:    conformer_{conformer_ids[uff_idx]}, "
        f"DFT rank {dft_rank}/{len(dft_raw)}, "
        f"{dft_raw[uff_idx] - dft_raw[ref_idx]:.1f} kcal/mol above the DFT best"
    )
    print(f"{'─' * 60}\n")

    plot_parity(
        dft_raw,
        uff_raw,
        out_path=out_dir / "uff_parity.png",
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
