"""Validate the surrogate against DFT for the FPS-selected structures.

After the ``select`` step exports POSCARs and they are relaxed with DFT,
this script compares the surrogate (pre-DFT) energies against the DFT
energies of the same structures and produces five plots:

1. DFT energies only.
2. Surrogate (pre-DFT) energies only.
3. Overlay of both — a visual measure of surrogate accuracy.
4. Parity plot — surrogate vs DFT, each axis referenced to its own
   global minimum, annotated with the rank correlations.
5. The same parity plot with the DFT minimum, the DFT maximum and the
   surrogate minimum marked.

The DFT energies are read from ``config.ENERGIES_SELECTED_CSV`` (a
committed input, one row per selected structure). The surrogate energies
are read from ``selected_structures.csv`` (produced by ``select``). The
two are joined on ``configuration_id`` rather than by row position, so a
reordering of either file cannot silently mispair the two methods.

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
from matplotlib.lines import Line2D
from matplotlib import patheffects

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

DFT_ENERGY_LABEL: str = r"$E - E_{\mathrm{gm}}^{\mathrm{DFT}}$ (eV)"
SURROGATE_ENERGY_LABEL: str = r"$E - E_{\mathrm{gm}}$ (eV)"

#: Parity axes. Each method is referenced to its own global minimum, so
#: ``E_gm`` on the surrogate axis is the lowest *surrogate* energy — in
#: general a different structure from the DFT global minimum.
DFT_PARITY_LABEL: str = r"DFT, $E^{\mathrm{DFT}} - E_{\mathrm{gm}}^{\mathrm{DFT}}$ (eV)"
SURROGATE_PARITY_LABEL: str = r"Surrogate, $E - E_{\mathrm{gm}}$ (eV)"


# ── Energy loading ────────────────────────────────────────────────────


def load_energies(
    csv_path: Path, selection_csv: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load DFT and surrogate energies, joined on ``configuration_id``.

    Both arrays are returned in FPS selection order (the row order of
    ``selected_structures.csv``), element *i* of each referring to the
    same structure.  Raw total energies are returned — no shifting is
    applied so that callers can choose their own reference.

    :param csv_path: Path to ``energies_selected.csv`` (columns
        ``configuration_id`` and ``E (eV)``).
    :param selection_csv: Path to ``selected_structures.csv``, produced by
        the ``select`` step.
    :returns: Tuple of ``(dft, surrogate, configuration_ids)``, in selection order.
    :raises ValueError: When the two files cover different configurations.
    """
    dft = pd.read_csv(csv_path)
    surrogate = pd.read_csv(selection_csv)
    print(f"Read {len(dft)} DFT energies from {csv_path.name}")

    missing = set(surrogate["configuration_id"]) - set(dft["configuration_id"])
    if missing:
        raise ValueError(
            f"{csv_path.name} is missing DFT energies for configurations "
            f"{sorted(missing)}."
        )

    merged = surrogate.merge(dft[["configuration_id", "E (eV)"]], on="configuration_id")
    return (
        merged["E (eV)"].values,
        merged["energy_eV"].values,
        merged["configuration_id"].values,
    )


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
    :param color: Marker color.
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
    r"""Parity plot of surrogate vs DFT relative energies.

    DFT is the reference and goes on the x-axis; the surrogate is the
    quantity being judged and goes on the y-axis. Points on the diagonal
    mean the surrogate reproduces the DFT relative energetics exactly.

    Surrogate and DFT live on different absolute scales — different
    Hamiltonians — so each axis is referenced to its own global minimum:

    .. math::

        x_i = E_i^{\mathrm{DFT}} - \min_j E_j^{\mathrm{DFT}}, \qquad
        y_i = E_i^{\mathrm{sur}} - \min_j E_j^{\mathrm{sur}}.

    Both axes then read "energy above the method's own ground state" — the
    quantity each method acts on — and no single structure's error is built
    into the gauge (referencing both methods to one structure would add that
    structure's surrogate error to every residual).

    The annotated statistics pair a ranking measure with a magnitude
    measure. The Spearman :math:`\rho` rank correlation asks whether the
    surrogate *orders* the structures as DFT does — what selection rests
    on — and is invariant to the reference choice. The MAE is that of the
    plotted points about the diagonal: the mean vertical deviation in the
    figure's own gauge, i.e. the error in energies above each method's own
    minimum. The CLI additionally reports Kendall :math:`\tau`, RMSE, and
    the mean-aligned error variants.

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

    # Each method referenced to its own global minimum.
    x = dft_energies - dft_energies.min()
    y = surrogate_energies - surrogate_energies.min()

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

    # y = x reference (perfect surrogate–DFT agreement)
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
    ax.set_ylabel(SURROGATE_PARITY_LABEL)

    # Text-only legend: one ranking statistic, one magnitude statistic
    # (handles hidden). Lower right: with each axis anchored at its own
    # minimum the cloud gathers in the upper-left half, leaving that
    # corner clear.
    stats = [
        Line2D(
            [], [], linestyle="none", marker="", label=rf"Spearman $\rho$ = {rho:.2f}"
        ),
        Line2D([], [], linestyle="none", marker="", label=f"MAE = {mae:.2f} eV"),
    ]
    styled_legend(ax, handles=stats, loc="lower right", handlelength=0, handletextpad=0)

    apply_axis_style(ax, xfmt=".1f", yfmt=".1f")

    if out_path is not None:
        savefig(fig, out_path, dpi=dpi, also_pdf=True)
        print(f"Saved -> {out_path}")
    if show:
        plt.show()
    return fig


#: Marker style for each called-out structure: (palette key, marker, size).
#: Distinct hues *and* distinct shapes, so identity never rests on color alone.
_HIGHLIGHT_STYLES: dict[str, tuple[str, str, float]] = {
    "dft_min": ("dark blue", "D", 110.0),
    "surrogate_min": ("dark orange", "*", 320.0),
    "dft_max": ("magenta", "^", 150.0),
}


def plot_parity_marked(
    dft_energies: np.ndarray,
    surrogate_energies: np.ndarray,
    configuration_ids: np.ndarray,
    *,
    out_path: Path | None = None,
    show: bool = True,
    dpi: int = 300,
) -> plt.Figure:
    """Parity plot with three structures of interest called out.

    Same axes and reference as :func:`plot_parity` -- DFT on x, surrogate on
    y, each relative to its own global minimum -- with the remaining points
    pushed back to a neutral gray so three structures read immediately:

    * the **lowest-energy structure by DFT**, which sits on the ``x = 0``
      axis; its height is how far above its own ground state the surrogate
      places the true one;
    * the **lowest-energy structure by the surrogate**, which sits on the
      ``y = 0`` axis; its horizontal distance from the origin is the DFT
      energy penalty of trusting the surrogate;
    * the **highest-energy structure by DFT**.

    Each is drawn with its own hue *and* marker shape and carries a direct
    label, so the three stay distinguishable in grayscale and for
    color-vision-deficient readers.

    :param dft_energies: Raw DFT energies (not pre-shifted).
    :param surrogate_energies: Raw surrogate energies (not pre-shifted).
    :param configuration_ids: ``configuration_id`` of each structure, in the
        same order as the two energy arrays.
    :param out_path: Save path. *None* = don't save.
    :param show: Call ``plt.show()``.
    :param dpi: Resolution when saving.
    :returns: The figure.
    :raises ValueError: If the three arrays differ in length.
    """
    n = len(dft_energies)
    if not len(surrogate_energies) == len(configuration_ids) == n:
        raise ValueError(
            f"Length mismatch: {n} DFT energies, {len(surrogate_energies)} "
            f"surrogate energies, {len(configuration_ids)} configuration IDs."
        )

    # Each method referenced to its own global minimum, as in plot_parity.
    x = dft_energies - dft_energies.min()
    y = surrogate_energies - surrogate_energies.min()

    marked = {
        "dft_min": (int(np.argmin(dft_energies)), "DFT minimum"),
        "surrogate_min": (int(np.argmin(surrogate_energies)), "Surrogate minimum"),
        "dft_max": (int(np.argmax(dft_energies)), "DFT maximum"),
    }

    set_mpl_style()
    fig, ax = plt.subplots(figsize=(max(6, n * 0.15), 4), constrained_layout=True)

    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = 0.08 * (hi - lo)
    lims = (lo - pad, hi + pad)

    # y = x reference (perfect surrogate-DFT agreement)
    ax.plot(lims, lims, color="0.6", lw=0.8, ls="--", zorder=1)

    # Everything not called out recedes to neutral gray.
    rest = [i for i in range(n) if i not in {idx for idx, _ in marked.values()}]
    ax.scatter(
        x[rest],
        y[rest],
        color="0.75",
        s=55,
        zorder=2,
        edgecolors="white",
        linewidths=0.5,
    )

    handles = []
    for key, (idx, description) in marked.items():
        color_key, marker, size = _HIGHLIGHT_STYLES[key]
        color = palette[color_key]
        ax.scatter(
            x[idx],
            y[idx],
            color=color,
            marker=marker,
            s=size,
            zorder=4,
            edgecolors="white",
            linewidths=1.0,
        )
        # Direct label: keeps identity legible without leaning on the legend,
        # and supplies the contrast relief the lighter hues need.
        ax.annotate(
            f"{configuration_ids[idx]}",
            (x[idx], y[idx]),
            textcoords="offset points",
            xytext=(9, 6),
            fontsize=8,
            color="0.25",
            zorder=5,
            # Halo keeps the label readable where it lands on another marker.
            path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")],
        )
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker=marker,
                color=color,
                markersize=np.sqrt(size) * 0.55,
                markeredgecolor="white",
                # Description only: the cfg id already sits beside the marker
                # as a direct label, and the shorter entries keep the legend
                # narrow enough to fit the clear lower-right corner.
                label=description,
            )
        )

    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect("equal", adjustable="box")
    # Same step on both axes: the two carry the same quantity.
    match_major_ticks(ax)
    ax.set_xlabel(DFT_PARITY_LABEL)
    ax.set_ylabel(SURROGATE_PARITY_LABEL)

    # Lower right: the DFT minimum now sits top-left, where a legend would
    # cover it; the own-minimum axes leave the lower-right corner clear.
    styled_legend(ax, handles=handles, loc="lower right")
    apply_axis_style(ax, xfmt=".1f", yfmt=".1f")

    if out_path is not None:
        savefig(fig, out_path, dpi=dpi, also_pdf=True)
        print(f"Saved -> {out_path}")
    if show:
        plt.show()
    return fig


# ── CLI ───────────────────────────────────────────────────────────────


def main() -> None:
    """Load DFT + surrogate energies and render the five comparison plots."""
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
    dft_raw, surrogate_raw, configuration_ids = load_energies(
        args.energies_csv, args.selection_csv
    )

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

    # Rank correlations (reference-independent), as annotated on the figure.
    rho = float(pd.Series(surrogate_raw).corr(pd.Series(dft_raw), method="spearman"))
    tau = float(pd.Series(surrogate_raw).corr(pd.Series(dft_raw), method="kendall"))
    print(f"  Spearman rho:     {rho:+.3f}")
    print(f"  Kendall tau:      {tau:+.3f}")

    # Quantitative errors depend on how the surrogate-DFT offset is fixed:
    # min-aligned matches the figure's axes; mean-aligned (the least-squares
    # constant) measures pure scatter with no structure privileged.
    residuals = (surrogate_raw - surrogate_raw.min()) - (dft_raw - dft_raw.min())
    centered = residuals - residuals.mean()
    print(
        f"  MAE / RMSE (min-aligned):   {np.mean(np.abs(residuals)):.3f} / "
        f"{np.sqrt(np.mean(residuals**2)):.3f} eV"
    )
    print(
        f"  MAE / RMSE (mean-aligned):  {np.mean(np.abs(centered)):.3f} / "
        f"{np.sqrt(np.mean(centered**2)):.3f} eV"
    )
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

    # ── Plot 4: Parity (surrogate vs DFT, each axis to its own minimum) ──
    plot_parity(
        dft_raw,
        surrogate_raw,
        out_path=out_dir / "energy_profile_parity.png",
        show=show,
    )

    # ── Plot 5: Parity with the three structures of interest marked ─
    plot_parity_marked(
        dft_raw,
        surrogate_raw,
        configuration_ids,
        out_path=out_dir / "energy_profile_parity_marked.png",
        show=show,
    )


if __name__ == "__main__":
    main()
