"""Rank grid-search distance distributions to pick a representation.

Unsupervised diagnostic for choosing a SOAP + kernel combination from
the values in ``pairwise_distances.csv``. Two metrics, ranked
lexicographically:

n_peaks  (primary)
    Number of significant peaks in the KDE of the distance
    distribution. Peaks must clear a prominence threshold of 10% of
    the maximum density, so the count is robust to undersmoothing
    wiggles.

iqr      (secondary, used to break ties on ``n_peaks``)
    Inter-quartile range divided by ``√2`` (the theoretical maximum
    distance for a normalised kernel). A combination that achieves
    high ``n_peaks`` only by zooming into a narrow region of feature
    space will have small ``iqr`` and lose the tiebreak to a
    combination that spreads conformers across the available range.

Every grid-search directory under ``output/<analysis>/grid_search/`` that
contains a ``pairwise_distances.csv`` is discovered automatically and
treated independently (no consolidation across directories). For each
one, results are written to its ``analysis/`` subfolder so that
script-generated diagnostics never mix with the pipeline-generated grid
files:

* ``rankings.csv`` — one row per combination with both metrics and a
  lexicographic ``rank`` column (rank 1 = best). Per-metric ranks
  ``rank_n_peaks`` and ``rank_iqr`` are also saved so you can see
  which combinations win on either axis individually.
* ``top<N>.png`` — KDE overlay on the histogram for the top *N*
  combinations by lexicographic rank.

Tunables (KDE bandwidth, top-*N*, IQR floor) live in ``config.py``
(``GRID_ANALYSIS_*``). Run with::

    python analysis/analyze_gridsearch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity

from tastet.distance import D_MAX
from tastet.plotting._panel import panel_title
from tastet.plotting.style import (
    apply_axis_style,
    palette,
    savefig,
    set_mpl_style,
)

# config.py lives in the example root, one level up from analysis/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg  # noqa: E402


# ── Diagnostics ───────────────────────────────────────────────────────


def _compute_diagnostics(
    distances: np.ndarray,
    x_grid: np.ndarray,
    bandwidth: float,
) -> tuple[int, float, np.ndarray]:
    """Compute n_peaks, IQR/√2, and the KDE for one combination.

    :param distances: 1-D array of pairwise distances for one
        combination.
    :param x_grid: Common KDE evaluation grid (so densities are
        comparable across combinations).
    :param bandwidth: KDE bandwidth, applied uniformly.
    :returns: Tuple ``(n_peaks, iqr, density)`` where *density* is the
        KDE evaluated on *x_grid*. Returns zero/NaN and an empty array
        when the input is too small for any diagnostic.
    """
    if distances.size < 4:
        return 0, float("nan"), np.zeros_like(x_grid)

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(distances[:, None])
    density = np.exp(kde.score_samples(x_grid[:, None]))

    prominence = 0.1 * density.max()
    peaks, _ = find_peaks(density, prominence=prominence)
    n_peaks = int(len(peaks))

    q75, q25 = np.percentile(distances, [75, 25])
    iqr = float((q75 - q25) / np.sqrt(2.0))

    return n_peaks, iqr, density


def _detect_param_columns(df: pd.DataFrame) -> list[str]:
    """Detect parameter columns in a pairwise-distance DataFrame.

    Pair-identifier and distance columns (``id_i``, ``id_j``, ``d``)
    are excluded; everything else is treated as a sweep parameter.

    :param df: DataFrame loaded from ``pairwise_distances.csv``.
    :returns: List of column names that identify a unique combination.
    """
    reserved = {"id_i", "id_j", "d"}
    return [c for c in df.columns if c not in reserved]


# ── Per-directory pipeline ────────────────────────────────────────────


def _process_directory(
    grid_dir: Path,
    *,
    bandwidth: float,
    top_n: int,
    iqr_floor: float | None,
    n_grid: int = 500,
) -> None:
    """Rank combinations in one grid-search directory and render top-N.

    :param grid_dir: Path to the grid-search subdirectory containing
        ``pairwise_distances.csv``.
    :param bandwidth: Shared KDE bandwidth across all combinations.
    :param top_n: Number of best candidates to render.
    :param iqr_floor: Optional minimum ``iqr`` for a combination to be
        eligible for the top-*N* plot. ``None`` disables the filter.
        Disqualified rows are still present in ``rankings.csv`` (with
        an explicit ``disqualified`` flag) so the analysis stays
        transparent.
    :param n_grid: Number of points on the KDE evaluation grid.
    :returns: ``None``. Writes ``rankings.csv`` and ``top<N>.png``
        into the directory's ``analysis/`` subfolder.
    """
    csv_path = grid_dir / "pairwise_distances.csv"
    if not csv_path.exists():
        sys.exit(f"Missing pairwise_distances.csv in {grid_dir}")

    out_dir = cfg.grid_search_analysis_dir(grid_dir)

    print(f"\n[{grid_dir.name}] reading {csv_path.name}")
    df = pd.read_csv(csv_path)
    param_cols = _detect_param_columns(df)
    print(f"  parameter columns: {param_cols}")

    x_grid = np.linspace(0.0, D_MAX, n_grid)

    grouped = df.groupby(param_cols, dropna=False, sort=False)
    n_combos = grouped.ngroups
    print(f"  {n_combos} combinations, KDE bandwidth = {bandwidth:g}")

    rows: list[dict] = []
    distances_by_idx: dict[int, np.ndarray] = {}
    density_by_idx: dict[int, np.ndarray] = {}

    for combo_idx, (key, sub) in enumerate(grouped):
        distances = sub["d"].dropna().to_numpy()
        n_peaks, iqr, density = _compute_diagnostics(
            distances,
            x_grid,
            bandwidth,
        )
        params = (
            dict(zip(param_cols, key))
            if isinstance(key, tuple)
            else {param_cols[0]: key}
        )

        row = {**params, "n_pairs": int(distances.size), "n_peaks": n_peaks, "iqr": iqr}
        rows.append(row)
        distances_by_idx[combo_idx] = distances
        density_by_idx[combo_idx] = density

    rankings = pd.DataFrame(rows)

    # Per-metric ranks (rank 1 = best, ascending=False because for
    # both metrics "more is better"). Useful for inspecting which
    # combinations win on either axis individually.
    rankings["rank_n_peaks"] = (
        rankings["n_peaks"]
        .rank(ascending=False, method="min", na_option="bottom")
        .astype("Int64")
    )
    rankings["rank_iqr"] = (
        rankings["iqr"]
        .rank(ascending=False, method="min", na_option="bottom")
        .astype("Int64")
    )

    # Optional disqualification by IQR floor — applied before the
    # lexicographic ranking so disqualified rows are pushed to the
    # bottom regardless of peak count.
    if iqr_floor is not None:
        disqualified = rankings["iqr"] < iqr_floor
        rankings["disqualified"] = disqualified
        n_disq = int(disqualified.sum())
        print(
            f"  IQR floor = {iqr_floor:g}: {n_disq}/{n_combos} combinations disqualified"
        )
    else:
        rankings["disqualified"] = False

    # Lexicographic rank: primary = n_peaks (more is better), tiebreak
    # = iqr (more is better). Disqualified rows go to the bottom.
    rankings_sorted = rankings.sort_values(
        ["disqualified", "n_peaks", "iqr"],
        ascending=[True, False, False],
        kind="mergesort",  # stable sort so the lexicographic order is honest
    ).reset_index(drop=False)
    rankings_sorted["rank"] = np.arange(1, len(rankings_sorted) + 1, dtype="int64")

    # Bring rank back to the original-order frame for the saved CSV
    rank_map = dict(zip(rankings_sorted["index"], rankings_sorted["rank"]))
    rankings["rank"] = rankings.index.map(rank_map).astype("Int64")

    out_csv = out_dir / "rankings.csv"
    rankings.to_csv(out_csv, index=False)
    print(f"  rankings    -> {out_csv}")

    # ── Top-N figure (lexicographic) ─────────────────────────────────
    eligible = rankings_sorted[~rankings_sorted["disqualified"]]
    if eligible.empty:
        print("  No combinations survive the IQR floor — skipping figure.")
        return

    top = eligible.head(top_n)
    fig_path = out_dir / f"top{top_n}.png"
    _plot_top(
        rankings_top=top,
        distances_by_idx=distances_by_idx,
        density_by_idx=density_by_idx,
        x_grid=x_grid,
        param_cols=param_cols,
        out_path=fig_path,
        bandwidth=bandwidth,
        iqr_floor=iqr_floor,
    )
    print(f"  top{top_n} plot   -> {fig_path}")


# ── Plotting ──────────────────────────────────────────────────────────


def _plot_top(
    rankings_top: pd.DataFrame,
    *,
    distances_by_idx: dict[int, np.ndarray],
    density_by_idx: dict[int, np.ndarray],
    x_grid: np.ndarray,
    param_cols: list[str],
    out_path: Path,
    bandwidth: float,
    iqr_floor: float | None,
) -> None:
    """Render the top-*N* panels.

    Each panel: histogram of distances (light fill) + KDE line on
    top, panel title built from the combination's swept parameters
    via :func:`tastet.plotting._panel.panel_title`, and an
    annotation with the lexicographic rank, ``n_peaks``, and ``iqr``.

    :param rankings_top: Already-sliced rows for the top candidates,
        in lexicographic order.
    :param distances_by_idx: Map from groupby iteration index → that
        group's distance array.
    :param density_by_idx: Same mapping for the cached KDE density.
    :param x_grid: KDE evaluation grid (shared across panels).
    :param param_cols: Names of the columns holding sweep parameters.
    :param out_path: Figure destination.
    :param bandwidth: KDE bandwidth, surfaced in the suptitle for
        traceability.
    :param iqr_floor: IQR floor that was applied (if any), surfaced
        in the suptitle.
    :returns: ``None``.
    """
    set_mpl_style()
    n = len(rankings_top)
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(7.0, 2.0 * n),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )
    if n == 1:
        axes = np.array([axes])

    for ax, (_, row) in zip(axes, rankings_top.iterrows()):
        idx = int(row["index"])
        d = distances_by_idx[idx]
        density = density_by_idx[idx]

        ax.hist(
            d,
            bins=80,
            range=(0.0, D_MAX),
            density=True,
            color=palette["dark blue"],
            alpha=0.30,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.plot(
            x_grid,
            density,
            color=palette["dark orange"],
            linewidth=1.6,
        )
        ax.fill_between(
            x_grid,
            density,
            color=palette["dark orange"],
            alpha=0.15,
        )

        title = panel_title({k: row[k] for k in param_cols})
        n_lines = title.count("\n") + 1
        fontsize = 6 if n_lines > 1 else 8
        pad = 4 + 8 * (n_lines - 1)
        ax.set_title(title, fontsize=fontsize, pad=pad)

        annotation = (
            f"#{int(row['rank'])}  "
            rf"$n_{{\mathrm{{peaks}}}}$ = {int(row['n_peaks'])},  "
            rf"$\mathrm{{IQR}}/\sqrt{{2}}$ = {row['iqr']:.3g}"
        )
        ax.text(
            0.98,
            0.95,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85),
        )

        ax.set_xlim(0.0, D_MAX)
        ax.set_ylabel("Density", fontsize=9)
        apply_axis_style(ax, xfmt=".2f", yfmt=".1f")

    axes[-1].set_xlabel(r"$d(i,\,j)$", fontsize=10)

    suptitle = (
        f"Top {n} by lexicographic ranking "
        rf"($n_{{\mathrm{{peaks}}}}$ → $\mathrm{{IQR}}/\sqrt{{2}}$)"
        f"  (KDE bandwidth = {bandwidth:g}"
    )
    if iqr_floor is not None:
        suptitle += rf", $\mathrm{{IQR}}/\sqrt{{2}} \geq {iqr_floor:g}$"
    suptitle += ")"
    fig.suptitle(suptitle, fontweight="bold", fontsize=11)

    savefig(fig, out_path)
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────


def main() -> None:
    """Analyse every discovered grid-search directory.

    Discovers all ``grid_search/<hash>/`` directories holding a
    ``pairwise_distances.csv`` and ranks each one using the
    ``GRID_ANALYSIS_*`` settings from :mod:`config`.
    """
    base = cfg.analysis_dir() / "grid_search"
    if not base.exists():
        sys.exit(f"No grid_search root: {base}")

    grid_dirs = sorted(
        d
        for d in base.iterdir()
        if d.is_dir() and (d / "pairwise_distances.csv").exists()
    )
    if not grid_dirs:
        sys.exit(f"No grid-search directories with pairwise_distances.csv under {base}")

    print(f"Found {len(grid_dirs)} grid-search directories to analyse.")
    for grid_dir in grid_dirs:
        _process_directory(
            grid_dir,
            bandwidth=cfg.GRID_ANALYSIS_BANDWIDTH,
            top_n=cfg.GRID_ANALYSIS_TOP_N,
            iqr_floor=cfg.GRID_ANALYSIS_IQR_FLOOR,
        )


if __name__ == "__main__":
    main()
