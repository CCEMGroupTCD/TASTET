"""Shared helpers for the round-2 scripts.

Holds the namespace switch (:func:`activate_round2`), round-1 energy
loading and kPCA-space geometry, and the three-layer kPCA scatter
plotters reused by every selection strategy. All user-facing knobs live
in ``config.py`` (Round 2 section); this module holds only logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

from tastet.plotting.style import (
    apply_axis_style,
    cmap,
    palette,
    savefig,
    set_mpl_style,
    styled_legend,
)

# config.py lives in the example root, one level up from round2/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg  # noqa: E402

# Hartree → kcal/mol, for ΔE colourbars.
HARTREE_TO_KCAL: float = 627.509474


# ─────────────────────────────────────────────────────────────────────
#  Namespace
# ─────────────────────────────────────────────────────────────────────


def activate_round2() -> None:
    """Point :mod:`config` at the round-2 output namespace and channels.

    Round 2 uses a separate ``ANALYSIS_NAME`` and its own re-optimised
    ``KERNEL_CHANNELS`` / ``KERNEL_COMBINE`` (the CKA-optimal linear
    representation). Setting them on the shared ``config`` module here
    means the reused ``run._soap`` / ``run._kernel`` wrappers and every
    ``config`` path helper resolve to ``output/<ROUND2_ANALYSIS_NAME>/``
    with the round-2 parameters — production (``run.py``) is untouched.

    Idempotent; call once at the start of each subcommand.
    """
    cfg.ANALYSIS_NAME = cfg.ROUND2_ANALYSIS_NAME
    cfg.KERNEL_CHANNELS = cfg.ROUND2_KERNEL_CHANNELS
    cfg.KERNEL_COMBINE = cfg.ROUND2_KERNEL_COMBINE


# ─────────────────────────────────────────────────────────────────────
#  Round-1 energies & kPCA-space geometry
# ─────────────────────────────────────────────────────────────────────


def position_of(meta: pd.DataFrame) -> dict[int, int]:
    """Map ``configuration_id`` to its 0-based kernel-row position.

    :param meta: Structure metadata (must have ``configuration_id``).
    :returns: Mapping from configuration_id (int) to row index (int).
    """
    return {int(c): i for i, c in enumerate(meta["configuration_id"])}


def load_round1_energies() -> pd.DataFrame:
    """Load and validate the round-1 energies CSV.

    Parses ``configuration_id`` from the ``file`` column by taking the
    last underscore-separated component (e.g. ``conformer_732`` → ``732``).

    :returns: DataFrame with an added ``configuration_id`` column,
        alongside the original ``file`` and energy columns.
    :raises SystemExit: If the CSV is missing or malformed.
    """
    if not cfg.ENERGIES_CSV.exists():
        sys.exit(f"Missing round-1 energies CSV: {cfg.ENERGIES_CSV}")
    df = pd.read_csv(cfg.ENERGIES_CSV)
    if "file" not in df.columns:
        sys.exit(f"{cfg.ENERGIES_CSV} must have a 'file' column.")
    if cfg.ENERGY_COL not in df.columns:
        sys.exit(f"{cfg.ENERGIES_CSV} must have a {cfg.ENERGY_COL!r} column.")
    df = df.copy()
    df["configuration_id"] = df["file"].apply(lambda s: int(str(s).split("_")[-1]))
    return df


def round1_dE(e_df: pd.DataFrame) -> np.ndarray:
    """Relative energies (kcal/mol) for the round-1 conformers.

    :param e_df: Output of :func:`load_round1_energies`.
    :returns: ``(E − min E) · 627.509`` in the row order of *e_df*.
    """
    e = e_df[cfg.ENERGY_COL].to_numpy(dtype=float)
    return (e - e.min()) * HARTREE_TO_KCAL


def study_wide_dE_by_cid() -> dict[int, float]:
    """ΔE (kcal/mol) per conformer, referenced to the study-wide minimum.

    Pools the round-1 and round-2 DFT energies and subtracts the single
    lowest absolute energy across both (the found global minimum
    :math:`E_\\mathrm{gm}`), so every energy figure in the example shares
    one zero. Used to colour both the round-1 seeds and the round-2 picks
    on a single shared colourbar.

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


def in_box(p: np.ndarray, box: dict) -> np.ndarray:
    """Return a boolean mask selecting points inside the kPCA box.

    Missing axis keys and ``None`` bounds are treated as unbounded.
    A ``kpc3`` key is silently ignored if ``p`` has fewer than three
    components (i.e. the caller passed a 2-component projection).

    :param p: kPCA projections, shape ``(N, n_components)``.
    :param box: ``ZOOM_BOX``-style dict.
    :returns: Boolean mask of length ``N`` (``True`` inside the box).
    :raises ValueError: If an axis key is not in
        ``{"kpc1", "kpc2", "kpc3"}`` or a bounds value is not a 2-tuple.
    """
    mask = np.ones(len(p), dtype=bool)
    axis_to_col = {"kpc1": 0, "kpc2": 1, "kpc3": 2}
    for axis, bounds in box.items():
        if axis not in axis_to_col:
            raise ValueError(f"Unknown ZOOM_BOX axis: {axis!r}")
        if not (isinstance(bounds, tuple) and len(bounds) == 2):
            raise ValueError(f"ZOOM_BOX[{axis!r}] must be a (lo, hi) tuple")
        col = axis_to_col[axis]
        if col >= p.shape[1]:
            continue
        lo, hi = bounds
        if lo is not None:
            mask &= p[:, col] >= lo
        if hi is not None:
            mask &= p[:, col] <= hi
    return mask


def nearest_indices(
    K: np.ndarray,
    center_pos: int,
    exclude_positions: np.ndarray,
    k: int,
) -> np.ndarray:
    """Return positions of the ``k`` conformers most similar to ``center_pos``.

    Nearest means highest normalised-kernel similarity: the row
    ``K[center_pos, :]`` is sorted descending and the top ``k`` entries
    are taken after filtering out the centre itself and any
    ``exclude_positions``. Using the kernel directly is consistent with
    the FPS / k-medoids selectors elsewhere in the package, which all
    reason about distance via the same kernel.

    :param K: Normalised kernel matrix (diagonal = 1).
    :param center_pos: Row index of the centre conformer.
    :param exclude_positions: Positions to exclude from candidates
        (typically the round-1 set; the centre is also excluded
        unconditionally).
    :param k: Number of nearest neighbours to return.
    :returns: 1-D array of selected positions, sorted nearest-first.
        Length is ``min(k, candidate_count)``.
    """
    n = K.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[exclude_positions] = False
    mask[center_pos] = False
    candidate_idx = np.where(mask)[0]
    sims = K[center_pos, candidate_idx]
    order = np.argsort(-sims)
    return candidate_idx[order[: int(k)]]


def resolve_center(meta: pd.DataFrame, e_df: pd.DataFrame) -> tuple[int, int]:
    """Resolve the nearest-select centre to a (configuration_id, position) pair.

    When :data:`config.ZOOM_CENTER` is ``None``, picks the round-1
    conformer with the lowest absolute energy and announces the choice
    on stdout. When ``ZOOM_CENTER`` is set, validates that the
    configuration_id exists in the database (it need not be a round-1
    member; that simply means the candidate exclusion only removes the
    round-1 set, not the centre via that membership).

    :param meta: Full structure metadata.
    :param e_df: Round-1 energies DataFrame (output of
        :func:`load_round1_energies`).
    :returns: ``(configuration_id, kernel_row_position)``.
    :raises SystemExit: If ``ZOOM_CENTER`` is set but missing from
        the database.
    """
    pos_of = position_of(meta)

    if cfg.ZOOM_CENTER is None:
        e = e_df[cfg.ENERGY_COL].to_numpy(dtype=float)
        i_min = int(np.argmin(e))
        center_cid = int(e_df["configuration_id"].iloc[i_min])
        print(
            f"  ZOOM_CENTER auto-set to configuration_id={center_cid}  "
            f"(lowest-energy round-1)"
        )
    else:
        center_cid = int(cfg.ZOOM_CENTER)
        if center_cid not in pos_of:
            sys.exit(
                f"ZOOM_CENTER={center_cid} not found in the database "
                f"(no such configuration_id)."
            )

    return center_cid, pos_of[center_cid]


# ─────────────────────────────────────────────────────────────────────
#  Plotting (three layers: all / round-1 seeds / new picks)
# ─────────────────────────────────────────────────────────────────────


def _shared_norm(
    seed_values: np.ndarray | None,
    pick_values: np.ndarray | None,
) -> tuple[float | None, float | None]:
    """Common ``(vmin, vmax)`` spanning the seed and pick colour values.

    Pools whichever of *seed_values* / *pick_values* are provided so the
    round-1 seeds and the round-2 picks share one colourbar scale.
    ``NaN`` entries are ignored.

    :param seed_values: Round-1 colour values, or ``None``.
    :param pick_values: Round-2 pick colour values, or ``None``.
    :returns: ``(vmin, vmax)``, or ``(None, None)`` when no values are
        given (matplotlib then auto-scales each artist).
    """
    vals = [v for v in (seed_values, pick_values) if v is not None]
    if not vals:
        return None, None
    pooled = np.concatenate([np.asarray(v, dtype=float).ravel() for v in vals])
    return float(np.nanmin(pooled)), float(np.nanmax(pooled))


def plot_selection_2d(
    p: np.ndarray,
    ev: np.ndarray,
    preselected: np.ndarray,
    sel_idx: np.ndarray,
    save_path: Path | str,
    *,
    seed_values: np.ndarray | None = None,
    pick_values: np.ndarray | None = None,
    seed_label: str = "",
    box: dict | None = None,
    center_pos: int | None = None,
) -> None:
    """Render a 2-D kPCA scatter with three layers + optional centre / box.

    Follows the project plotting style from
    :mod:`tastet.plotting.style`: project palette / gradient cmap,
    ``figsize=(6, 4)`` with ``constrained_layout=True``,
    :func:`apply_axis_style` for ticks and Unicode minus, and
    :func:`savefig` for fixed-canvas-size output.

    :param p: kPCA projections, shape ``(N, >=2)``.
    :param ev: Explained-variance percentages, length ``>=2``.
    :param preselected: Kernel-row positions of round-1 conformers.
    :param sel_idx: Kernel-row positions of new picks.
    :param save_path: PNG output path. A vector ``.pdf`` sibling is
        written alongside it.
    :param seed_values: Optional per-round-1 scalar for the colorbar
        (e.g. dE in kcal/mol). When given, the round-1 layer is
        coloured with the project gradient cmap; otherwise round-1 is
        drawn in ``palette["dark blue"]``.
    :param pick_values: Optional per-pick scalar (e.g. round-2 dE). When
        given, the new picks are coloured by the same gradient cmap and
        share the colourbar normalisation with *seed_values*; otherwise
        they are drawn as ``palette["magenta"]`` stars.
    :param seed_label: Colorbar label.
    :param box: Optional ``ZOOM_BOX``-style dict; when given, a dashed
        rectangle is overlaid for the kpc1 × kpc2 bounds (kpc3 ignored
        in 2-D). Unbounded sides default to the data extent.
    :param center_pos: Optional kernel-row index of a point of interest
        (used by ``nearest_select``). When given, drawn as an open
        black ring around the conformer's existing dot.
    :returns: ``None``.
    """
    set_mpl_style()
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    # Shared colour normalisation across seeds + energy-coloured picks, so
    # both layers read against one colourbar (the study-wide ΔE scale).
    vmin, vmax = _shared_norm(seed_values, pick_values)

    # Background: all conformers (recessive).
    ax.scatter(
        p[:, 0],
        p[:, 1],
        c="lightgrey",
        s=60,
        alpha=0.5,
        edgecolors="none",
        label=f"all ({len(p)})",
    )

    # Middle layer: round-1 seeds, optionally colored by energy.
    if seed_values is not None:
        sc = ax.scatter(
            p[preselected, 0],
            p[preselected, 1],
            c=seed_values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=60,
            alpha=0.9,
            edgecolors="none",
            zorder=3,
            label=f"round 1 ({len(preselected)})",
        )
        cbar = fig.colorbar(sc, ax=ax)
        if seed_label:
            cbar.set_label(seed_label)
    else:
        ax.scatter(
            p[preselected, 0],
            p[preselected, 1],
            c=palette["dark blue"],
            s=60,
            alpha=0.9,
            edgecolors="none",
            zorder=3,
            label=f"round 1 ({len(preselected)})",
        )

    # Top layer: new picks — coloured by energy if available, else magenta.
    if len(sel_idx):
        if pick_values is not None:
            ax.scatter(
                p[sel_idx, 0],
                p[sel_idx, 1],
                marker="*",
                c=pick_values,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=120,
                edgecolors=palette["black"],
                linewidth=0.5,
                zorder=4,
                label=f"new ({len(sel_idx)})",
            )
        else:
            ax.scatter(
                p[sel_idx, 0],
                p[sel_idx, 1],
                marker="*",
                c=palette["magenta"],
                s=120,
                edgecolors=palette["black"],
                linewidth=0.5,
                zorder=4,
                label=f"new ({len(sel_idx)})",
            )

    # Optional centre marker (open black ring around the dot).
    if center_pos is not None:
        ax.scatter(
            [p[center_pos, 0]],
            [p[center_pos, 1]],
            marker="o",
            facecolors="none",
            edgecolors=palette["black"],
            s=200,
            linewidth=1.5,
            zorder=5,
            label="centre",
        )

    # Optional ZOOM_BOX rectangle.
    if box:
        x_lo, x_hi = box.get("kpc1", (None, None))
        y_lo, y_hi = box.get("kpc2", (None, None))
        if not all(v is None for v in (x_lo, x_hi, y_lo, y_hi)):
            if x_lo is None:
                x_lo = p[:, 0].min()
            if x_hi is None:
                x_hi = p[:, 0].max()
            if y_lo is None:
                y_lo = p[:, 1].min()
            if y_hi is None:
                y_hi = p[:, 1].max()
            ax.add_patch(
                Rectangle(
                    (x_lo, y_lo),
                    x_hi - x_lo,
                    y_hi - y_lo,
                    fill=False,
                    linestyle="--",
                    edgecolor=palette["black"],
                    linewidth=1.2,
                    zorder=2,
                )
            )

    ax.set_xlabel(rf"kPC#1 ({ev[0]:.1f}%)")
    ax.set_ylabel(rf"kPC#2 ({ev[1]:.1f}%)")
    apply_axis_style(ax)
    styled_legend(ax, loc="best")

    savefig(fig, Path(save_path), dpi=300, also_pdf=True)
    plt.close(fig)


def plot_selection_3d(
    p: np.ndarray,
    ev: np.ndarray,
    preselected: np.ndarray,
    sel_idx: np.ndarray,
    save_path: Path | str,
    *,
    seed_values: np.ndarray | None = None,
    pick_values: np.ndarray | None = None,
    seed_label: str = "",
    center_pos: int | None = None,
) -> None:
    """Render a static 3-D kPCA scatter with three layers + optional centre.

    Background dots are kept deliberately small and faint
    (``s=15``, ``alpha=0.15``) to mitigate matplotlib's 3-D z-sort
    occlusion of highlighted layers, and the round-1 / picks / centre
    layers are bumped to compensate. Even so, mpl batches every point
    across all artists into a single depth-sorted draw list, so for
    proper viewing-angle independence uncomment the ``plt.show()``
    line near the end of the function and drag the window
    interactively (it blocks until you close it).

    Otherwise follows :mod:`tastet.plotting.kpca` conventions:
    same project palette and gradient cmap as the 2-D companion,
    ``figsize=(6, 4)`` without ``constrained_layout`` (which clips
    z-labels in mpl_toolkits 3-D), and ``depthshade=False`` so colour
    isn't washed out by perspective fog. The box overlay is omitted in
    3-D.

    :param p: kPCA projections, shape ``(N, >=3)``.
    :param ev: Explained-variance percentages, length ``>=3``.
    :param preselected: Kernel-row positions of round-1 conformers.
    :param sel_idx: Kernel-row positions of new picks.
    :param save_path: PNG output path. A vector ``.pdf`` sibling is
        written alongside it.
    :param seed_values: Optional per-round-1 scalar for the colorbar.
    :param pick_values: Optional per-pick scalar (e.g. round-2 dE). When
        given, the new picks are coloured by the same gradient cmap and
        share the colourbar normalisation with *seed_values*; otherwise
        they are drawn as ``palette["magenta"]`` stars.
    :param seed_label: Colorbar label.
    :param center_pos: Optional kernel-row index of a point of interest
        (used by ``nearest_select``). When given, drawn as an open
        black ring around the conformer's existing dot.
    :returns: ``None``.
    """
    set_mpl_style()
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d")

    # Shared colour normalisation across seeds + energy-coloured picks.
    vmin, vmax = _shared_norm(seed_values, pick_values)

    # Background: small, faint, so it doesn't occlude.
    ax.scatter(
        p[:, 0],
        p[:, 1],
        p[:, 2],
        c="lightgrey",
        s=15,
        alpha=0.15,
        edgecolors="none",
        depthshade=False,
        label=f"all ({len(p)})",
    )

    # Middle layer: round-1 seeds, optionally colored by energy.
    if seed_values is not None:
        sc = ax.scatter(
            p[preselected, 0],
            p[preselected, 1],
            p[preselected, 2],
            c=seed_values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=80,
            alpha=0.95,
            edgecolors=palette["black"],
            linewidth=0.3,
            depthshade=False,
            label=f"round 1 ({len(preselected)})",
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.1)
        if seed_label:
            cbar.set_label(seed_label)
    else:
        ax.scatter(
            p[preselected, 0],
            p[preselected, 1],
            p[preselected, 2],
            c=palette["dark blue"],
            s=80,
            alpha=0.95,
            edgecolors=palette["black"],
            linewidth=0.3,
            depthshade=False,
            label=f"round 1 ({len(preselected)})",
        )

    # Top layer: new picks, made larger so they survive the depth sort —
    # coloured by energy if available, else magenta.
    if len(sel_idx):
        if pick_values is not None:
            ax.scatter(
                p[sel_idx, 0],
                p[sel_idx, 1],
                p[sel_idx, 2],
                marker="*",
                c=pick_values,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=180,
                edgecolors=palette["black"],
                linewidth=0.6,
                depthshade=False,
                label=f"new ({len(sel_idx)})",
            )
        else:
            ax.scatter(
                p[sel_idx, 0],
                p[sel_idx, 1],
                p[sel_idx, 2],
                marker="*",
                c=palette["magenta"],
                s=180,
                edgecolors=palette["black"],
                linewidth=0.6,
                depthshade=False,
                label=f"new ({len(sel_idx)})",
            )

    # Optional centre marker.
    if center_pos is not None:
        ax.scatter(
            [p[center_pos, 0]],
            [p[center_pos, 1]],
            [p[center_pos, 2]],
            marker="o",
            facecolors="none",
            edgecolors=palette["black"],
            s=280,
            linewidth=2.0,
            depthshade=False,
            label="centre",
        )

    ax.set_xlabel(rf"kPC#1 ({ev[0]:.1f}%)")
    ax.set_ylabel(rf"kPC#2 ({ev[1]:.1f}%)")
    ax.set_zlabel(rf"kPC#3 ({ev[2]:.1f}%)")
    styled_legend(ax, loc="upper left")

    savefig(fig, Path(save_path), dpi=300, also_pdf=True)
    # plt.show()  # uncomment to drag/rotate the 3-D window (blocks until closed)
    plt.close(fig)
