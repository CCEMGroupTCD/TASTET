"""Round-2 step 2 — incremental selection on the re-optimised kernel.

Three strategies, all building (or loading) the round-2 SOAP + kernel via
the shared ``run._soap`` / ``run._kernel`` wrappers under the round-2
namespace, then picking new conformers that complement the round-1 set:

- ``select``: extends round 1 by FPS-seeded selection up to
  :data:`config.TOTAL_BUDGET` total structures;
- ``zoom_select``: zooms into a kPCA region (:data:`config.ZOOM_BOX`) and
  picks :data:`config.ZOOM_K` more from there (FPS seeded by round-1
  in-box, or k-medoids excluding them);
- ``nearest_select``: picks the :data:`config.ZOOM_K` conformers most
  similar (normalised-kernel) to a point of interest
  (:data:`config.ZOOM_CENTER`, or the lowest-energy round-1 conformer).

Each writes a CSV (``configuration_id`` + kpc1/kpc2/kpc3), one ``.xyz``
per new pick, and plain + energy-coloured 2-D/3-D kPCA scatters. Run::

    python round2/reselect.py select
    python round2/reselect.py zoom_select
    python round2/reselect.py nearest_select
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ase.io import write as ase_write

from tastet.io import load_atoms_and_meta, load_kernel
from tastet.kpca import fit_kpca
from tastet.selection import select_additional, select_structures, _select_fps_seeded

# config.py, prepare.py and run.py live in the example root, one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg  # noqa: E402
import run  # noqa: E402  (reused _soap / _kernel wrappers)
from prepare import ensure_database  # noqa: E402
from _common import (  # noqa: E402
    activate_round2,
    in_box,
    load_round1_energies,
    nearest_indices,
    plot_selection_2d,
    plot_selection_3d,
    position_of,
    resolve_center,
    round1_dE,
    study_wide_dE_by_cid,
)

ENERGY_LABEL = r"$E - E_\mathrm{gm}$ (kcal mol$^{-1}$)"


def _energy_values(
    meta,
    e_df,
    round1_pos: np.ndarray,
    sel_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Seed and pick ΔE (kcal/mol) for the energy-coloured plots.

    The round-1 seeds are always coloured by ΔE. Once the round-2 DFT
    energies exist (:data:`config.ROUND2_ENERGIES_CSV`), both layers are
    referenced to the study-wide minimum :math:`E_\\mathrm{gm}` and the
    new picks are coloured on the same scale; before then (the picks have
    not been relaxed yet) the picks stay magenta (``pick_values`` is
    ``None``).

    :param meta: Full structure metadata.
    :param e_df: Round-1 energies DataFrame (:func:`load_round1_energies`).
    :param round1_pos: Kernel-row positions of round-1 conformers, in
        ``e_df`` row order.
    :param sel_idx: Kernel-row positions of the new picks.
    :returns: ``(seed_values, pick_values)`` aligned with *round1_pos* and
        *sel_idx*; ``pick_values`` is ``None`` when round-2 energies are
        not yet available.
    """
    if not cfg.ROUND2_ENERGIES_CSV.exists():
        return round1_dE(e_df), None
    dE_map = study_wide_dE_by_cid()
    cids = meta["configuration_id"].to_numpy()
    seed_vals = np.array([dE_map[int(cids[pos])] for pos in round1_pos], dtype=float)
    pick_vals = np.array(
        [dE_map.get(int(cids[pos]), np.nan) for pos in sel_idx], dtype=float
    )
    return seed_vals, pick_vals


def _build_round2_kernel():
    """Activate the round-2 namespace, build/load its kernel and kPCA.

    :returns: Tuple ``(atoms, meta, K, p, ev)`` — structures, metadata,
        the round-2 kernel, its 3-component kPCA projections, and the
        explained-variance percentages.
    """
    activate_round2()
    ensure_database()
    atoms, meta = load_atoms_and_meta(cfg.db_path())

    run._soap()
    run._kernel()
    K = load_kernel(cfg.kernel_path())

    result = fit_kpca(K, n_components=3)
    return atoms, meta, K, result.projections, result.explained_variance * 100


def _write_selection(meta, atoms, sel_idx, p, out_dir: Path, csv_name: str) -> None:
    """Write the selection CSV and one ``.xyz`` per new pick.

    :param meta: Full structure metadata.
    :param atoms: Structures aligned with *meta* row order.
    :param sel_idx: Kernel-row positions of the new picks.
    :param p: kPCA projections, shape ``(N, >=3)``.
    :param out_dir: Output directory (created if missing).
    :param csv_name: Filename for the selection CSV.
    :returns: ``None``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = meta.iloc[sel_idx][["configuration_id"]].reset_index(drop=True)
    selected["kpc1"] = p[sel_idx, 0]
    selected["kpc2"] = p[sel_idx, 1]
    selected["kpc3"] = p[sel_idx, 2]
    csv_path = out_dir / csv_name
    selected.to_csv(csv_path, index=False)
    print(f"  Selection CSV -> {csv_path}  ({len(selected)} structures)")

    xyz_dir = out_dir / "xyz"
    xyz_dir.mkdir(exist_ok=True)
    for pos in sel_idx:
        cid = int(meta["configuration_id"].iloc[pos])
        ase_write(str(xyz_dir / f"conformer_{cid}.xyz"), atoms[pos])
    print(f"  New .xyz files -> {xyz_dir}  ({len(sel_idx)} files)")


def select() -> None:
    """Extend the round-1 set up to :data:`config.TOTAL_BUDGET` via seeded FPS.

    Calls :func:`tastet.selection.select_additional` to add picks that
    are farthest (in kernel space) from the round-1 conformers, seeded
    by them. Writes ``selected_round2.csv`` + xyz + four plots (plain
    and energy-coloured 2-D/3-D) under ``selection_round2/``.
    """
    atoms, meta, K, p, ev = _build_round2_kernel()

    pos_of = position_of(meta)
    e_df = load_round1_energies()
    preselected = np.array(
        [pos_of[int(c)] for c in e_df["configuration_id"]],
        dtype=int,
    )

    n_round1 = len(preselected)
    k_remaining = cfg.TOTAL_BUDGET - n_round1
    if k_remaining <= 0:
        sys.exit(
            f"TOTAL_BUDGET ({cfg.TOTAL_BUDGET}) <= round-1 size "
            f"({n_round1}); nothing to add."
        )

    _, _, sel_idx = select_additional(
        K,
        meta,
        preselected_indices=preselected,
        k=k_remaining,
    )

    out_dir = cfg.kernel_dir() / "selection_round2"
    _write_selection(meta, atoms, sel_idx, p, out_dir, "selected_round2.csv")
    print(f"  ({len(sel_idx)} new + {n_round1} round-1 = {cfg.TOTAL_BUDGET} total)")

    seed_dE, pick_dE = _energy_values(meta, e_df, preselected, sel_idx)
    plot_selection_2d(p, ev, preselected, sel_idx, out_dir / "selection.png")
    plot_selection_3d(p, ev, preselected, sel_idx, out_dir / "selection_3d.png")
    plot_selection_2d(
        p,
        ev,
        preselected,
        sel_idx,
        out_dir / "selection_energy.png",
        seed_values=seed_dE,
        pick_values=pick_dE,
        seed_label=ENERGY_LABEL,
    )
    plot_selection_3d(
        p,
        ev,
        preselected,
        sel_idx,
        out_dir / "selection_3d_energy.png",
        seed_values=seed_dE,
        pick_values=pick_dE,
        seed_label=ENERGY_LABEL,
    )
    print(f"  Plots         -> {out_dir}/")


def zoom_select() -> None:
    """Focused selection within a user-defined kPCA box region.

    Picks :data:`config.ZOOM_K` structures from inside
    :data:`config.ZOOM_BOX`. Round-1 conformers in the box are honoured:
    FPS seeds on them; k-medoids excludes them. Outputs go to
    ``selection_zoom/`` and mirror :func:`select`, with the box drawn on
    the 2-D plots.

    :raises SystemExit: When :data:`config.ZOOM_METHOD` is unknown or
        the candidate pool is empty.
    """
    atoms, meta, K, p, ev = _build_round2_kernel()

    pos_of = position_of(meta)
    e_df = load_round1_energies()
    round1_pos = np.array(
        [pos_of[int(c)] for c in e_df["configuration_id"]],
        dtype=int,
    )

    box_mask = in_box(p, cfg.ZOOM_BOX)
    round1_in_box = round1_pos[box_mask[round1_pos]]
    print(
        f"Zoom box: {int(box_mask.sum())}/{len(p)} conformers in box  "
        f"({len(round1_in_box)}/{len(round1_pos)} from round 1)"
    )

    # Candidate pool = in-box minus round-1 in-box (never re-pick).
    pool_mask = box_mask.copy()
    pool_mask[round1_pos] = False
    pool_idx = np.where(pool_mask)[0]

    k = min(cfg.ZOOM_K, len(pool_idx))
    if k < cfg.ZOOM_K:
        print(
            f"  Warning: pool ({len(pool_idx)}) < ZOOM_K ({cfg.ZOOM_K}); selecting all {k}."
        )
    if k == 0:
        sys.exit("  Empty candidate pool -- adjust ZOOM_BOX.")

    if cfg.ZOOM_METHOD == "fps":
        sel_idx = _select_fps_seeded(K, pool_idx, round1_in_box, k)
        print(f"  Seeded FPS: k={k}, pool={len(pool_idx)}, seed={len(round1_in_box)}")
    elif cfg.ZOOM_METHOD == "kmedoids":
        meta_sub = meta.iloc[pool_idx].reset_index(drop=True)
        K_sub = K[np.ix_(pool_idx, pool_idx)]
        _, _, sub_idx = select_structures(
            K_sub,
            meta_sub,
            energy_max=None,
            energy_col=None,
            k=k,
            method="kmedoids",
            seed=cfg.SEED,
        )
        sel_idx = pool_idx[sub_idx]
        print(f"  k-medoids: k={k}, pool={len(pool_idx)}")
    else:
        sys.exit(
            f"Unknown ZOOM_METHOD: {cfg.ZOOM_METHOD!r}  (expected 'fps' or 'kmedoids')"
        )

    out_dir = cfg.kernel_dir() / "selection_zoom"
    _write_selection(meta, atoms, sel_idx, p, out_dir, "selected_zoom.csv")

    seed_dE, pick_dE = _energy_values(meta, e_df, round1_pos, sel_idx)
    plot_selection_2d(
        p,
        ev,
        round1_pos,
        sel_idx,
        out_dir / "selection.png",
        box=cfg.ZOOM_BOX,
    )
    plot_selection_3d(p, ev, round1_pos, sel_idx, out_dir / "selection_3d.png")
    plot_selection_2d(
        p,
        ev,
        round1_pos,
        sel_idx,
        out_dir / "selection_energy.png",
        seed_values=seed_dE,
        pick_values=pick_dE,
        seed_label=ENERGY_LABEL,
        box=cfg.ZOOM_BOX,
    )
    plot_selection_3d(
        p,
        ev,
        round1_pos,
        sel_idx,
        out_dir / "selection_3d_energy.png",
        seed_values=seed_dE,
        pick_values=pick_dE,
        seed_label=ENERGY_LABEL,
    )
    print(f"  Plots         -> {out_dir}/")


def nearest_select() -> None:
    """Pick :data:`config.ZOOM_K` conformers most similar to a centre point.

    The centre is :data:`config.ZOOM_CENTER` when set, or the
    lowest-energy round-1 conformer otherwise. Candidates exclude the
    centre and the rest of round 1. Nearest is the top-``ZOOM_K`` rows of
    ``K[centre, :]``. Outputs go to ``selection_nearest/`` and mirror
    :func:`zoom_select` but with the centre ringed instead of a box.

    :raises SystemExit: When :data:`config.ZOOM_CENTER` is set but not
        in the database, or no candidates remain.
    """
    atoms, meta, K, p, ev = _build_round2_kernel()

    pos_of = position_of(meta)
    e_df = load_round1_energies()
    round1_pos = np.array(
        [pos_of[int(c)] for c in e_df["configuration_id"]],
        dtype=int,
    )

    center_cid, center_pos = resolve_center(meta, e_df)

    n_candidates = len(meta) - len(set(round1_pos.tolist()) | {center_pos})
    k = min(cfg.ZOOM_K, n_candidates)
    if k < cfg.ZOOM_K:
        print(
            f"  Warning: candidates ({n_candidates}) < ZOOM_K ({cfg.ZOOM_K}); selecting all {k}."
        )
    if k == 0:
        sys.exit("  No candidates remain after exclusions.")

    sel_idx = nearest_indices(
        K, center_pos=center_pos, exclude_positions=round1_pos, k=k
    )

    sims = K[center_pos, sel_idx]
    print(
        f"  Nearest: k={k} around configuration_id={center_cid}.  "
        f"Similarities to centre in [{sims.min():.3f}, {sims.max():.3f}]"
    )

    out_dir = cfg.kernel_dir() / "selection_nearest"
    _write_selection(meta, atoms, sel_idx, p, out_dir, "selected_nearest.csv")

    seed_dE, pick_dE = _energy_values(meta, e_df, round1_pos, sel_idx)
    plot_selection_2d(
        p,
        ev,
        round1_pos,
        sel_idx,
        out_dir / "selection.png",
        center_pos=center_pos,
    )
    plot_selection_3d(
        p,
        ev,
        round1_pos,
        sel_idx,
        out_dir / "selection_3d.png",
        center_pos=center_pos,
    )
    plot_selection_2d(
        p,
        ev,
        round1_pos,
        sel_idx,
        out_dir / "selection_energy.png",
        seed_values=seed_dE,
        pick_values=pick_dE,
        seed_label=ENERGY_LABEL,
        center_pos=center_pos,
    )
    plot_selection_3d(
        p,
        ev,
        round1_pos,
        sel_idx,
        out_dir / "selection_3d_energy.png",
        seed_values=seed_dE,
        pick_values=pick_dE,
        seed_label=ENERGY_LABEL,
        center_pos=center_pos,
    )
    print(f"  Plots         -> {out_dir}/")


# ─────────────────────────────────────────────────────────────────────
#  CLI dispatch
# ─────────────────────────────────────────────────────────────────────

STEPS = {
    "select": select,
    "zoom_select": zoom_select,
    "nearest_select": nearest_select,
}

USAGE = """\
Round-2 incremental selection on the re-optimised kernel.

  select          Seeded-FPS picks to reach TOTAL_BUDGET.
  zoom_select     Pick ZOOM_K structures inside ZOOM_BOX (FPS seeded by
                  round-1 in-box, or k-medoids excluding them).
  nearest_select  Pick the ZOOM_K conformers most similar to ZOOM_CENTER
                  (or the lowest-energy round-1 conformer).

  Examples:  python round2/reselect.py select
             python round2/reselect.py nearest_select
"""


def main() -> None:
    """Dispatch each named selection strategy in order."""
    requested = sys.argv[1:]
    if not requested or requested == ["help"] or requested == ["--help"]:
        print(USAGE)
        return
    for name in requested:
        if name not in STEPS:
            print(USAGE)
            sys.exit(f"Unknown step: {name!r}")
        print(f"\n{'=' * 60}\n  Round 2: {name}\n{'=' * 60}")
        STEPS[name]()


if __name__ == "__main__":
    main()
