"""Round-2 supervised re-optimisation + incremental selection — Rh complex.

Round 1 selected 34 conformers by unsupervised farthest-point sampling
and sent them to DFT. With their energies now in hand, this script:

  1. ``cka_grid`` — re-optimises the representation by a CKA-scored grid
     search on the 34 labelled conformers, so the kernel is aligned with
     the conformer energy landscape. Works in either mode:

       * single-kernel  (USE_TENSOR_PRODUCT = False) → sweeps
         ``SOAP_GRID`` × ``KERNEL_GRID``;
       * multi-channel  (USE_TENSOR_PRODUCT = True)  → sweeps each
         channel's ``soap_grid`` × ``kernel_grid`` and scores the
         combined (product/sum) kernel.

     Inspect the heatmaps, then set the best parameters in ``config.py``.
  2. ``select`` — builds the kernel over *all* conformers with those
     parameters (re-using the example's own ``soap`` / ``kernel`` steps,
     so single- and multi-channel both work) and picks the remaining
     budget by farthest-point sampling *seeded with the round-1 set*, so
     the new structures are maximally complementary.

Run under a dedicated ``ANALYSIS_NAME`` (e.g. ``"round2_cka"``) so the
round-2 outputs do not mix with round 1.

Usage::

    # in config.py: ANALYSIS_NAME = "round2_cka"
    python round2.py cka_grid          # then inspect grid_search heatmaps
    # set the chosen params in config.py
    python round2.py select            # writes the remaining-N selection
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase.io import write as ase_write

from tastet.io import load_atoms_and_meta, load_kernel
from tastet.pipeline import grid_search_step
from tastet.sweep.multichannel import grid_search_multichannel_step
from tastet.kpca import fit_kpca
from tastet.metrics.cka import CKAScorer
from tastet.selection import select_additional

import config as cfg
from prepare import ensure_database, resolve_channel_soap


# ── Round-2 settings ──────────────────────────────────────────────────
ENERGIES_CSV = cfg.USE_CASE_DIR / "input" / "optimised_energies.csv"
TOTAL_BUDGET: int = 69          # total structures wanted across both rounds
ENERGY_COL: str = "E (Ha)"      # absolute DFT energy column in ENERGIES_CSV
# The CKA target-kernel type ("linear" / "rbf") is read from config.py
# (cfg.CKA_TARGET_KERNEL), so the same setting feeds grid_search_tag() and
# keeps linear/rbf sweeps in separate output directories.
# The CKA target is E (Ha) shifted to the most stable labelled conformer
# (E - min E).  CKA is invariant to that offset — the target kernel is
# mean-centred internally — so the shift does not change the score; it is
# applied only for numerical safety, since a linear target kernel built
# from absolute energies (~-4862 Ha) buries the ~0.01 Ha signal under a
# ~2e7 offset and loses precision when centred.  We use E, not the CSV's
# dE column, because dE here is not a clean affine transform of E (its
# reference is inconsistent), so the two are not interchangeable.


# ── Helpers ───────────────────────────────────────────────────────────

def _load_round1_energies() -> pd.DataFrame:
    """Read the round-1 DFT energies and parse configuration ids.

    The ``file`` column holds ``conformer_<id>`` stems matching the
    round-1 ``.xyz`` filenames; ``<id>`` is the ``configuration_id``.

    :returns: DataFrame with columns ``configuration_id`` and
        :data:`ENERGY_COL` (absolute energy in Ha), one row per labelled
        conformer.
    :raises SystemExit: If the CSV is missing.
    """
    if not ENERGIES_CSV.exists():
        sys.exit(f"Energies CSV not found: {ENERGIES_CSV}")
    df = pd.read_csv(ENERGIES_CSV)
    df["configuration_id"] = df["file"].apply(lambda s: int(str(s).split("_")[-1]))
    print(f"Round-1 labelled conformers: {len(df)}  "
          f"(target = {ENERGY_COL} relative to the most stable)")
    return df[["configuration_id", ENERGY_COL]]


def _position_of(meta: pd.DataFrame) -> dict[int, int]:
    """Map ``configuration_id`` → row position in *meta* (and the kernel)."""
    return {int(c): i for i, c in enumerate(meta["configuration_id"].values)}


def _report_top(scorer) -> None:
    """Print the top combinations by score from the grid-search results."""
    results_csv = cfg.grid_search_csv()
    if results_csv.exists():
        res = pd.read_csv(results_csv)
        score_col = scorer.name if scorer.name in res.columns else res.columns[-1]
        top = res.sort_values(score_col, ascending=False).head(5)
        print(f"\nTop 5 by {score_col!r} (inspect the heatmaps too — only a "
              f"few labelled points, so treat as guidance):")
        print(top.to_string(index=False))
        print(f"\nResults : {results_csv}")
        print(f"Heatmaps: {cfg.grid_search_heatmap_path()}")


# ── Plot helpers ──────────────────────────────────────────────────────

def _plot_selection_2d(p, ev, preselected, sel_idx, save_path,
                       *, seed_values=None, seed_label="") -> None:
    """2-D kPCA scatter: pool (grey) + round-1 seed + round-2 picks.

    When *seed_values* is given, the round-1 seed points are coloured by
    it (with a colorbar) instead of a flat colour.

    :param p: kPCA projections, shape (N, >=2).
    :param ev: Explained-variance percentages per component.
    :param preselected: Kernel-row positions of the round-1 seed.
    :param sel_idx: Kernel-row positions of the round-2 picks.
    :param save_path: Output PNG path.
    :param seed_values: Optional per-seed values to colour by (aligned
        with *preselected*).
    :param seed_label: Colorbar label when *seed_values* is given.
    :returns: ``None``.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(p[:, 0], p[:, 1], s=16, c="0.85", edgecolors="none",
               label=f"all ({len(p)})", zorder=1)
    if seed_values is None:
        ax.scatter(p[preselected, 0], p[preselected, 1], s=60, c="#1f5fbf",
                   edgecolors="white", linewidths=0.5,
                   label=f"round 1 ({len(preselected)})", zorder=2)
    else:
        sc = ax.scatter(p[preselected, 0], p[preselected, 1], s=70,
                        c=seed_values, cmap="viridis",
                        edgecolors="white", linewidths=0.5, zorder=2)
        fig.colorbar(sc, ax=ax, label=seed_label)
    ax.scatter(p[sel_idx, 0], p[sel_idx, 1], s=90, marker="*", c="#d6336c",
               edgecolors="white", linewidths=0.5,
               label=f"round 2 ({len(sel_idx)})", zorder=3)
    ax.set_xlabel(f"kPC1 ({ev[0]:.1f}%)")
    ax.set_ylabel(f"kPC2 ({ev[1]:.1f}%)")
    ax.set_title("Round-2 incremental selection")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def _plot_selection_3d(p, ev, preselected, sel_idx, save_path,
                       *, seed_values=None, seed_label="") -> None:
    """3-D kPCA scatter: pool (grey) + round-1 seed + round-2 picks.

    See :func:`_plot_selection_2d`; this is the three-component variant.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=10, c="0.85", edgecolors="none",
               label=f"all ({len(p)})", depthshade=False)
    if seed_values is None:
        ax.scatter(p[preselected, 0], p[preselected, 1], p[preselected, 2], s=45,
                   c="#1f5fbf", edgecolors="white", linewidths=0.4,
                   label=f"round 1 ({len(preselected)})", depthshade=False)
    else:
        sc = ax.scatter(p[preselected, 0], p[preselected, 1], p[preselected, 2], s=55,
                        c=seed_values, cmap="viridis", edgecolors="white",
                        linewidths=0.4, depthshade=False)
        fig.colorbar(sc, ax=ax, label=seed_label, shrink=0.6, pad=0.1)
    ax.scatter(p[sel_idx, 0], p[sel_idx, 1], p[sel_idx, 2], s=70, marker="*",
               c="#d6336c", edgecolors="white", linewidths=0.4,
               label=f"round 2 ({len(sel_idx)})", depthshade=False)
    ax.set_xlabel(f"kPC1 ({ev[0]:.1f}%)")
    ax.set_ylabel(f"kPC2 ({ev[1]:.1f}%)")
    ax.set_zlabel(f"kPC3 ({ev[2]:.1f}%)")
    ax.set_title("Round-2 incremental selection")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# ── Subcommands ───────────────────────────────────────────────────────

def cka_grid() -> None:
    """CKA-scored grid search on the round-1 labelled set.

    Dispatches by ``cfg._use_channels()``: a single-kernel sweep
    (:func:`tastet.pipeline.grid_search_step`) or a multi-channel sweep
    (:func:`tastet.sweep.multichannel.grid_search_multichannel_step`),
    both scored by CKA against the round-1 energies.
    """
    ensure_database()
    atoms, meta = load_atoms_and_meta(cfg.db_path())
    pos_of = _position_of(meta)

    e_df = _load_round1_energies()
    missing = [c for c in e_df["configuration_id"] if c not in pos_of]
    if missing:
        sys.exit(f"configuration_ids not in database: {missing}")

    labelled_atoms = [atoms[pos_of[c]] for c in e_df["configuration_id"]]
    e = e_df[ENERGY_COL].to_numpy(dtype=float)
    target = e - e.min()   # relative to the most stable labelled conformer (offset-safe)
    ids = e_df["configuration_id"].to_numpy()
    scorer = CKAScorer(target_kernel=cfg.CKA_TARGET_KERNEL)

    if cfg._use_channels():
        grid_search_multichannel_step(
            cfg=cfg,
            atoms_list=labelled_atoms,
            ids=ids,
            channels=cfg.KERNEL_CHANNELS,
            scorer=scorer,
            target=target,
            resolve_channel_soap=resolve_channel_soap,
        )
    else:
        grid_search_step(
            cfg=cfg,
            atoms_list=labelled_atoms,
            ids=ids,
            scorer=scorer,
            target=target,
            fixed_soap_kw=cfg.FIXED_SOAP_KW,
        )

    _report_top(scorer)
    print("\nNext: set the chosen params in config.py, then run:  "
          "python round2.py select")


def select() -> None:
    """Build the full kernel and pick the remaining budget by seeded FPS.

    Re-uses the example's own ``soap`` / ``kernel`` steps so the kernel is
    built (and cached) exactly as the main pipeline would for the current
    parameters — single- or multi-channel. Then selects
    ``TOTAL_BUDGET − N_round1`` new structures by farthest-point sampling
    seeded with the round-1 selection. Writes the new ids, ``.xyz``
    files, and a kPCA plot distinguishing the two rounds.
    """
    import run  # reuse the example's _soap / _kernel (handles both modes)

    ensure_database()
    atoms, meta = load_atoms_and_meta(cfg.db_path())

    run._soap()
    run._kernel()
    K = load_kernel(cfg.kernel_path())

    # Round-1 set → kernel-row positions (the FPS seed).
    pos_of = _position_of(meta)
    e_df = _load_round1_energies()
    preselected = np.array([pos_of[c] for c in e_df["configuration_id"]], dtype=int)

    k_remaining = TOTAL_BUDGET - len(preselected)
    if k_remaining <= 0:
        sys.exit(f"Budget already met: {len(preselected)} ≥ {TOTAL_BUDGET}.")
    print(f"Selecting {k_remaining} more (budget {TOTAL_BUDGET}, "
          f"{len(preselected)} already chosen) ...")

    selected, idx_pool, sel_idx = select_additional(
        K, meta,
        preselected_indices=preselected,
        k=k_remaining,
    )

    out_dir = cfg.kernel_dir() / "selection_round2"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "selected_round2.csv"
    selected.to_csv(csv_path, index=False)
    print(f"  New selection CSV -> {csv_path}  ({len(selected)} structures)")

    xyz_dir = out_dir / "xyz"
    xyz_dir.mkdir(exist_ok=True)
    for pos in sel_idx:
        cid = int(meta["configuration_id"].iloc[pos])
        ase_write(str(xyz_dir / f"conformer_{cid}.xyz"), atoms[pos])
    print(f"  New .xyz files    -> {xyz_dir}  ({len(sel_idx)} files)")

    # ── Plots: 2-D + 3-D, plain and energy-coloured ──────────────────
    # preselected is in e_df row order, so the energies line up with it.
    result = fit_kpca(K, n_components=3)
    p = result.projections
    ev = result.explained_variance * 100

    e = e_df[ENERGY_COL].to_numpy(dtype=float)
    seed_dE = (e - e.min()) * 627.509474   # Ha -> kcal/mol, relative to most stable
    elabel = "round 1  \u0394E (kcal/mol)"

    _plot_selection_2d(p, ev, preselected, sel_idx, out_dir / "selection.png")
    _plot_selection_3d(p, ev, preselected, sel_idx, out_dir / "selection_3d.png")
    _plot_selection_2d(p, ev, preselected, sel_idx, out_dir / "selection_energy.png",
                       seed_values=seed_dE, seed_label=elabel)
    _plot_selection_3d(p, ev, preselected, sel_idx, out_dir / "selection_3d_energy.png",
                       seed_values=seed_dE, seed_label=elabel)
    print(f"  Plots             -> {out_dir}  "
          f"(selection[_3d][_energy].png)")


# ── CLI dispatch ──────────────────────────────────────────────────────

STEPS = {"cka_grid": cka_grid, "select": select}

USAGE = """\
Round-2 supervised re-optimisation + incremental selection.

  cka_grid   CKA grid search on the round-1 labelled conformers
             (single- or multi-channel, per USE_TENSOR_PRODUCT).
  select     Build the kernel and pick the remaining budget (seeded FPS).

  Typical flow:
    1. set ANALYSIS_NAME = "round2_cka" in config.py
    2. python round2.py cka_grid          # inspect grid_search heatmaps
    3. set best params in config.py (single- or multi-channel)
    4. python round2.py select
"""


def main() -> None:
    """Dispatch the requested round-2 subcommand."""
    args = sys.argv[1:]
    if not args or args[0] in ("help", "--help"):
        print(USAGE)
        return
    name = args[0]
    if name not in STEPS:
        print(USAGE)
        sys.exit(f"Unknown step: {name!r}")
    STEPS[name]()


if __name__ == "__main__":
    main()