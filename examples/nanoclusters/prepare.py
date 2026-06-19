"""Data preparation for the Cu-cluster-on-surface use case.

Shared interface (imported by run.py):
    ensure_database()
    load_grid_search_structures() -> tuple[list[Atoms], pd.DataFrame]
    resolve_channel_soap()

Use-case-specific internals:
    _build_database()           — reads the per-run trajectories into the DB

Energy convention
-----------------
Each structure stores its raw potential energy in the ``energy_eV``
column (``energy`` is a reserved ASE-db key) straight from the
trajectory. The only reference used anywhere downstream is the global
minimum ``E_gm`` (the lowest ``energy_eV`` across the set), applied on
the fly as ``E - E_gm``; energies are never referenced to a bulk or
surface reservoir.

Database schema convention (shared across all use cases)
--------------------------------------------------------
Every row carries a single identifier ``configuration_id`` — sequential,
1-based, gap-free. The position of a structure in the kernel matrix is
``configuration_id - 1``; provenance back to the source run is preserved
via ``run_name``.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read

from tastet.io import (
    build_database,
    load_atoms_and_meta,
    ensure_database as tastet_ensure_database,
)

import config as cfg


# ── Shared interface ──────────────────────────────────────────────────


def ensure_database() -> None:
    """Build the database from the per-run trajectories if it is missing."""
    for name in cfg.TARGET_RUNS:
        p = cfg.RUNS_DIR / f"{name}.traj"
        if not p.exists():
            sys.exit(
                f"Per-run trajectory not found: {p}.  "
                f"Run 'python input/split_trajectory.py' first to split "
                f"input/all_runs.traj into per-run trajectories."
            )
    tastet_ensure_database(cfg.db_path(), build_fn=_build_database)


def load_grid_search_structures() -> tuple[list[Atoms], pd.DataFrame]:
    """Load structures for the grid search.

    Draws an energy-balanced subset from the production database via
    inverse-density sampling (:func:`_subsample_indices_energy`), which
    over-represents rare energy regions so the grid search is not
    dominated by the abundant low-energy basin. The draw is fully
    determined by ``cfg.GRID_SEARCH_N_SAMPLES`` / ``cfg.NUM_BINS`` / ``cfg.SEED``,
    so it is reproducible without persisting a separate subset database.

    :returns: ``(atoms_list, meta)`` for the subset. ``meta`` is re-keyed
        with a fresh 1-based, gap-free ``configuration_id`` (the kernel
        row position is ``configuration_id - 1``); ``run_name`` keeps the
        link to the source run.
    :raises SystemExit: If the production database does not exist yet.
    """
    if not cfg.db_path().exists():
        sys.exit(f"Database not found: {cfg.db_path()}.  Run the 'db' step first.")

    atoms_list, meta = load_atoms_and_meta(cfg.db_path())
    idx = _subsample_indices_energy(
        meta["energy_eV"].values,
        cfg.GRID_SEARCH_N_SAMPLES,
        cfg.SEED,
        cfg.NUM_BINS,
    )

    subset_atoms = [atoms_list[i] for i in idx]
    subset_meta = meta.iloc[idx].reset_index(drop=True).copy()
    subset_meta["configuration_id"] = np.arange(1, len(subset_meta) + 1)
    print(
        f"Grid search source: {len(subset_atoms)} energy-balanced structures "
        f"(seed={cfg.SEED}, bins={cfg.NUM_BINS})"
    )
    return subset_atoms, subset_meta


def resolve_channel_soap(channel: dict) -> dict:
    """Build SOAP keyword arguments for one kernel channel.

    Returns a copy of ``channel["soap"]``.  For this use case there is
    no SMARTS-based centre resolution — centres are always determined by
    ``center_atoms`` in the channel's SOAP dict (or default to all atoms).

    :param channel: A single entry from ``KERNEL_CHANNELS``.
    :returns: Keyword dict ready to pass to :func:`tastet.soap_utils.compute_soap`.
    """
    return dict(channel["soap"])


# ── Use-case-specific: database construction ──────────────────────────


def _build_database() -> None:
    """Read the per-run trajectories and write .db + .csv.

    Each structure stores its raw potential energy in the ``energy_eV``
    column (``energy`` is a reserved ASE-db key) straight from the
    trajectory, plus a single identifier ``configuration_id``, assigned
    in read order across ``TARGET_RUNS`` — 1-based and gap-free.
    ``run_name`` is retained for provenance. No reference energy is
    applied here; ``E - E_gm`` is computed downstream from these raw
    values.
    """
    print(f"Building database from {len(cfg.TARGET_RUNS)} runs ...")
    atoms_list: list[Atoms] = []
    records: list[dict] = []
    cid: int = 0

    for run_name in cfg.TARGET_RUNS:
        traj = cfg.RUNS_DIR / f"{run_name}.traj"
        if not traj.exists():
            print(f"  Skipping {run_name} (no {traj.name})")
            continue

        images: list[Atoms] = read(str(traj), index=":")
        print(f"  {run_name}: {len(images)} images")

        for atoms in images:
            n_cu = atoms.get_chemical_symbols().count("Cu")
            cid += 1
            atoms_list.append(atoms)
            records.append(
                dict(
                    configuration_id=cid,
                    run_name=run_name,
                    n_cu=n_cu,
                    energy_eV=atoms.get_potential_energy(),
                )
            )

    build_database(cfg.db_path(), cfg.csv_path(), atoms_list, records)


# ── Use-case-specific: energy-balanced subsampling ───────────────────


def _subsample_indices_energy(
    energy_arr: np.ndarray,
    n_samples: int,
    seed: int,
    num_bins: int,
) -> np.ndarray:
    """Inverse-density sampling: over-represent rare energy regions.

    :param energy_arr: Array of per-structure energies (eV).
    :param n_samples: Number of samples to draw.
    :param seed: Random seed.
    :param num_bins: Number of histogram bins for density estimation.
    :returns: Array of selected indices into ``energy_arr``.
    """
    rng = np.random.default_rng(seed)

    counts, bin_edges = np.histogram(energy_arr, bins=num_bins)
    bin_indices = np.digitize(energy_arr, bin_edges) - 1
    bin_indices[bin_indices == num_bins] = num_bins - 1

    weights = np.zeros_like(energy_arr, dtype=float)
    valid = counts[bin_indices] > 0
    weights[valid] = 1.0 / counts[bin_indices][valid]
    p = weights / weights.sum()

    return rng.choice(len(energy_arr), size=n_samples, replace=False, p=p)
