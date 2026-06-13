"""Data preparation for the Cu-cluster-on-surface use case.

Shared interface (imported by run.py):
    ensure_database()
    subsample()
    resolve_channel_soap()

Use-case-specific internals:
    _build_database()           — reads trajectories, computes formation energies
    update_formation_energies() — recomputes when reference values change

Database schema convention (shared across all use cases)
--------------------------------------------------------
Every row carries a single identifier ``configuration_id`` — sequential,
1-based, gap-free. The position of a structure in the kernel matrix is
``configuration_id - 1``. On subsampling, the subset is re-keyed
``1..N`` so the invariant holds for the working database too;
provenance back to the source run is preserved via ``run_name``.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from ase import Atoms
from ase.db import connect
from ase.io import read

from sads.io import (
    build_database,
    load_atoms_and_meta,
    ensure_database as sads_ensure_database,
)

import config as cfg


# ── Shared interface ──────────────────────────────────────────────────

def ensure_database() -> None:
    """Build the master database or update energies if it already exists."""
    for name in cfg.TARGET_RUNS:
        p = cfg.RUNS_DIR / name
        if not p.exists():
            sys.exit(f"Run directory not found: {p}")
    sads_ensure_database(cfg.db_path(), build_fn=_build_database, update_fn=update_formation_energies)


def resolve_channel_soap(channel: dict) -> dict:
    """Build SOAP keyword arguments for one kernel channel.

    Returns a copy of ``channel["soap"]``.  For this use case there is
    no SMARTS-based centre resolution — centres are always determined by
    ``center_atoms`` in the channel's SOAP dict (or default to all atoms).

    :param channel: A single entry from ``KERNEL_CHANNELS``.
    :returns: Keyword dict ready to pass to :func:`sads.soap_utils.compute_soap`.
    """
    return dict(channel["soap"])


# ── Use-case-specific: database construction ──────────────────────────

def _build_database() -> None:
    """Read trajectories, compute formation energies, write .db + .csv.

    Each structure receives a single identifier ``configuration_id``,
    assigned in read order across ``TARGET_RUNS`` — 1-based and
    gap-free. ``run_name`` is retained for provenance.
    """
    print(f"Building database from {len(cfg.TARGET_RUNS)} runs ...")
    atoms_list: list[Atoms] = []
    records: list[dict] = []
    cid: int = 0

    for run_name in cfg.TARGET_RUNS:
        traj = cfg.RUNS_DIR / run_name / "structures.traj"
        if not traj.exists():
            print(f"  Skipping {run_name} (no structures.traj)")
            continue

        e_surf = cfg.surface_energy(run_name)
        images: list[Atoms] = read(str(traj), index=":")
        print(f"  {run_name}: {len(images)} images (E_surf={e_surf:.4f} eV)")

        for atoms in images:
            n_cu = atoms.get_chemical_symbols().count("Cu")
            e_form = atoms.get_potential_energy() - e_surf - n_cu * cfg.E_CU_BULK
            cid += 1
            atoms_list.append(atoms)
            records.append(dict(
                configuration_id=cid,
                run_name=run_name,
                n_cu=n_cu,
                formation_energy=e_form,
            ))

    build_database(cfg.db_path(), cfg.csv_path(), atoms_list, records)


def update_formation_energies() -> None:
    """Recompute formation energies (e.g. after changing reference values)."""
    print("Updating formation energies ...")
    db = connect(str(cfg.db_path()))
    records: list[dict] = []

    for row in db.select():
        e_total = row.toatoms().get_potential_energy()
        e_form = (
            e_total
            - cfg.surface_energy(row.run_name)
            - row.n_cu * cfg.E_CU_BULK
        )
        db.update(row.id, formation_energy=e_form)
        records.append(dict(
            configuration_id=row.configuration_id,
            run_name=row.run_name,
            n_cu=row.n_cu,
            formation_energy=e_form,
        ))

    pd.DataFrame(records).to_csv(cfg.csv_path(), index=False)
    print(f"  Updated {len(records)} entries")


# ── Use-case-specific: subsampling ────────────────────────────────────

def _subsample_indices_energy(
    eform_arr: np.ndarray, n_samples: int, seed: int, num_bins: int,
) -> np.ndarray:
    """Inverse-density sampling: over-represent rare energy regions.

    :param eform_arr: Array of formation energies.
    :param n_samples: Number of samples to draw.
    :param seed: Random seed.
    :param num_bins: Number of histogram bins for density estimation.
    :returns: Array of selected indices into ``eform_arr``.
    """
    rng = np.random.default_rng(seed)

    counts, bin_edges = np.histogram(eform_arr, bins=num_bins)
    bin_indices = np.digitize(eform_arr, bin_edges) - 1
    bin_indices[bin_indices == num_bins] = num_bins - 1

    weights = np.zeros_like(eform_arr, dtype=float)
    valid = counts[bin_indices] > 0
    weights[valid] = 1.0 / counts[bin_indices][valid]
    p = weights / weights.sum()

    return rng.choice(len(eform_arr), size=n_samples, replace=False, p=p)


def subsample() -> None:
    """Create a subset database from the master via energy-based sampling.

    The subset is re-keyed with a fresh ``configuration_id = 1..N`` so
    the 1-based, gap-free invariant holds for the working database (the
    pipeline indexes the kernel matrix as ``configuration_id - 1``).
    The sampling is fully determined by ``cfg.SEED`` / ``cfg.NUM_BINS``
    and the master energies, so the same settings reproduce the same
    structures; ``run_name`` is retained for provenance.
    """
    if not cfg.master_db_path().exists():
        sys.exit(f"Master database not found: {cfg.master_db_path()}")
    if cfg.db_path().exists():
        print(f"Subset database already exists -> {cfg.db_path()}")
        return

    print(f"Subsampling {cfg.N_SUBSAMPLE} structures from {cfg.MASTER_ANALYSIS_NAME} ...")

    master_atoms, master_meta = load_atoms_and_meta(cfg.master_db_path())
    eform_arr = master_meta["formation_energy"].values

    idx = _subsample_indices_energy(
        eform_arr, cfg.N_SUBSAMPLE, cfg.SEED, cfg.NUM_BINS,
    )

    subset_atoms = [master_atoms[i] for i in idx]
    subset_meta = master_meta.iloc[idx].reset_index(drop=True).copy()
    # Re-key 1-based, gap-free for the working DB (kernel row position
    # is configuration_id - 1).  run_name keeps the link to the source.
    subset_meta["configuration_id"] = np.arange(1, len(subset_meta) + 1)
    subset_records = subset_meta.to_dict("records")

    build_database(cfg.db_path(), cfg.csv_path(), subset_atoms, subset_records)

    meta = {
        "master_analysis_name": cfg.MASTER_ANALYSIS_NAME,
        "n_subsample": cfg.N_SUBSAMPLE,
        "seed": cfg.SEED,
        "num_bins": cfg.NUM_BINS,
    }
    with open(cfg.subsample_meta_path(), "w") as f:
        json.dump(meta, f, indent=4)
    print(f"  Saved metadata -> {cfg.subsample_meta_path()}")