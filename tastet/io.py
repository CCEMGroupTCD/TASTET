"""Disk I/O helpers for SOAP features and kernel matrices.

All files use ``np.savez_compressed`` (``.npz``) for significant size
reduction compared to raw ``.npy`` dumps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from ase.db import connect


def build_database(db_path, csv_path, atoms_list, metadata_records):
    """Write structures + per-structure metadata to an ASE database and CSV.

    :param db_path: Path to the .db file (created fresh).
    :param csv_path: Path to the CSV mirror.
    :param atoms_list: List of Atoms objects.
    :param metadata_records: List of dicts, one per structure.
        Each dict is stored as key-value pairs in the ASE database.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = connect(str(db_path))
    for atoms, meta in zip(atoms_list, metadata_records):
        db.write(atoms, **meta)
    pd.DataFrame(metadata_records).to_csv(csv_path, index=False)
    print(f"  Database -> {db_path}  ({len(atoms_list)} structures)")
    print(f"  CSV      -> {csv_path}")


def load_atoms_and_meta(db_path):
    """Read all structures + metadata from an ASE database.

    :param db_path: Path to the .db file.
    :returns: ``(atoms_list, meta)`` where meta is a DataFrame of all
        key-value pairs stored per row. Works regardless of schema.
    :raises FileNotFoundError: If the database does not exist.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}.  Run the 'db' step first."
        )
    db = connect(str(db_path))
    atoms_list = []
    records = []
    for row in db.select():
        atoms_list.append(row.toatoms())
        records.append(row.key_value_pairs)
    return atoms_list, pd.DataFrame(records)


def subsample_from_db(
    db_path: Path | str,
    n: int,
    *,
    seed: int = 42,
) -> tuple[list, pd.DataFrame]:
    """Randomly subsample ``n`` structures from an ASE database.

    If the database contains ``n`` or fewer structures, all structures are returned
    without subsampling.

    :param db_path: Path to the ``.db`` file.
    :type db_path: path
    :param n: Desired number of structures.
    :type n: int
    :param seed: Random state for reproducibility.
    :type seed: int:
    :returns: Subsampled structures and their metadata, row-aligned.
    :rtype: tuple[list, list]
    """
    atoms_list, meta = load_atoms_and_meta(db_path)
    total = len(atoms_list)

    if n >= total:
        print(f"  Subsample: requested {n} ≥ {total} available — using all.")
        return atoms_list, meta

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(total, size=n, replace=False))

    atoms_sub = [atoms_list[i] for i in idx]
    meta_sub = meta.iloc[idx].reset_index(drop=True)
    print(f"  Subsample: {n}/{total} structures (seed={seed})")
    return atoms_sub, meta_sub


def ensure_database(db_path, build_fn, update_fn=None):
    """Build the database if missing, optionally update if it exists.

    :param db_path: Path to the .db file.
    :param build_fn: Called (no args) when the database doesn't exist.
    :param update_fn: Called (no args) when it already exists.
        None = just print that it exists.
    """
    if not db_path.exists():
        build_fn()
    elif update_fn is not None:
        print(f"Database exists -> {db_path}")
        update_fn()
    else:
        print(f"Database exists -> {db_path}")


def save_soap(soap_list: list[np.ndarray], path: Path | str) -> None:
    """Save a list of SOAP feature matrices as a compressed ``.npz``.

    Each structure is stored under a key ``"0"``, ``"1"``, … so that
    variable-length arrays (different numbers of centers) are handled
    naturally.
    """
    arrays = {str(i): feat for i, feat in enumerate(soap_list)}
    np.savez_compressed(str(path), **arrays)


def load_soap(path: Path | str) -> list[np.ndarray]:
    """Load SOAP features saved with :func:`save_soap`."""
    data = np.load(str(path))
    keys = sorted(data.files, key=int)
    return [data[k] for k in keys]


def save_kernel(K: np.ndarray, path: Path | str) -> None:
    """Save a kernel matrix (compressed)."""
    np.savez_compressed(str(path), K=K)


def load_kernel(path: Path | str) -> np.ndarray:
    """Load a kernel matrix."""
    return np.load(str(path))["K"]