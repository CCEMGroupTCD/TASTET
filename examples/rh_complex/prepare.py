"""Data preparation for the conformer analysis of the Rh complex.

Shared interface (imported by run.py):
    ensure_database()
    load_grid_search_structures() -> tuple[list[Atoms], pd.DataFrame]
    resolve_soap_centers(center_atoms, centers) -> list[int] | None
    resolve_channel_soap()      -> dict
    get_flexible_indices()      -> list[int]

Use-case-specific internals:
    _build_database()          — parses SDF, writes conformers to .db
    load_conformers()          — SDF reader

Database schema convention (shared across all use cases)
--------------------------------------------------------
Every row carries a single identifier ``configuration_id`` — sequential,
1-based, gap-free. Used by every downstream step (SOAP, kernel, kPCA,
selection). The position of a conformer in the kernel matrix is
``configuration_id - 1`` and is computed wherever needed; it is not
stored as a separate column.
"""

from __future__ import annotations

import sys
import warnings

import pandas as pd
from ase import Atoms
from rdkit import Chem

from tastet.io import (
    build_database,
    ensure_database as tastet_ensure_database,
    load_atoms_and_meta,
    subsample_from_db,
)

import config as cfg

_UNSET = object()  # sentinel for resolve_soap_centers default


# ── Shared interface ──────────────────────────────────────────────────


def ensure_database() -> None:
    """Build the conformer database from the SDF if it doesn't exist."""
    tastet_ensure_database(cfg.db_path(), build_fn=_build_database)


def load_grid_search_structures() -> tuple[list[Atoms], pd.DataFrame]:
    """Load structures for the grid search.

    Behavior controlled by ``cfg.GRID_SEARCH_N_SAMPLES``:

    * ``None``  → all conformers in the database.
    * ``int``   → randomly subsample that many for speed.

    :returns: ``(atoms_list, meta)`` with the structures and their
        metadata. ``meta`` always contains ``configuration_id``.
    """
    if not cfg.db_path().exists():
        sys.exit(f"Database not found: {cfg.db_path()}.  Run the 'db' step first.")

    n_samples = getattr(cfg, "GRID_SEARCH_N_SAMPLES", None)
    if n_samples is None:
        atoms_list, meta = load_atoms_and_meta(cfg.db_path())
        print(f"Grid search source: all {len(atoms_list)} conformers from DB")
        return atoms_list, meta

    print(f"Grid search source: {n_samples} subsampled structures from DB")
    return subsample_from_db(cfg.db_path(), n_samples, seed=cfg.SEED)


def resolve_soap_centers(
    center_atoms=_UNSET,
    centers: list[int] | None = None,
) -> list[int] | None:
    """Decide which atoms to use as SOAP centers.

    Priority (highest → lowest):

    1. ``centers`` (explicit 0-based indices) → returned as-is. The same
       indices are used for every structure.
    2. ``center_atoms`` (species names) → return ``None``; SOAP centers
       on those species directly, resolved per structure (no index list
       needed).
    3. ``FLEXIBLE_SMARTS`` set in config → return a list of 0-based
       indices for the SMARTS-matched flexible atoms.
    4. None of the above → return ``None`` (all atoms).

    Whenever a higher-priority source is set, any lower-priority ones
    are ignored with a warning, so the effective selection is never
    silently overridden.

    :param center_atoms: Explicit center species (including ``None``).
        When left as the ``_UNSET`` sentinel (the default), the value
        is looked up from ``SOAP_PARAMS["center_atoms"]``. Passing it
        explicitly lets callers point at a different source of truth
        (e.g. ``FIXED_SOAP_KW`` for the grid search).
    :param centers: Explicit 0-based atom indices. Takes precedence over
        every other source when not ``None``. Like *center_atoms*, the
        usual source is ``SOAP_PARAMS["centers"]`` (or
        ``FIXED_SOAP_KW["centers"]`` for the grid search), passed in by
        the caller.
    :returns: A list of 0-based atom indices (explicit-``centers`` or
        flexible-atom case), or ``None`` (species-based or all-atoms
        case, where SOAP resolves the centers itself).
    """
    if center_atoms is _UNSET:
        center_atoms = cfg.SOAP_PARAMS.get("center_atoms")

    has_center_atoms = bool(center_atoms)
    has_flex_smarts = bool(getattr(cfg, "FLEXIBLE_SMARTS", None))

    # ── Explicit indices win over everything else ─────────────────────
    if centers is not None:
        shadowed = []
        if has_center_atoms:
            shadowed.append(f"center_atoms={center_atoms}")
        if has_flex_smarts:
            shadowed.append("FLEXIBLE_SMARTS")
        if shadowed:
            preview = list(centers)[:5]
            ellipsis = "..." if len(centers) > 5 else ""
            warnings.warn(
                f"\n  Explicit centers={preview}{ellipsis} is set, "
                f"shadowing {' and '.join(shadowed)}.\n"
                f"  → Proceeding with the explicit indices.",
                stacklevel=2,
            )
        print(f"SOAP centers: {len(centers)} explicit indices")
        return list(centers)

    # ── Both center_atoms and FLEXIBLE_SMARTS: warn, prefer the former ─
    if has_center_atoms and has_flex_smarts:
        warnings.warn(
            f"\n  Both center_atoms={center_atoms} and FLEXIBLE_SMARTS="
            f"{cfg.FLEXIBLE_SMARTS} are set in config.\n"
            f"  → Proceeding with center_atoms={center_atoms}.  "
            f"FLEXIBLE_SMARTS will be ignored.\n"
            f"  If you intended to use SMARTS-based centers instead, "
            f"set center_atoms=None.",
            stacklevel=2,
        )
        return None  # SOAP will use center_atoms from the caller's dict

    # ── Only center_atoms ─────────────────────────────────────────────
    if has_center_atoms:
        print(f"SOAP centers: species {center_atoms} (from center_atoms)")
        return None

    # ── Only FLEXIBLE_SMARTS ──────────────────────────────────────────
    if has_flex_smarts:
        flex_idx = get_flexible_indices()
        print(
            f"SOAP centers: {len(flex_idx)} flexible-atom indices (from FLEXIBLE_SMARTS)"
        )
        return flex_idx

    # ── Neither → all atoms ───────────────────────────────────────────
    print("SOAP centers: all atoms (no centers, center_atoms, or FLEXIBLE_SMARTS set)")
    return None


def resolve_channel_soap(channel: dict) -> dict:
    """Build SOAP keyword arguments for one kernel channel.

    Returns a copy of ``channel["soap"]`` with SMARTS-based center
    indices resolved when ``channel["centers_from_smarts"]`` is *True*.
    Otherwise the SOAP dict is returned unchanged (centers are either
    specified via ``center_atoms`` or default to all atoms).

    :param channel: A single entry from ``KERNEL_CHANNELS``.
    :returns: Keyword dict ready to pass to :func:`tastet.soap_utils.compute_soap`.
    """
    soap_kw = dict(channel["soap"])
    if channel.get("centers_from_smarts"):
        soap_kw["centers"] = get_flexible_indices()
    return soap_kw


def get_flexible_indices(
    sdf_path=None,
    flexible_smarts: list[str] | None = None,
    include_h: bool | None = None,
) -> list[int]:
    """Identify flexible atom indices by matching flexible SMARTS.

    Flexible heavy atoms are found via SMARTS.  When *include_h* is
    ``True``, hydrogens bonded exclusively to flexible heavy atoms are
    also included.  Otherwise only the SMARTS-matched heavy atoms are
    returned.

    :param sdf_path: Path to the SDF file.  Defaults to ``cfg.SDF_FILE``.
    :param flexible_smarts: SMARTS patterns.  Defaults to
        ``cfg.FLEXIBLE_SMARTS``.
    :param include_h: Include hydrogens attached to matched heavy atoms.
        Defaults to ``cfg.FLEXIBLE_INCLUDE_H``.
    :returns: Sorted list of 0-based atom indices for flexible atoms.
    """
    sdf_path = sdf_path or cfg.SDF_FILE
    flexible_smarts = flexible_smarts or cfg.FLEXIBLE_SMARTS
    if include_h is None:
        include_h = getattr(cfg, "FLEXIBLE_INCLUDE_H", True)

    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    mol = next(m for m in suppl if m is not None)

    flex_heavy: set[int] = set()
    for smarts in flexible_smarts:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            print(f"  Warning: invalid SMARTS pattern: {smarts!r}")
            continue
        for match in mol.GetSubstructMatches(pattern):
            flex_heavy.update(match)

    flex_all: set[int] = set(flex_heavy)
    if include_h:
        for idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() == 1:
                neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
                if neighbors and all(n in flex_heavy for n in neighbors):
                    flex_all.add(idx)

    flex_idx = sorted(flex_all)
    n_total = mol.GetNumAtoms()
    h_note = " (incl. H)" if include_h else " (no H)"
    print(
        f"Flexible centers: {len(flex_idx)}/{n_total} atoms{h_note} "
        f"({n_total - len(flex_idx)} rigid)"
    )
    return flex_idx


# ── Use-case-specific: database construction ──────────────────────────


def _build_database() -> None:
    """Parse the SDF file and write all conformers to .db + .csv.

    Each row carries one identifier — ``configuration_id`` — assigned in
    SDF reading order, 1-based and gap-free. No filename or
    record-position column is preserved; the database is the source of
    truth from this point onward.
    """
    atoms_list = load_conformers()
    records = [{"configuration_id": i + 1} for i in range(len(atoms_list))]
    build_database(cfg.db_path(), cfg.csv_path(), atoms_list, records)


# ── Use-case-specific: SDF parsing ───────────────────────────────────


def load_conformers(sdf_path=None) -> list[Atoms]:
    """Read all conformers from an SDF file and return ASE Atoms objects.

    Handles both single-record multi-conformer and multi-record
    single-conformer SDF files.

    :param sdf_path: Path to the SDF file. Defaults to ``cfg.SDF_FILE``.
    :returns: List of ASE :class:`Atoms` objects.
    :raises RuntimeError: If the file contains no readable molecules
        or no conformers at all.
    """
    sdf_path = sdf_path or cfg.SDF_FILE
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    mols = [m for m in suppl if m is not None]
    if not mols:
        raise RuntimeError(f"No readable molecules in {sdf_path}")

    atoms_list: list[Atoms] = []

    if len(mols) == 1 and mols[0].GetNumConformers() > 1:
        mol = mols[0]
        symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        for conf in mol.GetConformers():
            atoms_list.append(Atoms(symbols=symbols, positions=conf.GetPositions()))
    else:
        for mol in mols:
            if mol.GetNumConformers() < 1:
                continue
            symbols = [a.GetSymbol() for a in mol.GetAtoms()]
            atoms_list.append(
                Atoms(symbols=symbols, positions=mol.GetConformer(0).GetPositions())
            )

    if not atoms_list:
        raise RuntimeError("No conformers found in SDF.")

    print(
        f"Loaded {len(atoms_list)} conformers ({atoms_list[0].get_chemical_formula()})"
    )
    return atoms_list
