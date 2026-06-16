"""SOAP descriptor utilities for ASE structures and ASE databases."""
from __future__ import annotations
import numpy as np

from typing import Sequence

from ase import Atoms
from dscribe.descriptors import SOAP


def compute_soap(
    atoms_list: Sequence[Atoms],
    *,
    species: list[str] | None = None,
    periodic: bool = False,
    r_cut: float = 4.0,
    n_max: int = 6,
    l_max: int = 6,
    sigma: float = 0.1,
    center_atoms: list[str] | None = None,
    centers: list[int] | None = None,
    average: str = "off",
    normalize: bool = False,
    n_jobs: int = -1,
) -> list[np.ndarray]:
    """Compute per-structure SOAP feature matrices.

    :param atoms_list: Structures to featurise.
    :param species: Chemical species for the SOAP basis. If ``None``, species
        are inferred from ``atoms_list`` using all elements present. When set to
        a subset of elements present in the structures, atoms of unlisted
        species are transparently stripped before computing SOAP, making the
        descriptor blind to those elements.
    :param periodic: Whether to treat structures as periodic.
    :param r_cut: SOAP radial cutoff.
    :param n_max: Number of radial basis functions.
    :param l_max: Maximum degree of spherical harmonics.
    :param sigma: Width of the Gaussian smearing.
    :param center_atoms: Restrict SOAP centres to atoms with these element
        symbols, resolved independently for each structure.
    :param centers: Restrict SOAP centres to these atom indices, using the same
        indices for every structure. Takes precedence over ``center_atoms``.
    :param average: DScribe averaging mode, for example ``"off"``,
        ``"inner"``, or ``"outer"``.
    :param normalize: If ``True``, L2-normalise each per-atom SOAP vector to
        unit length before returning. This can improve numerical stability for
        some kernels, for example REMatch with non-linear metrics.
    :param n_jobs: Number of parallel jobs. Use ``-1`` for all available cores.
    :returns: One feature matrix per structure, each with shape
        ``(n_centres, n_features)``.
    :rtype: list[numpy.ndarray]
    :raises ValueError: If no input structures are provided, or if any
        structure has no SOAP centres after applying ``center_atoms``,
        ``centers``, and ``species``.
    """
    atoms_list = list(atoms_list)

    if not atoms_list:
        raise ValueError("No structures were provided for SOAP featurisation.")

    if species is None:
        species = sorted(
            {s for atoms in atoms_list for s in atoms.get_chemical_symbols()}
        )

    soap = SOAP(
        species=species,
        periodic=periodic,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        average=average,
        sparse=False,
    )

    resolved_species = (
        species if species is not None
        else sorted({s for atoms in atoms_list for s in atoms.get_chemical_symbols()})
    )

    species_set = set(resolved_species)
    all_system_species: set[str] = set()
    for atoms in atoms_list:
        all_system_species.update(atoms.get_chemical_symbols())

    restrict = not all_system_species.issubset(species_set)

    soap_per_structure = []
    for idx, atoms in enumerate(atoms_list):
        if centers is not None:
            target_indices = centers
        elif center_atoms is not None:
            target_indices = [
                i
                for i, symbol in enumerate(atoms.get_chemical_symbols())
                if symbol in center_atoms
            ]
        else:
            target_indices = list(range(len(atoms)))

        if restrict:
            filtered, old_to_new = _filter_atoms_by_species(atoms, species_set)
            target_indices = [
                old_to_new[i] for i in target_indices if i in old_to_new
            ]
            system = filtered
        else:
            system = atoms

        if not target_indices:
            raise ValueError(
                f"No SOAP centres found in structure index {idx}. "
                "This would make the SOAP list shorter than the input "
                "structure list and break row alignment. Check "
                "``center_atoms``, ``centers``, and ``species``."
            )

        features = soap.create(
            system=system,
            centers=target_indices,
            n_jobs=n_jobs,
        )
        features = np.atleast_2d(features)

        if normalize:
            features = _l2_normalize(features)

        soap_per_structure.append(features)

    return soap_per_structure


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation (each SOAP vector → unit length).

    Zero-norm rows are left as zeros to avoid division by zero.
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return features / norms


def _filter_atoms_by_species(
    atoms: Atoms,
    species_set: set[str],
) -> tuple[Atoms, dict[int, int]]:
    """Return a copy of *atoms* containing only the requested species.

    :param atoms: Original structure.
    :param species_set: Set of element symbols to keep.
    :returns: ``(filtered_atoms, old_to_new)`` where *old_to_new* maps
        original atom indices to their positions in the filtered object.
        Atoms whose species are not in *species_set* are absent from
        the mapping.
    """
    symbols = atoms.get_chemical_symbols()
    keep = [s in species_set for s in symbols]

    old_to_new: dict[int, int] = {}
    new_idx = 0
    for old_idx, kept in enumerate(keep):
        if kept:
            old_to_new[old_idx] = new_idx
            new_idx += 1

    return atoms[keep], old_to_new