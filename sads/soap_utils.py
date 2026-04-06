from __future__ import annotations
import numpy as np

from pathlib import Path
from typing import Optional, List, Sequence

from ase import Atoms
from ase.db import connect
from ase.neighborlist import neighbor_list
from dscribe.descriptors import SOAP
from dscribe.kernels import AverageKernel, REMatchKernel


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

    Parameters
    ----------
    atoms_list : sequence of Atoms
        Structures to featurise.
    species : list of str, optional
        Chemical species for the SOAP basis.  If *None*, inferred from
        *atoms_list* (all elements).  When set to a subset of elements
        present in the structures, atoms of unlisted species are
        transparently stripped — making the descriptor blind to those
        elements.
    r_cut, n_max, l_max, sigma : float / int
        SOAP hyper-parameters.
    center_atoms : list of str, optional
        Restrict SOAP centres to these elements (resolved per structure).
    centers : list of int, optional
        Restrict SOAP centres to these atom indices (same indices for
        every structure).  Takes precedence over *center_atoms*.
    average : str
        dscribe averaging mode (``"off"``, ``"inner"``, ``"outer"``).
    normalize : bool
        If *True*, L2-normalise each per-atom SOAP vector (row) to unit
        length before returning.  This can improve numerical stability
        of certain kernels (e.g. REMatch with non-linear metrics).
    n_jobs : int
        Parallelism (``-1`` = all cores).

    Returns
    -------
    list of ndarray
        One feature matrix per structure, shape ``(n_centres, n_features)``.
    """
    if species is None:
        species = sorted({s for a in atoms_list for s in a.get_chemical_symbols()})

    return generate_environment_soap(
        atoms_list=list(atoms_list),
        species=species,
        periodic=periodic,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        average=average,
        center_atoms=center_atoms,
        centers=centers,
        normalize=normalize,
        n_jobs=n_jobs,
    )


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


def generate_environment_soap(
        atoms_list,
        species,
        periodic,
        r_cut,
        n_max=2,
        l_max=3,
        sigma=1.0,
        average="off",
        center_atoms=None,
        centers=None,
        use_neighbor_centers=False,
        normalize=False,
        n_jobs=1,
):
    """Compute per-environment SOAP descriptors for a list of structures.

    :param atoms_list: List of ASE Atoms objects.
    :param species: List of unique species for the SOAP basis.  Defines
        which element channels exist in the descriptor.  If this is a
        subset of the elements present in the structures, atoms of
        unlisted elements are transparently stripped before computing
        SOAP (dscribe requires every atom in the system to be in the
        species list).  This makes the descriptor blind to those
        elements — useful for encoding specific chemical environments.
    :param periodic: Set to True if you want the descriptor output to respect the periodicity of the atomic systems (see the pbc-parameter in the constructor of ase.Atoms).
    :param r_cut: Cutoff radius for the SOAP descriptor.
    :param n_max: Maximum number of radial basis functions.
    :param l_max: Maximum degree of spherical harmonics.
    :param sigma: Gaussian smearing width.
    :param average: Averaging method for the SOAP descriptor.
    :param center_atoms: List of element symbols to center on.
    :param centers: List of atom indices to center on (same for every
        structure).  Takes precedence over *center_atoms*.
    :param use_neighbor_centers: If True, use neighbor centers.
    :param normalize: If True, L2-normalise each per-atom SOAP vector
        (row) to unit length.  Improves numerical stability for kernels
        with non-linear metrics (e.g. REMatch + rbf/polynomial).
    :param n_jobs: Number of parallel jobs to run.
    :return: List of shape (num_structures,), each item is an array of
        shape (n_centers_in_structure, soap_vector_dim).
    """
    soap = SOAP(
        species=species,
        periodic=periodic,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        average=average,
        sparse=False
    )

    # Detect whether species is a strict subset of what the structures
    # contain.  If so, we need to filter each Atoms object down to only
    # the requested species before passing it to dscribe (which requires
    # that every atom in the system is in the species list).
    species_set = set(species)
    all_system_species: set[str] = set()
    for atoms in atoms_list:
        all_system_species.update(atoms.get_chemical_symbols())
    restrict = not all_system_species.issubset(species_set)

    soap_per_structure = []
    for idx, atoms in enumerate(atoms_list):
        if centers is not None:
            # Explicit atom indices — use directly
            target_indices = centers
        elif use_neighbor_centers and center_atoms:
            padding = sigma * np.sqrt(-2 * np.log(0.001))
            eff_cut = r_cut + padding
            ci, cj = neighbor_list('ij', atoms, eff_cut)
            neighbor_indices = set()
            for i, j in zip(ci, cj):
                if atoms[i].symbol in center_atoms:
                    neighbor_indices.add(j)
                elif atoms[j].symbol in center_atoms:
                    neighbor_indices.add(i)
            target_indices = list(neighbor_indices)
        elif center_atoms is not None:
            # Filter by element symbol
            target_indices = [
                i for i, symbol in enumerate(atoms.get_chemical_symbols())
                if symbol in center_atoms
            ]
        else:
            target_indices = list(range(len(atoms)))

        # ── Species filtering ────────────────────────────────────────
        if restrict:
            filtered, old_to_new = _filter_atoms_by_species(atoms, species_set)
            target_indices = [
                old_to_new[i] for i in target_indices if i in old_to_new
            ]
            system = filtered
        else:
            system = atoms

        if target_indices:
            features = soap.create(system=system, centers=target_indices, n_jobs=n_jobs)
            if normalize:
                features = _l2_normalize(np.atleast_2d(features))
            soap_per_structure.append(features)
        else:
            print(f"Warning: No target atoms found in structure index {idx}. Skipping.")

    if not soap_per_structure:
        raise ValueError("No SOAP features were generated. Check your center_atoms/centers parameter.")

    return soap_per_structure


def build_average_kernel(
    features_list: List[np.ndarray],
    metric: str = "linear",
    weights: Optional[np.ndarray] = None,
    **kwargs
) -> np.ndarray:
    """Build an average kernel matrix from per-structure SOAP features.

    :param features_list: List of SOAP feature arrays (ragged list of 2D arrays).
    :param metric: Similarity metric string (e.g., 'linear', 'rbf', 'polynomial').
    :param weights: Optional 1D array of weights (length N).
    :param kwargs: Additional scikit-learn parameters (e.g. gamma).
    :return: The square (N x N) kernel matrix.
    """
    avg_kernel = AverageKernel(metric=metric, **kwargs)
    K_matrix = avg_kernel.create(features_list)

    if weights is not None:
        weights = np.asarray(weights)
        if len(weights) != K_matrix.shape[0]:
            raise ValueError(
                f"Weights dimension ({len(weights)}) does not match "
                f"number of samples ({K_matrix.shape[0]})."
            )
        K_matrix = K_matrix * np.outer(weights, weights)

    return K_matrix


def build_rematch_kernel(soap_list, metric, alpha, threshold=1e-6, **kwargs):
    """Build a REMatch kernel matrix from per-structure SOAP features.

    :param soap_list: List of SOAP descriptors for each structure.
    :param metric: Kernel metric to use.
    :param alpha: REMatch regularisation parameter.
    :param threshold: Convergence threshold.
    :param kwargs: Additional keyword arguments for the kernel.
    :return: Kernel matrix for the structures.
    """
    avg_kern = REMatchKernel(metric=metric, alpha=alpha, threshold=threshold, **kwargs)
    K = avg_kern.create(soap_list)
    return K


def kernel_to_distance_matrix(K):
    """Convert a kernel (similarity) matrix to its kernel-induced distance.

    :param K: Kernel matrix.
    :return: Distance matrix, ``d(i,j) = sqrt(K[i,i] - 2K[i,j] + K[j,j])``.
    """
    diag = np.diag(K)[:, None]
    D = np.sqrt(diag + diag.T - 2*K)
    return D


def get_soap_from_db(db_path: Path, soap_params: dict, selection: dict = None) -> tuple[list, list[int]]:
    """Compute SOAP descriptors for candidates stored in an ASE database.

    :param db_path: Path to the ASE database file.
    :param soap_params: Dictionary containing the SOAP parameters.
    :param selection: Optional dict of key-value pairs to filter rows.
    :return: Tuple of (list of SOAP descriptor arrays, list of row ids).
    """
    db = connect(str(db_path))
    selection = selection or {}

    atoms_list = []
    row_ids = []
    for row in db.select(**selection):
        atoms_list.append(row.toatoms())
        row_ids.append(row.id)

    all_species = set()
    for atoms in atoms_list:
        all_species.update(atoms.get_chemical_symbols())

    soap_list = generate_environment_soap(
        atoms_list=atoms_list,
        species=sorted(all_species),
        periodic=soap_params['periodic'],
        r_cut=soap_params['r_cut'],
        n_max=soap_params['n_max'],
        l_max=soap_params['l_max'],
        average=soap_params['average'],
        center_atoms=soap_params.get('center_atoms'),
        centers=soap_params.get('centers'),
        normalize=soap_params.get('normalize', False),
        n_jobs=1,
    )

    if len(soap_list) != len(row_ids):
        raise ValueError(
            f"SOAP returned {len(soap_list)} descriptors but database "
            f"query returned {len(row_ids)} rows."
        )

    return soap_list, row_ids