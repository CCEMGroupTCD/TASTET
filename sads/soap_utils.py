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
    r_cut: float = 4.0,
    n_max: int = 6,
    l_max: int = 6,
    sigma: float = 0.1,
    center_atoms: list[str] | None = None,
    average: str = "off",
    remove_bulk: bool = False,
    n_jobs: int = -1,
) -> list[np.ndarray]:
    """Compute per-structure SOAP feature matrices.

    Parameters
    ----------
    atoms_list : sequence of Atoms
        Structures to featurise.
    species : list of str, optional
        Chemical species to consider.  If *None*, inferred from *atoms_list*.
    r_cut, n_max, l_max, sigma : float / int
        SOAP hyper-parameters.
    center_atoms : list of str, optional
        Restrict SOAP centres to these elements.
    average : str
        dscribe averaging mode (``"off"``, ``"inner"``, ``"outer"``).
    remove_bulk : bool
        Whether to remove bulk-like environments.
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
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        average=average,
        center_atoms=center_atoms,
        remove_bulk=remove_bulk,
        n_jobs=n_jobs,
    )



def build_average_kernel(
    features_list: List[np.ndarray],
    metric: str = "linear",
    weights: Optional[np.ndarray] = None,
    **kwargs
) -> np.ndarray:
    """
    Given a list of shape (num_structures,), where each item is an array
    (n_envs_in_struct, soap_dim),
    returns the NxN kernel matrix for those structures.

    The weighting logic is K_new[i, j] = K_old[i, j] * weights[i] * weights[j]

    :param features_list: List of SOAP feature arrays (ragged list of 2D arrays).
    :param metric: Similarity metric string (e.g., 'linear', 'rbf', 'polynomial').
    :param weights: Optional 1D array of weights (length N).
    :param kwargs: Additional scikit-learn parameters (e.g. gamma)
    :return: The square (N x N) kernel matrix.
    """
    avg_kernel = AverageKernel(metric=metric, **kwargs)
    K_matrix = avg_kernel.create(features_list)

    if weights is not None:
        weights = np.asarray(weights)

        # Safety Check: Dimensions must match
        if len(weights) != K_matrix.shape[0]:
            raise ValueError(
                f"Weights dimension ({len(weights)}) does not match "
                f"number of samples ({K_matrix.shape[0]})."
            )

        # K * (w @ w.T)
        scaling_factors = np.outer(weights, weights)
        K_matrix = K_matrix * scaling_factors

    return K_matrix

def build_rematch_kernel(soap_list, metric, alpha, threshold=1e-6, **kwargs):
    """
    Given a list of shape (num_structures,), where each item is an array
    (n_envs_in_struct, soap_dim),
    returns the NxN kernel matrix for those structures.

    'metric' can be "linear", "rbf", or anything supported by scikit-learn.

    Other scikit-learn parameters (e.g. gamma) are passed through **kwargs.

    :param soap_list: List of SOAP descriptors for each structure.
    :param metric: Kernel metric to use.
    :param kwargs: Additional keyword arguments for the kernel.
    :return: Kernel matrix for the structures.
    """
    avg_kern = REMatchKernel(metric=metric, alpha=alpha, threshold=threshold, **kwargs)
    # Create NxN kernel matrix
    K = avg_kern.create(soap_list)
    return K

def kernel_to_distance_matrix(K):
    """
    Convert a kernel (similarity) matrix to its kernel-induced distance.
    d(i,j) = sqrt(K[i,i] - 2K[i,j] + K[j,j])

    :param K: Kernel matrix.
    :return: Distance matrix.
    """
    diag = np.diag(K)[:, None]  # shape (N,1)
    D = np.sqrt(diag + diag.T - 2*K)
    return D

def generate_environment_soap(
        atoms_list,
        species,
        r_cut,
        n_max=2,
        l_max=3,
        sigma=1.0,
        average="off",
        center_atoms=None,
        use_neighbor_centers=False,
        n_jobs=1,
        remove_bulk=False
):
    """
    :param atoms_list: List of ASE Atoms objects.
    :param species: List of unique species in the dataset.
    :param r_cut: Cutoff radius for the SOAP descriptor.
    :param n_max: Maximum number of radial basis functions.
    :param l_max: Maximum degree of spherical harmonics.
    :param average: Averaging method for the SOAP descriptor. If average is not "off", then you'll get fewer features, but for an
    AverageKernel you usually want environment-level descriptors (so "off").
    :param center_atoms: List of atoms to center the SOAP descriptor around.
    :param use_neighbor_centers: If True, use neighbor centers for the SOAP descriptor.
    :param n_jobs: Number of parallel jobs to run.
    :param remove_bulk: Whether to remove the bulk atoms from the slab.
    :return: List of shape (num_structures,), each item is an array of shape (n_atoms_in_structure, soap_vector_dim).
    """
    soap = SOAP(
        species=species,
        periodic=True,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        average=average,
        sparse=False
    )

    soap_per_structure = []
    for idx, atoms in enumerate(atoms_list):
        if use_neighbor_centers and center_atoms:
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
        else:
            if center_atoms is not None:
                # Indices of the specified center atoms
                target_indices = [
                    i for i, symbol in enumerate(atoms.get_chemical_symbols())
                    if symbol in center_atoms
                ]
            else:
                target_indices = list(range(len(atoms)))

        if target_indices:
            features = soap.create(system=atoms, centers=target_indices, n_jobs=n_jobs)
            soap_per_structure.append(features)
        else:
            print(f"Warning: No target atoms found in structure index {idx}. Skipping.")

    if not soap_per_structure:
        raise ValueError("No SOAP features were generated. Check your center_atoms parameter.")

    return soap_per_structure

def get_soap_from_db(db_path: Path, soap_params: dict, selection: dict = None) -> tuple[list, list[int]]:
    """
    Compute SOAP descriptors for candidates stored in an ASE database.
    Atoms objects are read directly from the database, skipping any
    file-based geometry conversion.

    :param db_path: Path to the ASE database file.
    :param soap_params: Dictionary containing the SOAP parameters.
    :param selection: Optional dict of key-value pairs to filter rows
        (e.g. {'batch_index': 0}). Selects all rows if None.
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
        r_cut=soap_params['r_cut'],
        n_max=soap_params['n_max'],
        l_max=soap_params['l_max'],
        average=soap_params['average'],
        center_atoms=soap_params['center_atoms'],
        n_jobs=1,
        remove_bulk=soap_params['remove_bulk'],
    )

    if len(soap_list) != len(row_ids):
        raise ValueError(
            f"SOAP returned {len(soap_list)} descriptors but database "
            f"query returned {len(row_ids)} rows."
        )

    return soap_list, row_ids