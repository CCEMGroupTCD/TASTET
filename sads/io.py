"""Disk I/O helpers for SOAP features and kernel matrices.

All files use ``np.savez_compressed`` (``.npz``) for significant size
reduction compared to raw ``.npy`` dumps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_soap(soap_list: list[np.ndarray], path: Path | str) -> None:
    """Save a list of SOAP feature matrices as a compressed ``.npz``.

    Each structure is stored under a key ``"0"``, ``"1"``, … so that
    variable-length arrays (different numbers of centres) are handled
    naturally.
    """
    arrays = {str(i): feat for i, feat in enumerate(soap_list)}
    np.savez_compressed(str(path), **arrays)


def load_soap(path: Path | str) -> list[np.ndarray]:
    """Load SOAP features saved with :func:`save_soap`."""
    data = np.load(str(path))
    return [data[str(i)] for i in range(len(data.files))]


def save_kernel(K: np.ndarray, path: Path | str) -> None:
    """Save a kernel matrix (compressed)."""
    np.savez_compressed(str(path), K=K)


def load_kernel(path: Path | str) -> np.ndarray:
    """Load a kernel matrix."""
    return np.load(str(path))["K"]