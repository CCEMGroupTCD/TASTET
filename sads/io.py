"""Disk I/O helpers for SOAP features and kernel matrices."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_soap(soap_list: list[np.ndarray], path: Path | str) -> None:
    """Save a list of variable-length SOAP matrices as a pickled ``.npy``."""
    arr = np.empty(len(soap_list), dtype=object)
    for i, feat in enumerate(soap_list):
        arr[i] = feat
    np.save(str(path), arr, allow_pickle=True)


def load_soap(path: Path | str) -> list[np.ndarray]:
    """Load SOAP features saved with :func:`save_soap`."""
    return list(np.load(str(path), allow_pickle=True))


def save_kernel(K: np.ndarray, path: Path | str) -> None:
    """Save a kernel matrix."""
    np.save(str(path), K)


def load_kernel(path: Path | str) -> np.ndarray:
    """Load a kernel matrix."""
    return np.load(str(path))
