"""Generic SOAP × kernel parameter sweep.

Computes SOAP once per parameter set, then evaluates every kernel
configuration against it.  Scoring is delegated to a pluggable
:class:`~tastet.metrics.Scorer`.
"""

from __future__ import annotations

import traceback
import itertools
import warnings
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from tastet.soap_utils import compute_soap
from tastet.kernel import compute_kernel, resolve_kernel_params
from tastet.metrics import Scorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_grid(param_grid: dict[str, list]) -> list[dict[str, Any]]:
    """Cartesian product of a parameter grid → list of flat dicts."""
    keys = list(param_grid.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*param_grid.values())]


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(
    atoms_list: Sequence,
    target: np.ndarray | None,
    soap_grid: dict[str, list],
    kernel_grid: list[dict[str, Any]],
    scorer: Scorer,
    *,
    fixed_soap_kw: dict | None = None,
    soap_fn: Callable | None = None,
    normalize_kernel: bool = True,
) -> pd.DataFrame:
    """Sweep all SOAP × kernel combinations and score each one with ``scorer``.

    :param atoms_list: Structures to featurize.
    :param target: Reference values passed through to the scorer, such as energies.
        Ignored by scorers that do not need it.
    :param soap_grid: Mapping of SOAP keyword argument names to lists of values to
        sweep. Example: ``{"r_cut": [3.0, 5.0], "sigma": [0.1, 0.5]}``.
    :param kernel_grid: Mapping of :func:`~tastet.kernel.compute_kernel` keyword
        argument names to lists of values to sweep. Example:
        ``[{"method": "rematch", "alpha": 0.1}, {"method": "rematch", "alpha": 0.5}]``.
    :param scorer: Callable with signature ``(K, target) -> float | None``.
    :param fixed_soap_kw: SOAP keyword arguments kept constant across the sweep,
        such as ``species`` or ``center_atoms``.
    :param soap_fn: Drop-in replacement for :func:`~tastet.soap_utils.compute_soap`.
        It must accept ``(atoms_list, **kwargs)`` and return a list of ndarrays.
    :param normalize_kernel: Whether to normalize the kernel before scoring. Passed
        to :func:`~tastet.kernel.compute_kernel`.
    :returns: DataFrame with one row per parameter combination. Columns include
        every swept parameter, ``scorer.name`` for the score, and ``"status"`` with
        one of ``"OK"``, ``"SOAP_FAIL"``, ``"KERNEL_FAIL"``, ``"OVERFLOW"``, or
        ``"SCORE_FAIL"``. When ``gamma="median"`` is resolved, the ``gamma`` column
        contains the numeric value that was actually used.
    :rtype: pandas.DataFrame
    """
    if soap_fn is None:
        soap_fn = compute_soap
    fixed_soap_kw = fixed_soap_kw or {}

    soap_combos = _generate_grid(soap_grid)
    kernel_combos = kernel_grid
    total = len(soap_combos) * len(kernel_combos)

    rows: list[dict[str, Any]] = []

    with tqdm(total=total, desc="Sweep") as pbar:
        for soap_kw in soap_combos:
            merged = {**fixed_soap_kw, **soap_kw}

            # -- SOAP --------------------------------------------------
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    soap_list = soap_fn(list(atoms_list), **merged)
                soap_ok = True
            except Exception as e:
                print("SOAP failed:", type(e).__name__, e)
                traceback.print_exc()
                soap_ok = False

            # -- Kernel + Score ----------------------------------------
            for kern_kw in kernel_combos:
                # Resolve gamma="median" → float *before* compute_kernel
                resolved_kw = resolve_kernel_params(
                    soap_list if soap_ok else [],
                    kern_kw,
                    verbose=False,
                ) if soap_ok else kern_kw

                # Record the resolved values (float, not "median")
                row: dict[str, Any] = {**soap_kw, **resolved_kw}

                if not soap_ok:
                    row[scorer.name] = None
                    row["status"] = "SOAP_FAIL"
                    rows.append(row)
                    pbar.update(1)
                    continue

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        K = compute_kernel(
                            soap_list,
                            verbose=False,
                            normalize=normalize_kernel,
                            **resolved_kw,
                        )
                except Exception:
                    row[scorer.name] = None
                    row["status"] = "KERNEL_FAIL"
                    rows.append(row)
                    pbar.update(1)
                    continue

                if not np.all(np.isfinite(K)):
                    row[scorer.name] = None
                    row["status"] = "OVERFLOW"
                    rows.append(row)
                    pbar.update(1)
                    continue

                score = scorer(K, target)
                row[scorer.name] = score
                row["status"] = "OK" if score is not None else "SCORE_FAIL"
                rows.append(row)
                pbar.update(1)

    return pd.DataFrame(rows)