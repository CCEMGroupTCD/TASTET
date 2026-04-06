"""Shared pipeline steps for all use cases.

Each step takes a ``cfg`` module (with the standard path helpers and
parameter dicts) and any use-case-specific data as explicit arguments.
"""

from __future__ import annotations

import json
import sys
from itertools import product

import numpy as np
import pandas as pd

from sads.soap_utils import compute_soap
from sads.kernel import compute_kernel, resolve_kernel_params
from sads.kpca import fit_kpca
from sads.io import save_soap, load_soap, save_kernel, load_kernel
from sads.distance import pairwise_dataframe, pairwise_distances
from sads.plotting import plot_kpca
from sads.plotting.heatmap import plot_grid_heatmaps
from sads.plotting.distance import plot_distance_histogram, plot_grid_histograms
from sads.sweep import run_sweep, save_results


def soap_step(cfg, atoms_list: list, **extra_soap_kw) -> None:
    """Compute and cache SOAP descriptors.

    :param cfg: Config module (needs ``db_path``, ``soap_path``, ``soap_tag``,
        ``SOAP_PARAMS``).
    :param atoms_list: Structures to featurise.
    :param extra_soap_kw: Additional SOAP keyword arguments merged on top
        of ``cfg.SOAP_PARAMS`` (e.g. ``centers=[0, 3, 5]`` for runtime-
        determined centre indices).
    """
    if not cfg.db_path().exists():
        sys.exit(f"Missing: {cfg.db_path()}.  Run 'db' step first.")
    if cfg.soap_path().exists():
        print(f"Loading SOAP   -> {cfg.soap_path()}")
        return

    params = {**cfg.SOAP_PARAMS, **extra_soap_kw}

    print(f"Computing SOAP ({cfg.soap_tag()}) for {len(atoms_list)} structures ...")
    soap_list = compute_soap(atoms_list, **params)
    save_soap(soap_list, cfg.soap_path())
    print(f"  Saved        -> {cfg.soap_path()}")


def kernel_step(cfg, ids: np.ndarray | list) -> None:
    """Compute and cache the kernel matrix from cached SOAP.

    Resolves ``gamma="median"`` via the median heuristic before building
    the kernel.  The resolved parameters are persisted alongside the
    kernel matrix so the user can inspect the actual value used.

    Also produces a distance-distribution histogram and a per-pair
    distance CSV (sorted most-dissimilar-first) for inspecting specific
    conformer pairs against the structure database.

    :param cfg: Config module (needs ``soap_path``, ``kernel_path``,
        ``kernel_meta_path``, ``KERNEL_PARAMS``).
    :param ids: Structure identifiers (e.g. ``conformer_id`` values),
        one per row of the kernel.  Used for the pairwise CSV.
    """
    if not cfg.soap_path().exists():
        sys.exit(f"Missing: {cfg.soap_path()}.  Run 'soap' step first.")
    if cfg.kernel_path().exists():
        print(f"Loading kernel -> {cfg.kernel_path()}")
        _maybe_distance_outputs(cfg, ids)
        return

    print(f"Loading SOAP   -> {cfg.soap_path()}")
    soap_list = load_soap(cfg.soap_path())

    params = resolve_kernel_params(soap_list, cfg.KERNEL_PARAMS)
    K = compute_kernel(soap_list, **params)
    save_kernel(K, cfg.kernel_path())
    print(f"  Saved        -> {cfg.kernel_path()}")

    with open(cfg.kernel_meta_path(), "w") as f:
        json.dump(params, f, indent=2)
    print(f"  Kernel meta  -> {cfg.kernel_meta_path()}")

    _save_distance_outputs(cfg, K, ids)


def _maybe_distance_outputs(cfg, ids: np.ndarray | list) -> None:
    """Generate distance outputs if they don't already exist."""
    hist_path = cfg.kernel_dir() / "distance_distribution.png"
    csv_path = cfg.kernel_dir() / "pairwise_distances.csv"
    if hist_path.exists() and csv_path.exists():
        return
    K = load_kernel(cfg.kernel_path())
    _save_distance_outputs(cfg, K, ids)


def _save_distance_outputs(
    cfg, K: np.ndarray, ids: np.ndarray | list,
) -> None:
    """Save distance histogram and pairwise CSV."""
    hist_path = cfg.kernel_dir() / "distance_distribution.png"
    if not hist_path.exists():
        plot_distance_histogram(
            K,
            title=f"Distance distribution — {cfg.kernel_tag()}",
            out_path=hist_path,
            show=getattr(cfg, "SHOW", False),
        )
        print(f"  Dist plot    -> {hist_path}")

    csv_path = cfg.kernel_dir() / "pairwise_distances.csv"
    if not csv_path.exists():
        df = pairwise_dataframe(K, ids)
        df.to_csv(csv_path, index=False)
        print(f"  Pairwise CSV -> {csv_path}  ({len(df)} pairs)")


def kpca_step(
    cfg,
    meta: pd.DataFrame,
    *,
    color_values: np.ndarray | None = None,
    color_label: str = "",
    show: bool = True,
) -> None:
    """Run kPCA, save projections + metadata, and plot.

    :param cfg: Config module (needs ``kernel_path``, ``kpca_csv_path``,
        ``kpca_meta_path``, ``plot_path``).
    :param meta: Metadata DataFrame (row order must match the kernel).
    :param color_values: Optional per-point scalar for the colorbar.
    :param color_label: Colorbar label.
    :param show: Whether to display the plot interactively.
    """
    if not cfg.kernel_path().exists():
        sys.exit(f"Missing: {cfg.kernel_path()}.  Run 'kernel' step first.")

    print(f"Loading kernel -> {cfg.kernel_path()}")
    K = load_kernel(cfg.kernel_path())
    result = fit_kpca(K, n_components=2)

    proj_df = meta.copy()
    proj_df["kpc1"] = result.projections[:, 0]
    proj_df["kpc2"] = result.projections[:, 1]
    proj_df.to_csv(cfg.kpca_csv_path(), index=False)

    kpca_meta = {"explained_variance_pct": (result.explained_variance * 100).tolist()}
    with open(cfg.kpca_meta_path(), "w") as f:
        json.dump(kpca_meta, f, indent=2)

    print(f"  Projections  -> {cfg.kpca_csv_path()}")
    print(f"  kPCA meta    -> {cfg.kpca_meta_path()}")

    plot_kpca(
        result,
        color_values=color_values,
        color_label=color_label,
        save=cfg.plot_path(),
        show=show,
    )
    print(f"  Plot         -> {cfg.plot_path()}")


def grid_search_step(
    cfg,
    atoms_list: list,
    ids: np.ndarray | list,
    scorer,
    *,
    target: np.ndarray | None = None,
    fixed_soap_kw: dict | None = None,
) -> None:
    """Sweep SOAP × kernel parameters, score, save results, and plot.

    When *target* is provided (supervised), CKA / scorer heatmaps **and**
    distance distribution histograms are generated.  When *target* is
    ``None`` (unsupervised), only distance distributions are produced.

    A per-pair distance CSV is always saved so that specific conformer
    pairs can be cross-referenced with the structure database.

    :param cfg: Config module (needs ``grid_search_*`` paths, ``SOAP_GRID``,
        ``KERNEL_GRID``).
    :param atoms_list: Structures to featurise (e.g. two reference conformers,
        or a subsampled dataset).
    :param ids: Structure identifiers, one per element of *atoms_list*.
    :param scorer: A :class:`~sads.metrics.Scorer` instance.
    :param target: Target array passed to the scorer (e.g. energies).
    :param fixed_soap_kw: Fixed SOAP kwargs (constant across sweep), e.g.
        ``center_atoms``, ``centers``, ``average``, ``n_jobs``.
    """
    tag = cfg.grid_search_tag()

    if cfg.grid_search_config_path().exists():
        print(f"Grid search [{tag}] already exists -> {cfg.grid_search_dir()}")
        print("  Change grids in config.py to run a new sweep.")
        return

    # ── 1. Score-based sweep (only if supervised) ────────────────────
    if target is not None:
        print(f"Starting grid search [{tag}] ...")
        df = run_sweep(
            atoms_list,
            target,
            soap_grid=cfg.SOAP_GRID,
            kernel_grid=cfg.KERNEL_GRID,
            scorer=scorer,
            fixed_soap_kw=fixed_soap_kw or {},
        )

        csv_path = save_results(df, cfg.grid_search_csv())
        print(f"  Results      -> {csv_path}")

        plot_grid_heatmaps(
            df,
            value=scorer.name,
            out_path=cfg.grid_search_heatmap_path(),
            suptitle=f"Grid search — {scorer.name}",
            show=cfg.SHOW,
        )
        print(f"  Heatmaps     -> {cfg.grid_search_heatmap_path()}")
    else:
        print(f"No target provided — skipping scorer heatmaps.")

    # ── 2. Distance distributions (always) ───────────────────────────
    _grid_distributions(
        cfg, atoms_list, ids,
        fixed_soap_kw=fixed_soap_kw or {},
    )

    # ── 3. Persist config snapshot ───────────────────────────────────
    # Resolve the species that were actually used (auto-inferred when
    # not set explicitly — important to record for reproducibility).
    resolved_species = sorted({
        s for a in atoms_list for s in a.get_chemical_symbols()
    })
    fixed_with_species = dict(fixed_soap_kw or {})
    fixed_with_species.setdefault("species", resolved_species)

    snapshot = {
        "soap_grid": cfg.SOAP_GRID,
        "kernel_grid": cfg.KERNEL_GRID,
        "fixed_soap_kw": fixed_with_species,
        "random_seed": cfg.SEED,
        "number_subsamples": cfg.GRID_SEARCH_N_SAMPLES,
    }
    with open(cfg.grid_search_config_path(), "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    print(f"  Config       -> {cfg.grid_search_config_path()}")


def _grid_distributions(
    cfg,
    atoms_list: list,
    ids: np.ndarray | list,
    *,
    fixed_soap_kw: dict,
) -> None:
    """Compute and plot distance distributions for every grid combination.

    Iterates over ``SOAP_GRID × KERNEL_GRID``, computes SOAP + kernel
    for each combination, and collects the normalised kernel matrix.
    Produces a multi-panel histogram figure and a combined CSV of
    per-pair distances across all combinations.
    """
    soap_grid = cfg.SOAP_GRID
    kernel_grid = cfg.KERNEL_GRID

    # Expand SOAP grid into list-of-dicts
    soap_keys = list(soap_grid.keys())
    soap_combos = [
        dict(zip(soap_keys, vals))
        for vals in product(*(soap_grid[k] for k in soap_keys))
    ] if soap_keys else [{}]

    kernel_entries: list[dict] = []
    pair_frames: list[pd.DataFrame] = []

    n_total = len(soap_combos) * len(kernel_grid)
    print(f"  Computing distance distributions ({n_total} combinations) ...")

    for s_params in soap_combos:
        soap_kw = {**fixed_soap_kw, **s_params}
        soap_list = compute_soap(atoms_list, **soap_kw)

        for k_params in kernel_grid:
            resolved = resolve_kernel_params(soap_list, dict(k_params))
            K = compute_kernel(soap_list, **resolved)

            params = {**s_params, **k_params}
            kernel_entries.append({"K": K, "params": params})

            df_pairs = pairwise_dataframe(K, ids)
            for col, val in params.items():
                df_pairs[col] = val
            pair_frames.append(df_pairs)

    # ── Plot multi-panel histograms ──────────────────────────────────
    hist_path = cfg.grid_search_dir() / "distance_distributions.png"
    plot_grid_histograms(
        kernel_entries,
        out_path=hist_path,
        suptitle="Distance distributions — grid search",
        show=getattr(cfg, "SHOW", False),
    )
    print(f"  Dist plots   -> {hist_path}")

    # ── Combined pairwise CSV ────────────────────────────────────────
    combined = pd.concat(pair_frames, ignore_index=True)
    csv_path = cfg.grid_search_dir() / "pairwise_distances.csv"
    combined.to_csv(csv_path, index=False)
    print(f"  Pairwise CSV -> {csv_path}  ({len(combined)} rows)")

    # ── Distribution summary CSV ─────────────────────────────────────
    summary_rows: list[dict] = []
    for entry in kernel_entries:
        d = pairwise_distances(entry["K"])
        row = dict(entry["params"])
        row["n_pairs"] = len(d)
        if len(d) > 0:
            row["mean"] = float(np.mean(d))
            row["median"] = float(np.median(d))
            row["std"] = float(np.std(d))
            row["min"] = float(np.min(d))
            row["max"] = float(np.max(d))
            row["range"] = float(np.max(d) - np.min(d))
        summary_rows.append(row)
    summary_path = cfg.grid_search_dir() / "distance_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"  Dist summary -> {summary_path}  ({len(summary_rows)} combinations)")


def select_step(
    cfg,
    *,
    energy_max: float | None = None,
    energy_col: str | None = None,
    color_values_col: str | None = None,
    color_label: str = "",
    show: bool = True,
) -> None:
    """Select representative structures via diverse sampling.

    :param cfg: Config module (needs ``kpca_csv_path``, ``kpca_meta_path``,
        ``kernel_path``, ``selection_csv_path``, ``selection_plot_path``,
        ``SELECTION_K``, ``SELECTION_METHOD``, ``SEED``).
    :param energy_max: Energy threshold for filtering.  *None* = no filter.
    :param energy_col: Column name for the energy filter.  Required when
        *energy_max* is set.
    :param color_values_col: Column in the projections CSV to use for
        coloring the plot.  *None* = grey scatter.
    :param color_label: Colorbar label.
    :param show: Whether to display the plot interactively.
    """
    if energy_max is not None and not energy_col:
        raise ValueError("energy_col is required when energy_max is set.")

    for path, label in [
        (cfg.kpca_csv_path(), "kpca projections"),
        (cfg.kpca_meta_path(), "kpca metadata"),
        (cfg.kernel_path(), "kernel"),
    ]:
        if not path.exists():
            sys.exit(f"Missing {label}: {path}.  Run 'kpca' step first.")

    from sads.selection import select_structures, plot_selection

    proj_df = pd.read_csv(cfg.kpca_csv_path())
    K = load_kernel(cfg.kernel_path())
    with open(cfg.kpca_meta_path()) as f:
        ev_pct = json.load(f)["explained_variance_pct"]

    selected, idx_pool = select_structures(
        K,
        proj_df,
        energy_max=energy_max,
        energy_col=energy_col,
        k=cfg.SELECTION_K,
        method=cfg.SELECTION_METHOD,
        seed=cfg.SEED,
    )

    selected.to_csv(cfg.selection_csv_path(), index=False)
    print(f"  Selected     -> {cfg.selection_csv_path()}")

    color_values = proj_df[color_values_col].values if color_values_col else None

    plot_selection(
        proj_df,
        idx_pool=idx_pool,
        selected_indices=selected["array_index"].values,
        explained_variance_pct=ev_pct,
        color_values=color_values,
        color_label=color_label,
        save_path=cfg.selection_plot_path(),
        show=show,
    )
    print(f"  Plot         -> {cfg.selection_plot_path()}")