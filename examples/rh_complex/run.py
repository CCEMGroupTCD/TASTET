"""Conformer analysis pipeline — Rh complex.

Usage::

    python run.py db                       # 1. build master database from SDF
    python run.py grid_search              # 2. (optional) sweep on subsampled / full conformers
    python run.py soap kernel kpca         # 3. single-config pipeline on all conformers
    python run.py select                   # 4. select representatives for DFT
    python run.py help
"""

from __future__ import annotations

import json
import sys
import warnings
from itertools import product as iproduct
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sads.soap_utils import compute_soap
from sads.kernel import compute_kernel, resolve_kernel_params, combine_kernels
from sads.io import (
    load_atoms_and_meta,
    save_soap, load_soap,
    save_kernel, load_kernel,
)
from sads.distance import pairwise_dataframe, pairwise_distances
from sads.kpca import fit_kpca
from sads.pipeline import soap_step, kernel_step, kpca_step, grid_search_step, select_step
from sads.plotting import plot_kpca
from sads.plotting.distance import (
    plot_distance_histogram, plot_distance_histogram_kde, plot_grid_histograms,
)

import config as cfg

# Use-case specific
from prepare import (
    ensure_database,
    load_grid_search_structures,
    resolve_soap_centers,
    resolve_channel_soap,
)


warnings.filterwarnings("ignore", message="overflow encountered in exp", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide", category=RuntimeWarning)


# ── Cache helpers ─────────────────────────────────────────────────────

def _params_match(a: dict, b: dict) -> bool:
    """Compare two parameter dicts by their canonical JSON form.

    Using JSON canonicalisation rather than direct ``==`` so that
    bool/int collisions, dict ordering, and ``numpy`` scalars (when
    present) don't produce false negatives.

    :param a: First parameter dict.
    :param b: Second parameter dict.
    :returns: ``True`` if the two serialise to the same string.
    """
    sa = json.dumps(a, sort_keys=True, default=str)
    sb = json.dumps(b, sort_keys=True, default=str)
    return sa == sb


def _invalidate_channel_kpca(name: str) -> None:
    """Delete per-channel kPCA outputs for one channel.

    Called after a channel kernel is recomputed, so that the next
    ``kpca`` step regenerates the projection rather than serving a
    stale one from a previous kernel.

    :param name: Channel name (must match a ``KERNEL_CHANNELS`` entry).
    :returns: ``None``.
    """
    ch_dir = cfg.channel_dir(name)
    for fname in ("kpca.png", "kpca_projections.csv", "kpca_meta.json"):
        p = ch_dir / fname
        if p.exists():
            p.unlink()
            print(f"  Invalidated stale -> {p}")


def _save_combined_distance_outputs(
    cfg, K: np.ndarray, ids: np.ndarray | list,
) -> None:
    """Write the combined-kernel distance outputs, skipping existing files.

    Mirror of :func:`sads.pipeline._save_distance_outputs` for the
    multi-channel branch of :func:`_kernel`. Each output is regenerated
    only when missing, so the user can delete one file (typically
    ``kde_distance_distribution.png`` after bumping
    ``KERNEL_KDE_BANDWIDTH``) and rerun ``kernel`` to regenerate just
    that file without recomputing kernels.

    :param cfg: Config module.
    :param K: Combined kernel matrix.
    :param ids: Structure identifiers, one per kernel row.
    :returns: ``None``.
    """
    mode = getattr(cfg, "KERNEL_COMBINE", "product")

    hist_path = cfg.kernel_dir() / "distance_distribution.png"
    if not hist_path.exists():
        plot_distance_histogram(
            K,
            title=f"Distance distribution — combined ({mode})",
            out_path=hist_path,
            show=getattr(cfg, "SHOW", False),
        )
        print(f"  Dist plot    -> {hist_path}")

    kde_path = cfg.kernel_dir() / "kde_distance_distribution.png"
    if not kde_path.exists():
        plot_distance_histogram_kde(
            K,
            bandwidth=getattr(cfg, "KERNEL_KDE_BANDWIDTH", 0.02),
            title=f"Distance distribution (KDE) — combined ({mode})",
            out_path=kde_path,
            show=getattr(cfg, "SHOW", False),
        )
        print(f"  KDE plot     -> {kde_path}")

    csv_path = cfg.kernel_dir() / "pairwise_distances.csv"
    if not csv_path.exists():
        df = pairwise_dataframe(K, ids)
        df.to_csv(csv_path, index=False)
        print(f"  Pairwise CSV -> {csv_path}  ({len(df)} pairs)")


# ── Use-case wrappers ─────────────────────────────────────────────────

def _db() -> None:
    """Build the master conformer database from the configured SDF."""
    ensure_database()


def _grid_search() -> None:
    """Sweep SOAP × kernel parameters.

    Dispatches to :func:`_multichannel_grid_search` when
    ``USE_TENSOR_PRODUCT`` is set; otherwise runs the single-kernel
    grid via :func:`sads.pipeline.grid_search_step`. No scorer is
    plugged in (``target=None``), so only the distance-distribution
    outputs are produced.
    """
    if getattr(cfg, "USE_TENSOR_PRODUCT", False):
        _multichannel_grid_search()
        return

    # ── Single-kernel grid search ────────────────────────────────────
    n_soap = 1
    for vals in cfg.SOAP_GRID.values():
        n_soap *= len(vals)
    n_total = n_soap * len(cfg.KERNEL_GRID)
    max_combos = getattr(cfg, "MAX_GRID_COMBINATIONS", 500)
    if n_total > max_combos:
        sys.exit(
            f"Grid search has {n_total} combinations, exceeding "
            f"MAX_GRID_COMBINATIONS={max_combos}.  Reduce SOAP_GRID "
            f"or KERNEL_GRID."
        )

    structures, meta = load_grid_search_structures()
    centers = resolve_soap_centers(
        center_atoms=cfg.FIXED_SOAP_KW.get("center_atoms"),
    )
    fixed_kw = dict(cfg.FIXED_SOAP_KW)
    if centers is not None:
        fixed_kw["centers"] = centers

    grid_search_step(
        cfg=cfg,
        atoms_list=structures,
        ids=meta["conformer_id"].values,
        scorer=None,
        target=None,
        fixed_soap_kw=fixed_kw,
    )


def _multichannel_grid_search() -> None:
    """Sweep per-channel SOAP × kernel grids, combine, and analyse distances.

    Each channel can optionally define ``soap_grid`` (dict) and
    ``kernel_grid`` (list of dicts).  The total sweep is the cartesian
    product across all channels, subject to ``MAX_GRID_COMBINATIONS``.
    Combined kernels are formed via ``KERNEL_COMBINE`` and (when
    applicable) ``KERNEL_WEIGHTS``.
    """
    # ── Expand per-channel combinations ──────────────────────────────
    channel_options: list[list[tuple]] = []

    for ch in cfg.KERNEL_CHANNELS:
        name = ch["name"]
        base_soap = resolve_channel_soap(ch)
        base_kernel = dict(ch["kernel"])

        soap_grid = ch.get("soap_grid", {})
        kernel_grid = ch.get("kernel_grid", [base_kernel])

        soap_keys = list(soap_grid.keys())
        soap_combos = [
            dict(zip(soap_keys, vals))
            for vals in iproduct(*(soap_grid[k] for k in soap_keys))
        ] if soap_keys else [{}]

        options = []
        for s_params in soap_combos:
            merged_soap = {**base_soap, **s_params}
            for k_params in kernel_grid:
                # Record swept params with channel prefix
                swept: dict = {}
                for k, v in s_params.items():
                    swept[f"{name}__{k}"] = v
                for k, v in k_params.items():
                    swept[f"{name}__{k}"] = v
                options.append((name, merged_soap, dict(k_params), swept))
        channel_options.append(options)

    # ── Check total combinations ─────────────────────────────────────
    all_combos = list(iproduct(*channel_options))
    n_total = len(all_combos)
    max_combos = getattr(cfg, "MAX_GRID_COMBINATIONS", 500)
    if n_total > max_combos:
        per_ch = " × ".join(
            f"{ch['name']}({len(opts)})"
            for ch, opts in zip(cfg.KERNEL_CHANNELS, channel_options)
        )
        sys.exit(
            f"Multi-channel grid search has {n_total} combinations "
            f"({per_ch}), exceeding MAX_GRID_COMBINATIONS={max_combos}.  "
            f"Reduce per-channel soap_grid / kernel_grid entries."
        )

    out_dir = cfg.grid_search_dir()
    config_path = out_dir / "config.json"
    if config_path.exists():
        print(f"Grid search already exists -> {out_dir}")
        print("  Change channel grids in config.py to run a new sweep.")
        return

    structures, meta = load_grid_search_structures()
    ids = meta["conformer_id"].values

    print(f"Multi-channel grid search: {n_total} combinations ...")

    # ── Sweep ────────────────────────────────────────────────────────
    soap_cache: dict[str, list] = {}
    kernel_entries: list[dict] = []
    pair_frames: list[pd.DataFrame] = []
    mode = getattr(cfg, "KERNEL_COMBINE", "product")
    weights = getattr(cfg, "KERNEL_WEIGHTS", None)

    for combo in tqdm(all_combos, desc="Multi-channel sweep"):
        channel_Ks: list[np.ndarray] = []
        params: dict = {}

        for ch_name, soap_kw, kern_kw, swept in combo:
            cache_key = json.dumps(
                {ch_name: soap_kw}, sort_keys=True, default=str,
            )
            if cache_key not in soap_cache:
                soap_cache[cache_key] = compute_soap(structures, **soap_kw)
            soap_list = soap_cache[cache_key]

            resolved = resolve_kernel_params(soap_list, kern_kw, verbose=False)
            K_ch = compute_kernel(soap_list, **resolved, verbose=False)
            channel_Ks.append(K_ch)

            # Update swept with resolved values (e.g. gamma="median" → float)
            for k, v in resolved.items():
                swept[f"{ch_name}__{k}"] = v
            params.update(swept)

        K = combine_kernels(channel_Ks, mode=mode, weights=weights)
        kernel_entries.append({"K": K, "params": params})

        df_pairs = pairwise_dataframe(K, ids)
        for col, val in params.items():
            if isinstance(val, np.generic):
                val = val.item()
            elif isinstance(val, (np.ndarray, list, tuple)):
                val = val[0] if len(val) == 1 else str(val)
            df_pairs[col] = val
        pair_frames.append(df_pairs)

    # ── Plot distance distributions ──────────────────────────────────
    hist_path = out_dir / "distance_distributions.png"
    plot_grid_histograms(
        kernel_entries,
        out_path=hist_path,
        suptitle="Distance distributions — multi-channel grid search",
        show=getattr(cfg, "SHOW", False),
    )
    print(f"  Dist plots   -> {hist_path}")

    # ── Pairwise distances CSV ───────────────────────────────────────
    combined = pd.concat(pair_frames, ignore_index=True)
    csv_path = out_dir / "pairwise_distances.csv"
    combined.to_csv(csv_path, index=False)
    print(f"  Pairwise CSV -> {csv_path}  ({len(combined)} rows)")

    # ── Distance summary CSV ─────────────────────────────────────────
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
    summary_path = out_dir / "distance_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"  Dist summary -> {summary_path}  ({len(summary_rows)} combinations)")

    # ── Config snapshot ──────────────────────────────────────────────
    # Resolve species that were actually used (auto-inferred when not
    # set explicitly — important to record for reproducibility).
    all_species = sorted({
        s for a in structures for s in a.get_chemical_symbols()
    })
    snapshot_channels = []
    for ch in cfg.KERNEL_CHANNELS:
        soap_with_species = dict(ch["soap"])
        soap_with_species.setdefault("species", all_species)
        snapshot_channels.append({
            "name": ch["name"],
            "centers_from_smarts": ch.get("centers_from_smarts", False),
            "soap": soap_with_species,
            "kernel": ch["kernel"],
            "soap_grid": ch.get("soap_grid", {}),
            "kernel_grid": ch.get("kernel_grid", [ch["kernel"]]),
        })
    snapshot = {
        "use_tensor_product": True,
        "combine_mode": mode,
        "weights": weights,
        "channels": snapshot_channels,
        "random_seed": cfg.SEED,
        "number_subsamples": cfg.GRID_SEARCH_N_SAMPLES,
    }
    with open(config_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    print(f"  Config       -> {config_path}")


def _soap() -> None:
    """Compute and cache SOAP descriptors for the active database.

    Single-kernel mode produces one descriptor set; multi-channel mode
    produces one per entry in ``KERNEL_CHANNELS``. The per-channel
    cache is keyed by channel name only — if you change SOAP params
    in ``KERNEL_CHANNELS[i]["soap"]``, delete the corresponding
    ``channels/<name>/soap.npz`` before rerunning to avoid loading a
    stale descriptor.
    """
    atoms, _ = load_atoms_and_meta(cfg.db_path())

    if not getattr(cfg, "USE_TENSOR_PRODUCT", False):
        centers = resolve_soap_centers()
        soap_step(cfg, atoms, centers=centers)
        return

    # ── Multi-channel: one SOAP per channel ──────────────────────────
    for ch in cfg.KERNEL_CHANNELS:
        name = ch["name"]
        path = cfg.channel_soap_path(ch)
        if path.exists():
            print(f"Loading SOAP [{name}] -> {path}")
            continue

        soap_kw = resolve_channel_soap(ch)
        print(f"Computing SOAP [{name}] for {len(atoms)} structures ...")
        soap_list = compute_soap(atoms, **soap_kw)
        save_soap(soap_list, path)
        print(f"  Saved        -> {path}")


def _kernel() -> None:
    """Build the kernel matrix from cached SOAP descriptors.

    Single-kernel mode delegates to :func:`sads.pipeline.kernel_step`
    (which writes both the count-based histogram and the KDE overlay).

    Multi-channel mode loads each channel's SOAP, resolves the
    channel's kernel parameters, and compares them against the
    previously cached ``kernel_meta.json``. If the resolved parameters
    match, the cached kernel is loaded; otherwise the channel kernel
    is recomputed and the corresponding per-channel kPCA outputs are
    invalidated so the next ``kpca`` step regenerates them. The
    channels are then combined via ``KERNEL_COMBINE`` /
    ``KERNEL_WEIGHTS`` and the combined-kernel distance outputs are
    written via :func:`_save_combined_distance_outputs`, which skips
    any output that already exists. Re-running ``kernel`` after
    deleting one of the distance outputs (e.g.
    ``kde_distance_distribution.png`` after bumping
    ``KERNEL_KDE_BANDWIDTH``) regenerates just that file without
    recomputing kernels.
    """
    atoms, meta = load_atoms_and_meta(cfg.db_path())
    ids = meta["conformer_id"].values

    if not getattr(cfg, "USE_TENSOR_PRODUCT", False):
        kernel_step(cfg, ids=ids)
        return

    # ── Multi-channel: load cached combined kernel if available ──────
    if cfg.kernel_path().exists():
        print(f"Loading combined kernel -> {cfg.kernel_path()}")
        K = load_kernel(cfg.kernel_path())
        _save_combined_distance_outputs(cfg, K, ids)
        return

    # ── Multi-channel: one kernel per channel, then combine ──────────
    # Resolve species for metadata (auto-inferred when not set explicitly)
    all_species = sorted({
        s for a in atoms for s in a.get_chemical_symbols()
    })

    per_channel: list = []
    channel_meta: list[dict] = []

    for ch in cfg.KERNEL_CHANNELS:
        name = ch["name"]
        soap_p = cfg.channel_soap_path(ch)
        if not soap_p.exists():
            sys.exit(f"Missing SOAP [{name}]: {soap_p}.  Run 'soap' step first.")

        print(f"Loading SOAP [{name}] -> {soap_p}")
        soap_list = load_soap(soap_p)

        k_params = resolve_kernel_params(soap_list, ch["kernel"])
        kernel_p = cfg.channel_kernel_path(ch)
        meta_p = cfg.channel_kernel_dir(ch) / "kernel_meta.json"

        # ── Cache-or-compute decision ───────────────────────────────
        cached = None
        if kernel_p.exists() and meta_p.exists():
            with open(meta_p) as f:
                cached = json.load(f)

        if cached is not None and _params_match(cached, k_params):
            print(f"Loading kernel [{name}] -> {kernel_p}  (params match)")
            K_ch = load_kernel(kernel_p)
        else:
            if cached is not None:
                print(f"  Kernel params changed for [{name}]; recomputing ...")
            K_ch = compute_kernel(soap_list, **k_params)
            save_kernel(K_ch, kernel_p)
            print(f"  Channel kernel [{name}] -> {kernel_p}")
            with open(meta_p, "w") as f:
                json.dump(k_params, f, indent=2)
            # The previous per-channel kPCA was based on the old
            # kernel; force it to regenerate next time.
            _invalidate_channel_kpca(name)

        per_channel.append(K_ch)
        soap_with_species = dict(ch["soap"])
        soap_with_species.setdefault("species", all_species)
        channel_meta.append({"name": name, "soap": soap_with_species, "kernel": k_params})

    mode = getattr(cfg, "KERNEL_COMBINE", "product")
    weights = getattr(cfg, "KERNEL_WEIGHTS", None)
    K = combine_kernels(per_channel, mode=mode, weights=weights)
    save_kernel(K, cfg.kernel_path())
    print(f"  Combined kernel ({mode}) -> {cfg.kernel_path()}")

    with open(cfg.kernel_meta_path(), "w") as f:
        json.dump(
            {"combine_mode": mode, "weights": weights, "channels": channel_meta},
            f, indent=2, default=str,
        )
    print(f"  Kernel meta  -> {cfg.kernel_meta_path()}")

    _save_combined_distance_outputs(cfg, K, ids)


def _kpca() -> None:
    """Run kPCA in 2-D and 3-D.

    The combined kernel goes through :func:`sads.pipeline.kpca_step`,
    which fits three components, persists ``kpc1``/``kpc2``/``kpc3``
    to the projections CSV and produces both ``kpca.png`` and
    ``kpca_3d.png``. In multi-channel mode each per-channel kernel is
    also projected (2-D only — single-channel diagnostics, not used
    by ``select``).
    """
    _, meta = load_atoms_and_meta(cfg.db_path())

    if not getattr(cfg, "USE_TENSOR_PRODUCT", False):
        kpca_step(cfg, meta)
        return

    show = getattr(cfg, "SHOW", False)

    # Per-channel projections (diagnostic; 2-D is enough here)
    for ch in cfg.KERNEL_CHANNELS:
        name = ch["name"]
        k_path = cfg.channel_kernel_path(ch)
        if not k_path.exists():
            sys.exit(f"Missing channel kernel [{name}]: {k_path}.  Run 'kernel' step first.")

        ch_dir = cfg.channel_dir(name)
        plot_path = ch_dir / "kpca.png"
        csv_path = ch_dir / "kpca_projections.csv"

        if plot_path.exists() and csv_path.exists():
            print(f"kPCA [{name}] already exists -> {ch_dir}")
            continue

        print(f"Running kPCA [{name}] ...")
        K_ch = load_kernel(k_path)
        result = fit_kpca(K_ch, n_components=2)

        proj_df = meta.copy()
        proj_df["kpc1"] = result.projections[:, 0]
        proj_df["kpc2"] = result.projections[:, 1]
        proj_df.to_csv(csv_path, index=False)

        kpca_meta = {"explained_variance_pct": (result.explained_variance * 100).tolist()}
        with open(ch_dir / "kpca_meta.json", "w") as f:
            json.dump(kpca_meta, f, indent=2)

        plot_kpca(result, save=plot_path, show=show)
        print(f"  Plot         -> {plot_path}")
        print(f"  Projections  -> {csv_path}")

    # Combined kernel — 2-D + 3-D + kpc3 in CSV via kpca_step
    kpca_step(cfg, meta)


def _select() -> None:
    """Select representative conformers via diverse sampling.

    Conformers carry no target property in this use case, so no energy
    filter is applied. Writes ``selected_structures.csv`` and one
    ``.xyz`` per selected conformer; produces both 2-D and 3-D plots
    via :func:`sads.pipeline.select_step`.
    """
    select_step(cfg)


# ── CLI dispatch ──────────────────────────────────────────────────────

STEPS: dict[str, callable] = {
    "db":          _db,
    "grid_search": _grid_search,
    "soap":        _soap,
    "kernel":      _kernel,
    "kpca":        _kpca,
    "select":      _select,
}

USAGE: str = """\
Available steps:

  1.  db             Build database from SDF conformers.
  2.  grid_search    Sweep SOAP × kernel parameters.  When
                     USE_TENSOR_PRODUCT = True, sweeps per-channel grids
                     and combines; otherwise single-kernel sweep.
                     Subject to MAX_GRID_COMBINATIONS.
  3.  soap           Compute SOAP.  When USE_TENSOR_PRODUCT = True,
                     computes one SOAP per channel; otherwise single-config.
  4.  kernel         Build kernel matrix + distance histogram + KDE overlay
                     + pairwise CSV. Per-channel kernels are cached against
                     their kernel_meta.json; the combined-kernel distance
                     outputs are regenerated individually when missing.
  5.  kpca           Run kPCA, save projections + 2-D and 3-D plots.
  6.  select         Select representative conformers; write .xyz files.

  Examples:  python run.py db
             python run.py grid_search
             python run.py soap kernel kpca
             python run.py select
"""


def main() -> None:
    """Dispatch each named CLI step in order."""
    requested = sys.argv[1:]
    if not requested or requested == ["help"] or requested == ["--help"]:
        print(USAGE)
        return
    for name in requested:
        if name not in STEPS:
            print(USAGE)
            sys.exit(f"Unknown step: {name!r}")
        print(f"\n{'='*60}\n  Step: {name}\n{'='*60}")
        STEPS[name]()


if __name__ == "__main__":
    main()