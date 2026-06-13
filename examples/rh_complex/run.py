"""Conformer analysis pipeline — Rh complex.

Usage::

    python run.py db                       # 1. build master database from SDF
    python run.py grid_search              # 2. (optional) sweep on subsampled / full conformers
    python run.py soap kernel kpca         # 3. single-config pipeline on all conformers
    python run.py select                   # 4. select representatives for DFT
    python run.py help
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

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
from sads.sweep.multichannel import grid_search_multichannel_step
from sads.plotting import plot_kpca
from sads.plotting.distance import (
    plot_distance_histogram, plot_distance_histogram_kde,
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


# ── Combined-kernel output helper ────────────────────────────────────

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

    Dispatches by ``USE_TENSOR_PRODUCT``: multi-channel sweep via
    :func:`sads.sweep.multichannel.grid_search_multichannel_step`, or
    single-kernel sweep via :func:`sads.pipeline.grid_search_step`.
    No scorer is plugged in (``target=None``), so only the
    distance-distribution outputs are produced.
    """
    structures, meta = load_grid_search_structures()
    ids = meta["configuration_id"].values

    if getattr(cfg, "USE_TENSOR_PRODUCT", False):
        grid_search_multichannel_step(
            cfg=cfg,
            atoms_list=structures,
            ids=ids,
            channels=cfg.KERNEL_CHANNELS,
            resolve_channel_soap=resolve_channel_soap,
        )
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

    centers = resolve_soap_centers(
        center_atoms=cfg.FIXED_SOAP_KW.get("center_atoms"),
    )
    fixed_kw = dict(cfg.FIXED_SOAP_KW)
    if centers is not None:
        fixed_kw["centers"] = centers

    grid_search_step(
        cfg=cfg,
        atoms_list=structures,
        ids=ids,
        scorer=None,
        target=None,
        fixed_soap_kw=fixed_kw,
    )


def _soap() -> None:
    """Compute and cache SOAP descriptors for the active database.

    Single-kernel mode produces one descriptor set keyed by
    :func:`config.soap_tag`. Multi-channel mode produces one per entry
    in ``KERNEL_CHANNELS``, each cached under a hash-keyed path
    (``channels/<name>/<soap_tag>/soap.npz``) so that changing a
    channel's SOAP parameters yields a new cache directory rather than
    overwriting the old one.
    """
    atoms, _ = load_atoms_and_meta(cfg.db_path())

    if not getattr(cfg, "USE_TENSOR_PRODUCT", False):
        centers = resolve_soap_centers()
        soap_step(cfg, atoms, centers=centers)
        return

    # ── Multi-channel: one SOAP per channel (hash-keyed path) ────────
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

    Multi-channel mode looks up each channel's SOAP and kernel at their
    hash-keyed cache paths. A kernel cache hit is a plain path-existence
    check — different parameter sets land in different directories by
    construction, so swapping a channel's parameters back and forth
    (e.g. ``alpha=0.5`` ↔ ``alpha=0.1``) reuses both caches for free.
    The channels are combined via ``KERNEL_COMBINE`` / ``KERNEL_WEIGHTS``
    and the combined-kernel distance outputs are written via
    :func:`_save_combined_distance_outputs`, which skips outputs that
    already exist.
    """
    atoms, meta = load_atoms_and_meta(cfg.db_path())
    ids = meta["configuration_id"].values

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

        kernel_p = cfg.channel_kernel_path(ch)
        meta_p = cfg.channel_kernel_dir(ch) / "kernel_meta.json"

        if kernel_p.exists():
            print(f"Loading kernel [{name}] -> {kernel_p}")
            K_ch = load_kernel(kernel_p)
            if meta_p.exists():
                with open(meta_p) as f:
                    k_params = json.load(f)
            else:
                # Kernel present but meta missing: re-resolve to populate
                # channel_meta below, and write the meta for next time.
                k_params = resolve_kernel_params(soap_list, ch["kernel"])
                with open(meta_p, "w") as f:
                    json.dump(k_params, f, indent=2)
        else:
            k_params = resolve_kernel_params(soap_list, ch["kernel"])
            K_ch = compute_kernel(soap_list, **k_params)
            save_kernel(K_ch, kernel_p)
            print(f"  Channel kernel [{name}] -> {kernel_p}")
            with open(meta_p, "w") as f:
                json.dump(k_params, f, indent=2)

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
    also projected (2-D only — single-channel diagnostics, not used by
    ``select``). Per-channel kPCA outputs live inside the same
    hash-keyed directory as their kernel, so a kernel cached at one
    parameter set never inherits a kPCA computed from a different one.
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

        ch_kdir = cfg.channel_kernel_dir(ch)
        plot_path = ch_kdir / "kpca.png"
        csv_path = ch_kdir / "kpca_projections.csv"

        if plot_path.exists() and csv_path.exists():
            print(f"kPCA [{name}] already exists -> {ch_kdir}")
            continue

        print(f"Running kPCA [{name}] ...")
        K_ch = load_kernel(k_path)
        result = fit_kpca(K_ch, n_components=2)

        proj_df = meta.copy()
        proj_df["kpc1"] = result.projections[:, 0]
        proj_df["kpc2"] = result.projections[:, 1]
        proj_df.to_csv(csv_path, index=False)

        kpca_meta = {"explained_variance_pct": (result.explained_variance * 100).tolist()}
        with open(ch_kdir / "kpca_meta.json", "w") as f:
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
                     computes one SOAP per channel (hash-keyed path).
  4.  kernel         Build kernel matrix + distance histogram + KDE overlay
                     + pairwise CSV. Per-channel kernels are cached at
                     hash-keyed paths nested inside the channel's SOAP
                     directory, so swapping kernel parameters is free.
  5.  kpca           Run kPCA, save projections + 2-D and 3-D plots.
                     Per-channel kPCA outputs live next to the channel's
                     kernel and inherit its hash key.
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