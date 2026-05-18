"""Cu clusters on surface — analysis pipeline.

Usage::

    python run.py db                       # 1. build master database
    python run.py subsample                # 2. (optional) create subset
    python run.py grid_search              # 3. (optional) sweep hyperparameters
    python run.py soap kernel kpca         # 4. single-config pipeline
    python run.py select                   # 5. select structures for DFT
    python run.py help
"""

from __future__ import annotations

import json
import sys
import warnings
from itertools import product as iproduct

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
from sads.plotting.distance import plot_distance_histogram, plot_grid_histograms
from sads.metrics.cka import CKAScorer
from sads.selection import plot_selection

from ase.db import connect
from ase.io import write

import config as cfg

# Use-case specific
from prepare import ensure_database, subsample, resolve_channel_soap


warnings.filterwarnings("ignore", message="overflow encountered in exp", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="divide by zero encountered in divide", category=RuntimeWarning)


# ── Use-case wrappers ─────────────────────────────────────────────────

def _db() -> None:
    ensure_database()


def _subsample() -> None:
    subsample()


def _grid_search() -> None:
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

    atoms, meta = load_atoms_and_meta(cfg.db_path())
    grid_search_step(
        cfg=cfg,
        atoms_list=atoms,
        ids=meta["structure_id"].values,
        scorer=CKAScorer(target_kernel=cfg.CKA_TARGET_KERNEL),
        target=meta["formation_energy"].values,
        fixed_soap_kw=cfg.FIXED_SOAP_KW,
    )


def _multichannel_grid_search() -> None:
    """Sweep per-channel SOAP × kernel grids, combine, and analyse distances.

    Each channel can optionally define ``soap_grid`` (dict) and
    ``kernel_grid`` (list of dicts).  The total sweep is the cartesian
    product across all channels, subject to ``MAX_GRID_COMBINATIONS``.
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

    atoms, meta = load_atoms_and_meta(cfg.db_path())
    ids = meta["structure_id"].values

    print(f"Multi-channel grid search: {n_total} combinations ...")

    # ── Sweep ────────────────────────────────────────────────────────
    soap_cache: dict[str, list] = {}
    kernel_entries: list[dict] = []
    pair_frames: list[pd.DataFrame] = []
    mode = getattr(cfg, "KERNEL_COMBINE", "product")

    for combo in tqdm(all_combos, desc="Multi-channel sweep"):
        channel_Ks: list[np.ndarray] = []
        params: dict = {}

        for ch_name, soap_kw, kern_kw, swept in combo:
            cache_key = json.dumps(
                {ch_name: soap_kw}, sort_keys=True, default=str,
            )
            if cache_key not in soap_cache:
                soap_cache[cache_key] = compute_soap(atoms, **soap_kw)
            soap_list = soap_cache[cache_key]

            resolved = resolve_kernel_params(soap_list, kern_kw, verbose=False)
            K_ch = compute_kernel(soap_list, **resolved, verbose=False)
            channel_Ks.append(K_ch)

            # Update swept with resolved values (e.g. gamma="median" → float)
            for k, v in resolved.items():
                swept[f"{ch_name}__{k}"] = v
            params.update(swept)

        K = combine_kernels(channel_Ks, mode=mode)
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
    all_species = sorted({
        s for a in atoms for s in a.get_chemical_symbols()
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
        "channels": snapshot_channels,
        "random_seed": cfg.SEED,
    }
    with open(config_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    print(f"  Config       -> {config_path}")


def _soap() -> None:
    atoms, _ = load_atoms_and_meta(cfg.db_path())

    if not getattr(cfg, "USE_TENSOR_PRODUCT", False):
        soap_step(cfg, atoms)
        return

    # ── Multi-channel: one SOAP per channel ──────────────────────────
    for ch in cfg.KERNEL_CHANNELS:
        name = ch["name"]
        path = cfg.channel_soap_path(name)
        if path.exists():
            print(f"Loading SOAP [{name}] -> {path}")
            continue

        soap_kw = resolve_channel_soap(ch)
        print(f"Computing SOAP [{name}] for {len(atoms)} structures ...")
        soap_list = compute_soap(atoms, **soap_kw)
        save_soap(soap_list, path)
        print(f"  Saved        -> {path}")


def _kernel() -> None:
    atoms, meta = load_atoms_and_meta(cfg.db_path())

    if not getattr(cfg, "USE_TENSOR_PRODUCT", False):
        kernel_step(cfg, ids=meta["structure_id"].values)
        return

    # ── Multi-channel: one kernel per channel, then combine ──────────
    if cfg.kernel_path().exists():
        print(f"Loading combined kernel -> {cfg.kernel_path()}")
        return

    # Resolve species for metadata
    all_species = sorted({
        s for a in atoms for s in a.get_chemical_symbols()
    })

    per_channel: list = []
    channel_meta: list[dict] = []

    for ch in cfg.KERNEL_CHANNELS:
        name = ch["name"]
        soap_p = cfg.channel_soap_path(name)
        if not soap_p.exists():
            sys.exit(f"Missing SOAP [{name}]: {soap_p}.  Run 'soap' step first.")

        print(f"Loading SOAP [{name}] -> {soap_p}")
        soap_list = load_soap(soap_p)

        k_params = resolve_kernel_params(soap_list, ch["kernel"])
        K_ch = compute_kernel(soap_list, **k_params)

        save_kernel(K_ch, cfg.channel_kernel_path(name))
        print(f"  Channel kernel [{name}] -> {cfg.channel_kernel_path(name)}")

        with open(cfg.channel_dir(name) / "kernel_meta.json", "w") as f:
            json.dump(k_params, f, indent=2)

        per_channel.append(K_ch)
        soap_with_species = dict(ch["soap"])
        soap_with_species.setdefault("species", all_species)
        channel_meta.append({"name": name, "soap": soap_with_species, "kernel": k_params})

    mode = getattr(cfg, "KERNEL_COMBINE", "product")
    K = combine_kernels(per_channel, mode=mode)
    save_kernel(K, cfg.kernel_path())
    print(f"  Combined kernel ({mode}) -> {cfg.kernel_path()}")

    with open(cfg.kernel_meta_path(), "w") as f:
        json.dump({"combine_mode": mode, "channels": channel_meta}, f, indent=2, default=str)
    print(f"  Kernel meta  -> {cfg.kernel_meta_path()}")

    # ── Distance outputs ─────────────────────────────────────────────
    ids = meta["structure_id"].values

    hist_path = cfg.kernel_dir() / "distance_distribution.png"
    plot_distance_histogram(
        K,
        title=f"Distance distribution — combined ({mode})",
        out_path=hist_path,
        show=getattr(cfg, "SHOW", False),
    )
    print(f"  Dist plot    -> {hist_path}")

    csv_path = cfg.kernel_dir() / "pairwise_distances.csv"
    df = pairwise_dataframe(K, ids)
    df.to_csv(csv_path, index=False)
    print(f"  Pairwise CSV -> {csv_path}  ({len(df)} pairs)")


def _kpca() -> None:
    _, meta = load_atoms_and_meta(cfg.db_path())

    if not getattr(cfg, "USE_TENSOR_PRODUCT", False):
        kpca_step(
            cfg, meta,
            color_values=meta["formation_energy"].values,
            color_label="Formation energy (eV)",
        )
        return

    # ── Multi-channel: per-channel + combined kPCA ───────────────────
    show = getattr(cfg, "SHOW", False)
    color_values = meta["formation_energy"].values

    # Per-channel projections
    for ch in cfg.KERNEL_CHANNELS:
        name = ch["name"]
        k_path = cfg.channel_kernel_path(name)
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

        plot_kpca(
            result,
            color_values=color_values,
            color_label="Formation energy (eV)",
            save=plot_path, show=show,
        )
        print(f"  Plot         -> {plot_path}")
        print(f"  Projections  -> {csv_path}")

    # Combined kernel
    kpca_step(
        cfg, meta,
        color_values=color_values,
        color_label="Formation energy (eV)",
    )


def _select() -> None:
    # ── 1. Run selection (filtered pool + selection.png) ─────────────
    select_step(
        cfg,
        energy_max=cfg.SELECTION_ENERGY_MAX,
        energy_col="formation_energy",
        color_values_col="formation_energy",
        color_label="Formation energy (eV)",
    )

    _, meta = load_atoms_and_meta(cfg.db_path())
    show = getattr(cfg, "SHOW", False)

    # ── 2. Full-view kPCA with selections highlighted ────────────────
    full_plot_path = cfg.selection_dir() / "selection_full.png"
    if not full_plot_path.exists():
        proj_df = pd.read_csv(cfg.kpca_csv_path())
        selected = pd.read_csv(cfg.selection_csv_path())
        with open(cfg.kpca_meta_path()) as f:
            ev_pct = json.load(f)["explained_variance_pct"]

        plot_selection(
            proj_df,
            idx_pool=np.arange(len(proj_df)),
            selected_indices=selected["array_index"].values,
            explained_variance_pct=ev_pct,
            color_values=meta["formation_energy"].values,
            color_label="Formation energy (eV)",
            save_path=full_plot_path,
            show=show,
        )
        print(f"  Full plot    -> {full_plot_path}")

    # ── 3. Export POSCARs ────────────────────────────────────────────
    poscar_dir = cfg.selection_dir() / "poscars"
    if not poscar_dir.exists():
        selected = pd.read_csv(cfg.selection_csv_path())
        db = connect(str(cfg.db_path()))
        poscar_dir.mkdir(parents=True, exist_ok=True)

        for _, row in selected.iterrows():
            sid = int(row["structure_id"])
            atoms = db.get(structure_id=sid).toatoms()
            fname = f"POSCAR_{sid:05d}"
            write(str(poscar_dir / fname), atoms, format="vasp")

        print(f"  POSCARs      -> {poscar_dir}  ({len(selected)} files)")


# ── CLI dispatch ──────────────────────────────────────────────────────

STEPS: dict[str, callable] = {
    "db":          _db,
    "subsample":   _subsample,
    "grid_search": _grid_search,
    "soap":        _soap,
    "kernel":      _kernel,
    "kpca":        _kpca,
    "select":      _select,
}

USAGE: str = """\
Available steps:

  1.  db             Build / update the master database.
  2.  subsample      (Optional) Create an energy-balanced subset.
  3.  grid_search    Sweep SOAP × kernel parameters.  When
                     USE_TENSOR_PRODUCT = True, sweeps per-channel grids
                     and combines; otherwise single-kernel sweep with CKA.
                     Subject to MAX_GRID_COMBINATIONS.
  4.  soap           Compute SOAP.  When USE_TENSOR_PRODUCT = True,
                     computes one SOAP per channel; otherwise single-config.
  5.  kernel         Build kernel matrix + distance histogram + pairwise CSV.
                     When USE_TENSOR_PRODUCT = True, computes per-channel
                     kernels and combines them (product or sum).
  6.  kpca           Run kPCA, save projections + plot.
  7.  select         Select representative structures for DFT.
                     Produces selection.png (filtered pool),
                     selection_full.png (full kPCA), and POSCARs.

  Examples:  python run.py db
             python run.py soap kernel kpca
             python run.py select
"""


def main() -> None:
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