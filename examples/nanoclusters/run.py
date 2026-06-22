"""Cu clusters on surface — analysis pipeline.

Usage::

    python run.py db                       # 1. build database from the runs
    python run.py grid_search              # 2. (optional) sweep on energy-balanced subset
    python run.py soap kernel kpca         # 3. single-config pipeline on all structures
    python run.py select                   # 4. select structures for DFT
    python run.py help
"""

from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import pandas as pd

from tastet.soap_utils import compute_soap
from tastet.kernel import compute_kernel, resolve_kernel_params, combine_kernels
from tastet.io import (
    load_atoms_and_meta,
    save_soap,
    load_soap,
    save_kernel,
    load_kernel,
)
from tastet.distance import pairwise_dataframe
from tastet.kpca import fit_kpca
from tastet.pipeline import (
    soap_step,
    kernel_step,
    kpca_step,
    grid_search_step,
    select_step,
)
from tastet.sweep.multichannel import grid_search_multichannel_step
from tastet.plotting import plot_kpca
from tastet.plotting.distance import (
    plot_distance_histogram,
    plot_distance_histogram_kde,
)
from tastet.metrics.cka import CKAScorer
from tastet.selection import plot_selection

from ase.db import connect
from ase.io import write

import config as cfg

# Use-case specific
from prepare import ensure_database, load_grid_search_structures, resolve_channel_soap


warnings.filterwarnings(
    "ignore", message="overflow encountered in exp", category=RuntimeWarning
)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in scalar divide",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered in scalar divide",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore", message="divide by zero encountered in divide", category=RuntimeWarning
)


# Energy-coloured plots use ``E - E_gm`` (relative to the study minimum),
# matching the y-axis of analysis/energy_profile.py.
ENERGY_LABEL: str = r"$E - E_{\mathrm{gm}}$ (eV)"


def _relative_energy(meta: pd.DataFrame) -> np.ndarray:
    """Surrogate energies shifted to the global minimum ``E_gm``.

    Returns ``E - E_gm`` where ``E`` is the raw potential energy of each
    structure (the ``energy_eV`` column written by
    :func:`prepare._build_database`) and ``E_gm`` is the lowest such
    energy across the whole set. ``E_gm`` is the *only* reference used —
    energies are not referenced to any bulk/surface reservoir. Every
    energy-coloured plot uses this so they share one zero.

    These are the surrogate energies of the full 10k set — *not* the DFT
    energies of the selected structures, which live in
    ``cfg.ENERGIES_SELECTED_CSV`` and are used only by
    ``analysis/energy_profile.py``. The two are never mixed here.

    :param meta: Metadata for the full set (must hold ``energy_eV``).
    :returns: ``energy_eV - energy_eV.min()`` in eV.
    """
    e = meta["energy_eV"].values
    return e - e.min()


# ── Combined-kernel output helper ────────────────────────────────────


def _save_combined_distance_outputs(
    cfg,
    K: np.ndarray,
    ids: np.ndarray | list,
) -> None:
    """Write the combined-kernel distance outputs, skipping existing files.

    Mirror of :func:`tastet.pipeline._save_distance_outputs` for the
    multi-channel branch of :func:`_kernel`. Each output is regenerated
    only when missing, so deleting one file (e.g.
    ``kde_distance_distribution.png`` after bumping
    ``KERNEL_KDE_BANDWIDTH``) and rerunning ``kernel`` regenerates just
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
    """Build / update the database from the configured per-run trajectories."""
    ensure_database()


def _grid_search() -> None:
    """Sweep SOAP × kernel parameters, scored by CKA against formation energy.

    Dispatches by ``USE_TENSOR_PRODUCT``: multi-channel sweep via
    :func:`tastet.sweep.multichannel.grid_search_multichannel_step` or
    single-kernel sweep via :func:`tastet.pipeline.grid_search_step`,
    both scored by :class:`~tastet.metrics.cka.CKAScorer` against the
    formation energies.

    The grid search runs on the energy-balanced subset returned by
    :func:`prepare.load_grid_search_structures`, not the full database.
    """
    atoms, meta = load_grid_search_structures()
    ids = meta["configuration_id"].values
    scorer = CKAScorer(target_kernel=cfg.CKA_TARGET_KERNEL)
    target = meta["energy_eV"].values

    if getattr(cfg, "USE_TENSOR_PRODUCT", False):
        grid_search_multichannel_step(
            cfg=cfg,
            atoms_list=atoms,
            ids=ids,
            channels=cfg.KERNEL_CHANNELS,
            scorer=scorer,
            target=target,
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

    grid_search_step(
        cfg=cfg,
        atoms_list=atoms,
        ids=ids,
        scorer=scorer,
        target=target,
        fixed_soap_kw=cfg.FIXED_SOAP_KW,
    )


def _soap() -> None:
    """Compute and cache SOAP descriptors for the active database.

    Single-kernel mode produces one descriptor set keyed by
    :func:`config.soap_tag`. Multi-channel mode produces one per entry
    in ``KERNEL_CHANNELS``, each cached under a hash-keyed path so that
    changing a channel's SOAP parameters yields a new cache directory
    rather than overwriting the old one.
    """
    atoms, _ = load_atoms_and_meta(cfg.db_path())

    if not getattr(cfg, "USE_TENSOR_PRODUCT", False):
        soap_step(cfg, atoms)
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

    Single-kernel mode delegates to :func:`tastet.pipeline.kernel_step`.
    Multi-channel mode looks up each channel's SOAP and kernel at their
    hash-keyed cache paths (a cache hit is a plain path-existence
    check), combines them via ``KERNEL_COMBINE`` / ``KERNEL_WEIGHTS``,
    and writes the combined-kernel distance outputs via
    :func:`_save_combined_distance_outputs`.
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
    all_species = sorted({s for a in atoms for s in a.get_chemical_symbols()})

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
        channel_meta.append(
            {"name": name, "soap": soap_with_species, "kernel": k_params}
        )

    mode = getattr(cfg, "KERNEL_COMBINE", "product")
    weights = getattr(cfg, "KERNEL_WEIGHTS", None)
    K = combine_kernels(per_channel, mode=mode, weights=weights)
    save_kernel(K, cfg.kernel_path())
    print(f"  Combined kernel ({mode}) -> {cfg.kernel_path()}")

    with open(cfg.kernel_meta_path(), "w") as f:
        json.dump(
            {"combine_mode": mode, "weights": weights, "channels": channel_meta},
            f,
            indent=2,
            default=str,
        )
    print(f"  Kernel meta  -> {cfg.kernel_meta_path()}")

    _save_combined_distance_outputs(cfg, K, ids)


def _kpca() -> None:
    """Run kPCA, colouring projections by ``E - E_gm`` (surrogate energy).

    Colours use :func:`_relative_energy` (surrogate energies shifted to
    the study minimum ``E_gm``), so the scale matches
    ``analysis/energy_profile.py``.

    The combined kernel goes through :func:`tastet.pipeline.kpca_step`
    (2-D + 3-D). In multi-channel mode each per-channel kernel is also
    projected (2-D), with outputs living inside the channel's hash-keyed
    kernel directory.
    """
    _, meta = load_atoms_and_meta(cfg.db_path())
    color_values = _relative_energy(meta)

    if not getattr(cfg, "USE_TENSOR_PRODUCT", False):
        kpca_step(
            cfg,
            meta,
            color_values=color_values,
            color_label=ENERGY_LABEL,
        )
        return

    show = getattr(cfg, "SHOW", False)

    # Per-channel projections (diagnostic; 2-D is enough here)
    for ch in cfg.KERNEL_CHANNELS:
        name = ch["name"]
        k_path = cfg.channel_kernel_path(ch)
        if not k_path.exists():
            sys.exit(
                f"Missing channel kernel [{name}]: {k_path}.  Run 'kernel' step first."
            )

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

        kpca_meta = {
            "explained_variance_pct": (result.explained_variance * 100).tolist()
        }
        with open(ch_kdir / "kpca_meta.json", "w") as f:
            json.dump(kpca_meta, f, indent=2)

        plot_kpca(
            result,
            color_values=color_values,
            color_label=ENERGY_LABEL,
            save=plot_path,
            show=show,
        )
        print(f"  Plot         -> {plot_path}")
        print(f"  Projections  -> {csv_path}")

    # Combined kernel — 2-D + 3-D + kpc3 in CSV via kpca_step
    kpca_step(
        cfg,
        meta,
        color_values=color_values,
        color_label=ENERGY_LABEL,
    )


def _select() -> None:
    """Select representatives below the energy threshold, with full-view + POSCARs.

    Applies the formation-energy filter, runs diverse selection
    (``selection.png`` over the filtered pool, plus 2-D/3-D plots via
    :func:`tastet.pipeline.select_step`), then adds a full-view kPCA plot
    with the selections highlighted and exports VASP POSCARs.
    """
    _, meta = load_atoms_and_meta(cfg.db_path())
    show = getattr(cfg, "SHOW", False)
    color_values = _relative_energy(meta)

    # ── 1. Run selection (filtered pool + selection.png) ─────────────
    # Both the filter and the colour scale are on E - E_gm: the
    # threshold is relative (energy_relative=True), and the colour is the
    # pre-shifted surrogate energy.
    select_step(
        cfg,
        energy_max=cfg.SELECTION_ENERGY_MAX,
        energy_col="energy_eV",
        energy_relative=True,
        color_values=color_values,
        color_label=ENERGY_LABEL,
    )

    # ── 2. Full-view kPCA with selections highlighted ────────────────
    full_plot_path = cfg.selection_dir() / "selection_full.png"
    if not full_plot_path.exists():
        proj_df = pd.read_csv(cfg.kpca_csv_path())
        selected = pd.read_csv(cfg.selection_csv_path())
        with open(cfg.kpca_meta_path()) as f:
            ev_pct = json.load(f)["explained_variance_pct"]

        # configuration_id is 1-based and gap-free; row position in the
        # projections (and the kernel matrix) is configuration_id - 1.
        selected_indices = selected["configuration_id"].values - 1

        plot_selection(
            proj_df,
            idx_pool=np.arange(len(proj_df)),
            selected_indices=selected_indices,
            explained_variance_pct=ev_pct,
            color_values=color_values,
            color_label=ENERGY_LABEL,
            save_path=full_plot_path,
            show=show,
            also_pdf=True,
        )
        print(f"  Full plot    -> {full_plot_path}")

    # ── 3. Export POSCARs ────────────────────────────────────────────
    poscar_dir = cfg.selection_dir() / "poscars"
    if not poscar_dir.exists():
        selected = pd.read_csv(cfg.selection_csv_path())
        db = connect(str(cfg.db_path()))
        poscar_dir.mkdir(parents=True, exist_ok=True)

        for _, row in selected.iterrows():
            cid = int(row["configuration_id"])
            atoms = db.get(configuration_id=cid).toatoms()
            fname = f"POSCAR_{cid:05d}"
            write(str(poscar_dir / fname), atoms, format="vasp")

        print(f"  POSCARs      -> {poscar_dir}  ({len(selected)} files)")


# ── CLI dispatch ──────────────────────────────────────────────────────

STEPS: dict[str, callable] = {
    "db": _db,
    "grid_search": _grid_search,
    "soap": _soap,
    "kernel": _kernel,
    "kpca": _kpca,
    "select": _select,
}

USAGE: str = """\
Available steps:

  1.  db             Build / update the database from the per-run trajectories.
  2.  grid_search    Sweep SOAP × kernel parameters on an energy-balanced
                     subset of the database.  When USE_TENSOR_PRODUCT = True,
                     sweeps per-channel grids and combines; otherwise
                     single-kernel sweep with CKA.  Subject to
                     MAX_GRID_COMBINATIONS.
  3.  soap           Compute SOAP.  When USE_TENSOR_PRODUCT = True,
                     computes one SOAP per channel (hash-keyed path).
  4.  kernel         Build kernel matrix + distance histogram + KDE overlay
                     + pairwise CSV.  When USE_TENSOR_PRODUCT = True,
                     computes per-channel kernels and combines them.
  5.  kpca           Run kPCA (coloured by formation energy), save
                     projections + 2-D and 3-D plots.
  6.  select         Select representative structures for DFT.
                     Produces selection.png (filtered pool),
                     selection_full.png (full kPCA), and POSCARs.

  Examples:  python run.py db
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
        print(f"\n{'=' * 60}\n  Step: {name}\n{'=' * 60}")
        STEPS[name]()


if __name__ == "__main__":
    main()
