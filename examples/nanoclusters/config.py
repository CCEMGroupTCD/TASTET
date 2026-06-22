"""Configuration for the Cu-cluster-on-surface analysis.

General settings (top) — present in every use case.
Use-case-specific settings (middle) — unique to this Cu-cluster system.
Path helpers (bottom) — reusable output tree.
"""

from pathlib import Path


# =====================================================================
#  GENERAL SETTINGS
# =====================================================================

# ── Paths ─────────────────────────────────────────────────────────────
USE_CASE_DIR: Path = Path(__file__).resolve().parent
INPUT_DIR: Path = USE_CASE_DIR / "input"
OUTPUT_ROOT: Path = USE_CASE_DIR / "output"

# ── Analysis naming ──────────────────────────────────────────────────
ANALYSIS_NAME: str = "production"

# ── Seed ─────────────────────────────────────────────────────────────
SEED: int = 42

# Whether to display plots interactively
SHOW: bool = False

# ── Tensor product toggle ────────────────────────────────────────────
# True  → combine the kernel channels defined in KERNEL_CHANNELS.
# False → single-kernel mode using SOAP_PARAMS + KERNEL_PARAMS.
# Grid search operates in whichever mode is active; both are subject to
# MAX_GRID_COMBINATIONS.
USE_TENSOR_PRODUCT: bool = False


# ─────────────────────────────────────────────────────────────────────
#  SINGLE-KERNEL MODE  (used when USE_TENSOR_PRODUCT = False)
# ─────────────────────────────────────────────────────────────────────
# normalize: L2-normalize each per-atom SOAP vector to unit length
# *before* computing the kernel.  Recommended for REMatch with
# non-linear metrics (rbf, polynomial) to avoid numerical instability.
SOAP_PARAMS: dict = dict(
    r_cut=4.0,
    sigma=0.1,
    n_max=8,
    l_max=4,
    center_atoms=["Cu"],
    average="off",
    normalize=False,
    n_jobs=-1,
    periodic=True,
)

KERNEL_PARAMS: dict = dict(
    method="average",
    metric="linear",
    # gamma=5.0,                  # add for rbf / polynomial metrics
    # alpha=0.5,                  # add for the rematch method
)


# ─────────────────────────────────────────────────────────────────────
#  GRID SEARCH
# ─────────────────────────────────────────────────────────────────────
# When USE_TENSOR_PRODUCT = False:
#   sweeps FIXED_SOAP_KW × SOAP_GRID × KERNEL_GRID (single kernel),
#   scored with CKA against formation energy.
# When USE_TENSOR_PRODUCT = True:
#   sweeps per-channel soap_grid × kernel_grid across channels and
#   combines results with KERNEL_COMBINE.
MAX_GRID_COMBINATIONS: int = 500

# ── Single-kernel grid search (USE_TENSOR_PRODUCT = False) ───────────
# periodic must match SOAP_PARAMS (this is a periodic slab): a
# non-periodic grid search would tune descriptors that do not match
# the production ones.
FIXED_SOAP_KW: dict = dict(
    center_atoms=["Cu"],
    average="off",
    normalize=False,
    n_jobs=-1,
    periodic=True,
)

SOAP_GRID: dict = dict(
    r_cut=[3.0, 4.0, 5.0],
    sigma=[0.1, 0.5],
    n_max=[4, 6, 8],  # includes the production n_max=8
    l_max=[4, 6, 8],
)

# RBF gamma uses the median heuristic (``"median"``), resolved per SOAP
# descriptor by tastet.kernel.resolve_kernel_params so each representation
# gets a scale-appropriate gamma rather than a hardcoded value.
KERNEL_GRID = [
    dict(method="average", metric="linear"),
    dict(method="rematch", metric="linear", alpha=0.1),
    dict(method="average", metric="rbf", gamma="median"),
    dict(method="rematch", metric="rbf", gamma="median", alpha=0.1),
]

# CKA scorer target kernel for the (supervised) grid search.  The target
# values are the formation energies, pulled from the DB meta in run.py.
CKA_TARGET_KERNEL: str = "linear"


# ─────────────────────────────────────────────────────────────────────
#  MULTI-CHANNEL KERNEL  (used when USE_TENSOR_PRODUCT = True)
# ─────────────────────────────────────────────────────────────────────
# Each channel defines its own SOAP centers/species and kernel type.
#
# centers_from_smarts:
#   True  → resolve SOAP centers from FLEXIBLE_SMARTS atom indices.
#   False → use center_atoms from the channel's SOAP dict (or all atoms
#           if center_atoms is None).
#
# Per-channel grid search (optional):
#   soap_grid:   dict mapping SOAP param names to lists (same format as
#                SOAP_GRID).  Omit or {} for no sweep on this channel.
#   kernel_grid: list of kernel param dicts (same format as KERNEL_GRID).
#                Omit to use the channel's single "kernel" dict as-is.
KERNEL_CHANNELS: list[dict] = [
    {
        "name": "Cu_cluster",
        "centers_from_smarts": False,
        "soap": dict(
            center_atoms=["Cu"],
            species=["Cu"],
            r_cut=4.0,
            sigma=0.1,
            n_max=8,
            l_max=4,
            average="off",
            normalize=False,
            n_jobs=-1,
            periodic=True,
        ),
        "kernel": dict(method="average", metric="linear"),
        # "soap_grid": dict(sigma=[0.1, 0.5]),
        # "kernel_grid": [
        #     dict(method="average", metric="linear"),
        #     dict(method="average", metric="rbf", gamma=5.0),
        # ],
    },
    # Example: add a second channel for the Cu–surface interaction.
    # {
    #     "name": "Cu_surface",
    #     "centers_from_smarts": False,
    #     "soap": dict(
    #         center_atoms=["Cu"], species=["Cu", "O", "Ce"],
    #         r_cut=5.0, sigma=0.3, n_max=6, l_max=4,
    #         average="off", normalize=False, n_jobs=-1, periodic=True,
    #     ),
    #     "kernel": dict(method="rematch", metric="rbf", gamma=5.0, alpha=0.1),
    # },
]
KERNEL_COMBINE: str = "product"  # "product" (Hadamard), "sum" (mean), or "weighted_sum"
# KERNEL_WEIGHTS apply only when KERNEL_COMBINE == "weighted_sum";
# they are ignored for "product" and "sum".
KERNEL_WEIGHTS: list[float] | None = None


# ── Structure selection ──────────────────────────────────────────────
# Before FPS, keep only structures within this energy of the global
# minimum, i.e. E - E_gm ≤ SELECTION_ENERGY_MAX (E_gm = lowest surrogate
# formation energy over the whole set).  4.51 eV reproduces the same
# pool the study used (the old 15.0 eV absolute-formation-energy cutoff;
# E_gm ≈ 10.49 eV, so 15.0 - E_gm ≈ 4.51).
SELECTION_ENERGY_MAX: float = 4.51
SELECTION_K: int = 30
SELECTION_METHOD: str = "fps"
SELECTION_XYZ_TEMPLATE: str = "structure_{id}.xyz"

# ── KDE bandwidth for the kernel-step distance plot ─────────────────
KERNEL_KDE_BANDWIDTH: float = 0.02


# =====================================================================
#  USE-CASE-SPECIFIC — Cu clusters on surface
# =====================================================================

# ── Grid-search subsampling ──────────────────────────────────────────
# The grid search runs on an energy-balanced subset drawn on the fly
# from the production database by ``prepare.load_grid_search_structures``
# (inverse-density sampling that over-represents rare energy regions).
# The draw is fully determined by ``GRID_SEARCH_N_SAMPLES`` / ``NUM_BINS``
# / ``SEED``, so it is reproducible without persisting a separate subset.
GRID_SEARCH_N_SAMPLES: int = 50
NUM_BINS: int = 5

# ── Input data ───────────────────────────────────────────────────────
# Raw concatenated trajectory of all GOFFE runs (the committed source of
# truth).  ``input/split_trajectory.py`` splits it into one flat
# per-run trajectory ``<run_name>.traj`` inside ``RUNS_DIR``, which is
# what ``prepare._build_database`` reads.
RUNS_DIR: Path = INPUT_DIR
ALL_RUNS_TRAJ: Path = INPUT_DIR / "all_runs.traj"

# DFT energies of the FPS-selected structures (selection order), used by
# ``analysis/energy_profile.py`` to validate the surrogate.  Committed
# input, parallel to rh_complex's ``energies_round1.csv``.
ENERGIES_SELECTED_CSV: Path = INPUT_DIR / "energies_selected.csv"

TARGET_RUNS: list[str] = [
    "run_000_n1000_1L",
    "run_001_n1000_1L",
    "run_002_n1000_1L",
    "run_003_n1000_1L",
    "run_004_n1000_1L",
    "run_005_n1000_1L",
    "run_006_n1000_1L",
    "run_007_n1000_1L",
    "run_008_n1000_1L",
    "run_009_n1000_1L",
]


# =====================================================================
#  PATH HELPERS
# =====================================================================


def _use_channels() -> bool:
    """Whether multi-channel kernel mode is active.

    :returns: ``True`` when ``USE_TENSOR_PRODUCT`` is set *and*
        ``KERNEL_CHANNELS`` is non-empty.
    """
    return bool(globals().get("USE_TENSOR_PRODUCT", False)) and bool(KERNEL_CHANNELS)


def _centers_tag() -> str:
    """Short identifier for the active SOAP center selection.

    :returns: Tag like ``"c-Cu"``, ``"c-Cu-Zn"``, or ``"c-all"``.
    """
    ca = SOAP_PARAMS.get("center_atoms")
    if ca:
        return "c-" + "-".join(sorted(ca))
    return "c-all"


def soap_tag() -> str:
    """SOAP-parameters tag for the (single-kernel) soap output directory.

    :returns: Compact ``rcutN_sigN_nN_lN_<centers>`` string.
    """
    p = SOAP_PARAMS
    base = f"rcut{p['r_cut']}_sig{p['sigma']}_n{p['n_max']}_l{p['l_max']}"
    return f"{base}_{_centers_tag()}"


def kernel_tag() -> str:
    """Kernel-parameters tag for the (single-kernel) kernel output directory.

    Uses ``.get()`` for ``gamma`` / ``alpha`` so that ``KERNEL_PARAMS``
    can omit them safely.

    :returns: Compact ``method_metric[_gG][_aA]`` string.
    """
    p = KERNEL_PARAMS
    base = f"{p['method']}_{p['metric']}"
    if p.get("gamma") is not None:
        base += f"_g{p['gamma']}"
    if p.get("method") == "rematch":
        base += f"_a{p.get('alpha')}"
    return base


def combined_kernel_tag() -> str:
    """Hash-based tag for multi-channel combined kernels.

    :returns: ``f"{combine}_{8-char-hash}"``.
    """
    import hashlib
    import json

    blob = json.dumps(
        {
            "channels": KERNEL_CHANNELS,
            "combine": KERNEL_COMBINE,
            "weights": globals().get("KERNEL_WEIGHTS"),
        },
        sort_keys=True,
        default=str,
    )
    return f"{KERNEL_COMBINE}_{hashlib.sha256(blob.encode()).hexdigest()[:8]}"


def grid_search_tag() -> str:
    """Hash-based tag identifying a unique grid search configuration.

    :returns: 8-character hex hash.
    """
    import hashlib
    import json

    if _use_channels():
        blob = json.dumps(
            {
                "channels": KERNEL_CHANNELS,
                "combine": KERNEL_COMBINE,
                "weights": globals().get("KERNEL_WEIGHTS"),
                "scorer": CKA_TARGET_KERNEL,
                "random_seed": SEED,
            },
            sort_keys=True,
            default=str,
        )
    else:
        blob = json.dumps(
            {
                "soap_grid": SOAP_GRID,
                "kernel_grid": KERNEL_GRID,
                "fixed_soap_kw": FIXED_SOAP_KW,
                "scorer": CKA_TARGET_KERNEL,
                "centers": _centers_tag(),
                "random_seed": SEED,
            },
            sort_keys=True,
            default=str,
        )
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


# ── Per-channel hash helpers (multi-channel mode) ────────────────────


def channel_soap_tag(ch: dict) -> str:
    """Hash-keyed tag for a channel's SOAP cache.

    Includes a human-readable prefix (``rcut_sig_n_l``) followed by a
    hash that captures the full SOAP parameter set so that two
    configurations sharing the prefix but differing in, say, ``species``
    still produce distinct paths.

    :param ch: One entry from ``KERNEL_CHANNELS``.
    :returns: A directory name like ``rcut4.0_sig0.1_n8_l4_aabbccdd``.
    """
    import hashlib
    import json

    p = ch["soap"]
    base = (
        f"rcut{p.get('r_cut', '?')}"
        f"_sig{p.get('sigma', '?')}"
        f"_n{p.get('n_max', '?')}"
        f"_l{p.get('l_max', '?')}"
    )
    blob = {
        "soap": ch["soap"],
        "centers_from_smarts": ch.get("centers_from_smarts", False),
    }
    if ch.get("centers_from_smarts"):
        blob["flexible_smarts"] = globals().get("FLEXIBLE_SMARTS")
        blob["flexible_include_h"] = globals().get("FLEXIBLE_INCLUDE_H", True)
    h = hashlib.sha256(
        json.dumps(blob, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    return f"{base}_{h}"


def channel_kernel_tag(ch: dict) -> str:
    """Hash-keyed tag for a channel's kernel cache.

    Hashes ``ch["kernel"]`` as written in config; resolution of
    ``gamma="median"`` depends on the SOAP, which is handled by the
    nested-directory layout (the kernel directory lives inside the
    corresponding SOAP directory).

    :param ch: One entry from ``KERNEL_CHANNELS``.
    :returns: A directory name like ``average_linear_eeff0011``.
    """
    import hashlib
    import json

    k = ch["kernel"]
    method = k.get("method", "?")
    metric = k.get("metric", "?")
    base = f"{method}_{metric}"
    h = hashlib.sha256(json.dumps(k, sort_keys=True, default=str).encode()).hexdigest()[
        :8
    ]
    return f"{base}_{h}"


def analysis_dir() -> Path:
    """Return the analysis output directory (creates it if missing).

    :returns: ``OUTPUT_ROOT / ANALYSIS_NAME``.
    """
    d = OUTPUT_ROOT / ANALYSIS_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def soap_dir() -> Path:
    """Return the SOAP cache directory for the current SOAP_PARAMS.

    :returns: ``analysis_dir() / soap_tag()``.
    """
    d = analysis_dir() / soap_tag()
    d.mkdir(exist_ok=True)
    return d


def kernel_dir() -> Path:
    """Return the kernel cache directory.

    For single-kernel mode it lives under ``soap_dir() / kernel_tag()``;
    for multi-channel mode under ``analysis_dir() / combined_kernel_tag()``.
    """
    if _use_channels():
        d = analysis_dir() / combined_kernel_tag()
    else:
        d = soap_dir() / kernel_tag()
    d.mkdir(exist_ok=True)
    return d


def channel_dir(name: str) -> Path:
    """Return the base directory for a channel name.

    :param name: Channel name (must match ``KERNEL_CHANNELS[i]["name"]``).
    :returns: ``analysis_dir() / "channels" / name``.
    """
    d = analysis_dir() / "channels" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def channel_soap_dir(ch: dict) -> Path:
    """Hash-keyed SOAP cache directory for a channel.

    :param ch: One entry from ``KERNEL_CHANNELS``.
    :returns: ``channels/<name>/<soap_tag>``.
    """
    d = channel_dir(ch["name"]) / channel_soap_tag(ch)
    d.mkdir(parents=True, exist_ok=True)
    return d


def channel_kernel_dir(ch: dict) -> Path:
    """Hash-keyed kernel cache directory for a channel.

    Nested inside :func:`channel_soap_dir` so one SOAP file serves every
    kernel variant computed from it.

    :param ch: One entry from ``KERNEL_CHANNELS``.
    :returns: ``channels/<name>/<soap_tag>/<kernel_tag>``.
    """
    d = channel_soap_dir(ch) / channel_kernel_tag(ch)
    d.mkdir(parents=True, exist_ok=True)
    return d


def channel_soap_path(ch: dict) -> Path:
    """Cached SOAP descriptors for a channel.

    :param ch: One entry from ``KERNEL_CHANNELS``.
    :returns: ``channel_soap_dir(ch) / "soap.npz"``.
    """
    return channel_soap_dir(ch) / "soap.npz"


def channel_kernel_path(ch: dict) -> Path:
    """Cached kernel matrix for a channel.

    :param ch: One entry from ``KERNEL_CHANNELS``.
    :returns: ``channel_kernel_dir(ch) / "kernel.npz"``.
    """
    return channel_kernel_dir(ch) / "kernel.npz"


def grid_search_dir() -> Path:
    """Return the grid-search output directory for the current settings.

    :returns: ``analysis_dir() / "grid_search" / grid_search_tag()``.
    """
    d = analysis_dir() / "grid_search" / grid_search_tag()
    d.mkdir(parents=True, exist_ok=True)
    return d


def selection_dir() -> Path:
    """Return the selection output directory under the active kernel dir.

    :returns: ``kernel_dir() / "selection"``.
    """
    d = kernel_dir() / "selection"
    d.mkdir(exist_ok=True)
    return d


def db_path() -> Path:
    """ASE database for the active analysis."""
    return analysis_dir() / "structures.db"


def csv_path() -> Path:
    """CSV mirror of the active database."""
    return analysis_dir() / "structures.csv"


def soap_path() -> Path:
    """Cached SOAP descriptors for the active parameters."""
    return soap_dir() / "soap.npz"


def kernel_path() -> Path:
    """Cached kernel matrix for the active parameters."""
    return kernel_dir() / "kernel.npz"


def kernel_meta_path() -> Path:
    """JSON file recording resolved kernel parameters."""
    return kernel_dir() / "kernel_meta.json"


def plot_path() -> Path:
    """2-D kPCA plot."""
    return kernel_dir() / "kpca.png"


def kpca_csv_path() -> Path:
    """CSV of kPCA projections (kpc1, kpc2, kpc3 per structure)."""
    return kernel_dir() / "kpca_projections.csv"


def kpca_meta_path() -> Path:
    """JSON file recording kPCA metadata (explained variance)."""
    return kernel_dir() / "kpca_meta.json"


def grid_search_csv() -> Path:
    """Grid search results CSV."""
    return grid_search_dir() / "results.csv"


def grid_search_heatmap_path() -> Path:
    """Grid search results heatmap."""
    return grid_search_dir() / "heatmaps.png"


def grid_search_config_path() -> Path:
    """JSON file recording the grid search configuration."""
    return grid_search_dir() / "config.json"


def selection_csv_path() -> Path:
    """CSV of selected structure metadata."""
    return selection_dir() / "selected_structures.csv"


def selection_plot_path() -> Path:
    """2-D kPCA plot with selected structures highlighted."""
    return selection_dir() / "selection.png"
