"""Configuration for the Cu-cluster-on-surface analysis.

General settings (top) — present in every use case.
Use-case-specific settings (middle) — unique to Pablo's system.
Path helpers (bottom) — reusable output tree.
"""

from pathlib import Path


# =====================================================================
#  GENERAL SETTINGS
# =====================================================================

# ── Paths ─────────────────────────────────────────────────────────────
USE_CASE_DIR: Path = Path(__file__).resolve().parent
OUTPUT_ROOT: Path = USE_CASE_DIR / "output"

# ── Analysis naming ──────────────────────────────────────────────────
ANALYSIS_NAME: str = "new_all_1L"

# ── Seed ─────────────────────────────────────────────────────────────
SEED: int = 42

# Whether to display plots interactively
SHOW: bool = False

# ── Tensor product toggle ────────────────────────────────────────────
# True  → combine multiple kernel channels defined in KERNEL_CHANNELS.
# False → single-kernel mode using SOAP_PARAMS + KERNEL_PARAMS.
# Grid search operates in whichever mode is active.
# Both modes are subject to MAX_GRID_COMBINATIONS.
USE_TENSOR_PRODUCT: bool = False


# ─────────────────────────────────────────────────────────────────────
#  SINGLE-KERNEL MODE  (used when USE_TENSOR_PRODUCT = False)
# ─────────────────────────────────────────────────────────────────────
# normalize: L2-normalise each per-atom SOAP vector to unit length
# *before* computing the kernel.  Recommended for REMatch with
# non-linear metrics (rbf, polynomial) to avoid numerical instability.
SOAP_PARAMS: dict = dict(
    r_cut=4.0, sigma=0.1, n_max=8, l_max=4,
    center_atoms=["Cu"], average="off", normalize=False,
    n_jobs=-1, periodic=True,
)

KERNEL_PARAMS: dict = dict(
    method="average", metric="linear", gamma=None, alpha=0.5,
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
FIXED_SOAP_KW: dict = dict(
    center_atoms=["Cu"], average="off", normalize=False,
    n_jobs=-1, periodic=False,
)

SOAP_GRID: dict = dict(
    r_cut=[3.0, 4.0, 5.0], sigma=[0.1, 0.5], n_max=[4, 6], l_max=[4, 6],
)

KERNEL_GRID = [
    dict(method="average", metric="linear"),
    dict(method="rematch", metric="linear", alpha=0.1),

    dict(method="average", metric="rbf", gamma=1.0),
    dict(method="average", metric="rbf", gamma=5.0),
    dict(method="rematch", metric="rbf", gamma=1.0, alpha=0.1),
    dict(method="rematch", metric="rbf", gamma=5.0, alpha=0.1),

    dict(method="average", metric="polynomial", degree=2, gamma=1.0, coef0=0.0),
    dict(method="rematch", metric="polynomial", degree=2, gamma=1.0, coef0=0.0, alpha=0.1),
]

CKA_TARGET_KERNEL: str = "linear"


# ─────────────────────────────────────────────────────────────────────
#  MULTI-CHANNEL KERNEL  (used when USE_TENSOR_PRODUCT = True)
# ─────────────────────────────────────────────────────────────────────
# Each channel defines its own SOAP centres/species and kernel type.
# Channels are always defined here — USE_TENSOR_PRODUCT controls
# whether the pipeline reads them or falls back to single-kernel mode.
#
# centers_from_smarts:
#   True  → resolve SOAP centres from FLEXIBLE_SMARTS atom indices.
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
            center_atoms=["Cu"], species=["Cu"],
            r_cut=4.0, sigma=0.1, n_max=8, l_max=4,
            average="off", normalize=False, n_jobs=-1, periodic=True,
        ),
        "kernel": dict(method="average", metric="linear"),
        # "soap_grid": dict(sigma=[0.1, 0.5]),
        # "kernel_grid": [
        #     dict(method="average", metric="linear"),
        #     dict(method="average", metric="rbf", gamma=5.0),
        # ],
    },
    # Example: add a second channel for Cu–surface interaction
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
KERNEL_COMBINE: str = "product"  # "product" (Hadamard) or "sum" (mean)


# ── Structure selection ──────────────────────────────────────────────
SELECTION_ENERGY_MAX: float = 15.0
SELECTION_K: int = 30
SELECTION_METHOD: str = "fps"


# =====================================================================
#  USE-CASE-SPECIFIC — Cu clusters on surface
# =====================================================================

# ── Subsampling ──────────────────────────────────────────────────────
MASTER_ANALYSIS_NAME: str = "new_all_1L"
N_SUBSAMPLE: int = 50
NUM_BINS: int = 5

# ── Input data ───────────────────────────────────────────────────────
RUNS_DIR: Path = USE_CASE_DIR / "new_runs"

TARGET_RUNS: list[str] = [
    "run_000_n1000_1L", "run_001_n1000_1L", "run_002_n1000_1L",
    "run_003_n1000_1L", "run_004_n1000_1L", "run_005_n1000_1L",
    "run_006_n1000_1L", "run_007_n1000_1L", "run_008_n1000_1L",
    "run_009_n1000_1L",
]

E_SURFACE_2L: float = -274.2128
E_SURFACE_1L: float = -132.4813
E_CU_BULK: float = -3.73


def n_layers_from_dirname(dir_name: str) -> int:
    """Parse ``"1L"`` / ``"2L"`` from a run directory name.

    :param dir_name: Run directory name, e.g. ``"run_000_n1000_1L"``.
    :returns: Number of layers (1 or 2).
    :raises ValueError: If the layer count cannot be parsed.
    """
    for part in dir_name.split("_"):
        if part.endswith("L") and part[:-1].isdigit():
            return int(part[:-1])
    raise ValueError(f"Cannot parse layer count from {dir_name}")


def surface_energy(dir_name: str) -> float:
    """Return the DFT surface energy for a given run.

    :param dir_name: Run directory name.
    :returns: Surface energy in eV.
    """
    return E_SURFACE_2L if n_layers_from_dirname(dir_name) == 2 else E_SURFACE_1L


# =====================================================================
#  PATH HELPERS
# =====================================================================

def _use_channels() -> bool:
    """True when multi-channel kernel mode is active."""
    return bool(globals().get("USE_TENSOR_PRODUCT", False))


def _centers_tag() -> str:
    """Short string that uniquely identifies the active centre selection.

    center_atoms=["Cu"]        → "c-Cu"
    center_atoms=["Cu","Zn"]   → "c-Cu-Zn"
    Neither                    → "c-all"
    """
    ca = SOAP_PARAMS.get("center_atoms")
    if ca:
        return "c-" + "-".join(sorted(ca))
    return "c-all"


def soap_tag() -> str:
    p = SOAP_PARAMS
    base = f"rcut{p['r_cut']}_sig{p['sigma']}_n{p['n_max']}_l{p['l_max']}"
    return f"{base}_{_centers_tag()}"


def kernel_tag() -> str:
    p = KERNEL_PARAMS
    base = f"{p['method']}_{p['metric']}"
    if p["gamma"] is not None:
        base += f"_g{p['gamma']}"
    if p["method"] == "rematch":
        base += f"_a{p['alpha']}"
    return base


def combined_kernel_tag() -> str:
    """Hash-based tag for multi-channel combined kernels."""
    import hashlib, json
    blob = json.dumps(
        {"channels": KERNEL_CHANNELS, "combine": KERNEL_COMBINE},
        sort_keys=True, default=str,
    )
    return f"{KERNEL_COMBINE}_{hashlib.sha256(blob.encode()).hexdigest()[:8]}"


def grid_search_tag() -> str:
    import hashlib, json
    if _use_channels():
        blob = json.dumps(
            {"channels": KERNEL_CHANNELS,
             "combine": KERNEL_COMBINE,
             "scorer": CKA_TARGET_KERNEL,
             "random_seed": SEED},
            sort_keys=True, default=str,
        )
    else:
        blob = json.dumps(
            {"soap_grid": SOAP_GRID, "kernel_grid": KERNEL_GRID,
             "fixed_soap_kw": FIXED_SOAP_KW, "scorer": CKA_TARGET_KERNEL,
             "centers": _centers_tag(), "random_seed": SEED},
            sort_keys=True, default=str,
        )
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


def analysis_dir() -> Path:
    d = OUTPUT_ROOT / ANALYSIS_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d

def soap_dir() -> Path:
    d = analysis_dir() / soap_tag(); d.mkdir(exist_ok=True); return d

def kernel_dir() -> Path:
    if _use_channels():
        d = analysis_dir() / combined_kernel_tag()
    else:
        d = soap_dir() / kernel_tag()
    d.mkdir(exist_ok=True); return d

def channel_dir(name: str) -> Path:
    d = analysis_dir() / "channels" / name
    d.mkdir(parents=True, exist_ok=True); return d

def channel_soap_path(name: str) -> Path:
    return channel_dir(name) / "soap.npz"

def channel_kernel_path(name: str) -> Path:
    return channel_dir(name) / "kernel.npz"

def grid_search_dir() -> Path:
    d = analysis_dir() / "grid_search" / grid_search_tag()
    d.mkdir(parents=True, exist_ok=True); return d

def selection_dir() -> Path:
    d = kernel_dir() / "selection"; d.mkdir(exist_ok=True); return d

def db_path() -> Path:            return analysis_dir() / "structures.db"
def csv_path() -> Path:           return analysis_dir() / "structures.csv"
def master_db_path() -> Path:     return OUTPUT_ROOT / MASTER_ANALYSIS_NAME / "structures.db"
def subsample_meta_path() -> Path: return analysis_dir() / "subsample_settings.json"
def soap_path() -> Path:          return soap_dir() / "soap.npz"
def kernel_path() -> Path:        return kernel_dir() / "kernel.npz"
def kernel_meta_path() -> Path:   return kernel_dir() / "kernel_meta.json"
def plot_path() -> Path:          return kernel_dir() / "kpca.png"
def kpca_csv_path() -> Path:      return kernel_dir() / "kpca_projections.csv"
def kpca_meta_path() -> Path:     return kernel_dir() / "kpca_meta.json"
def grid_search_csv() -> Path:    return grid_search_dir() / "results.csv"
def grid_search_heatmap_path() -> Path: return grid_search_dir() / "heatmaps.png"
def grid_search_config_path() -> Path:  return grid_search_dir() / "config.json"
def selection_csv_path() -> Path:  return selection_dir() / "selected_structures.csv"
def selection_plot_path() -> Path: return selection_dir() / "selection.png"