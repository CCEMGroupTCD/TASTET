"""Configuration for the conformer analysis of the Rh complex.

General settings (top) — present in every use case.
Use-case-specific settings (middle) — unique to this conformer system.
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
ANALYSIS_NAME: str = "production"

# ── Seed ─────────────────────────────────────────────────────────────
SEED: int = 22

# Whether to display plots interactively
SHOW: bool = False

# ── Tensor product toggle ────────────────────────────────────────────
# True  → combine multiple kernel channels defined in KERNEL_CHANNELS.
# False → single-kernel mode using SOAP_PARAMS + KERNEL_PARAMS.
USE_TENSOR_PRODUCT: bool = True


# ─────────────────────────────────────────────────────────────────────
#  SINGLE-KERNEL MODE  (used when USE_TENSOR_PRODUCT = False)
# ─────────────────────────────────────────────────────────────────────
SOAP_PARAMS: dict = dict(
    r_cut=4.0,
    sigma=0.1,
    n_max=8,
    l_max=8,
    center_atoms=["Rh"],
    average="off",
    normalize=True,
    n_jobs=-1,
    periodic=False,
)

KERNEL_PARAMS: dict = dict(
    method="average",
    metric="rbf",
    gamma="median",
)


# ─────────────────────────────────────────────────────────────────────
#  GRID SEARCH
# ─────────────────────────────────────────────────────────────────────
MAX_GRID_COMBINATIONS: int = 500
GRID_SEARCH_N_SAMPLES: int | None = None

FIXED_SOAP_KW: dict = dict(
    n_max=8,
    l_max=8,
    average="off",
    normalize=True,
    n_jobs=-1,
    periodic=False,
)

SOAP_GRID: dict = dict(
    r_cut=[4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    sigma=[0.1, 0.3, 0.5],
    center_atoms=[["Rh"]],
)

KERNEL_GRID = [
    dict(method="average", metric="linear"),
    dict(method="average", metric="rbf", gamma="median"),
    dict(method="rematch", metric="rbf", gamma="median", alpha=0.1),
]

# CKA target-kernel type for round2.py's supervised grid search.
CKA_TARGET_KERNEL: str = "linear"  # "linear" or "rbf"

# ── Grid-search analysis (analysis/analyze_gridsearch.py) ────────────
# Diagnostics for ranking SOAP × kernel combinations by their pairwise
# distance distributions.
GRID_ANALYSIS_BANDWIDTH: float = 0.01  # KDE bandwidth for the distance distribution
GRID_ANALYSIS_TOP_N: int = 5  # number of top-ranked combinations to plot
GRID_ANALYSIS_IQR_FLOOR: float | None = (
    None  # min IQR/√2 to qualify; None disables the filter
)

# ─────────────────────────────────────────────────────────────────────
#  MULTI-CHANNEL KERNEL  (used when USE_TENSOR_PRODUCT = True)
# ─────────────────────────────────────────────────────────────────────
_SHARED_SPECIES: list[str] = ["Rh", "P", "C", "O", "H", "N"]
_SOAP_GRID_SHARED: dict = dict(
    r_cut=[2.0, 3.0, 4.0, 5.0],
    sigma=[0.1, 0.5],
)
KERNEL_CHANNELS: list[dict] = [
    {
        "name": "core_kernel",
        "soap": dict(
            r_cut=3.0,
            centers=[10, 55, 52, 0, 1],
            species=_SHARED_SPECIES,
            sigma=0.1,
            n_max=8,
            l_max=8,
            average="off",
            normalize=True,
            n_jobs=-1,
            periodic=False,
        ),
        "kernel": dict(method="average", metric="rbf", gamma="median"),
        "soap_grid": _SOAP_GRID_SHARED,
        "kernel_grid": [
            dict(method="average", metric="linear"),
            dict(method="average", metric="rbf", gamma="median"),
        ],
    },
    {
        "name": "periphery_kernel",
        "soap": dict(
            r_cut=2.0,
            centers=[7, 30, 36, 14, 23, 9, 17, 19],
            species=_SHARED_SPECIES,
            sigma=0.5,
            n_max=8,
            l_max=8,
            average="off",
            normalize=True,
            n_jobs=-1,
            periodic=False,
        ),
        "kernel": dict(
            method="rematch",
            metric="rbf",
            gamma="median",
            alpha=0.5,
        ),
        "soap_grid": _SOAP_GRID_SHARED,
        "kernel_grid": [
            dict(method="rematch", metric="linear", alpha=0.5),
            dict(method="rematch", metric="rbf", gamma="median", alpha=0.1),
        ],
    },
]

KERNEL_COMBINE: str = "product"


# ── Structure selection ──────────────────────────────────────────────
SELECTION_K: int = 28
SELECTION_METHOD: str = "fps"
SELECTION_XYZ_TEMPLATE: str = "conformer_{id}.xyz"

# ── KDE bandwidth for the kernel-step distance plot ─────────────────
KERNEL_KDE_BANDWIDTH: float = 0.01


# =====================================================================
#  ROUND 2 — supervised CKA re-optimisation + incremental selection
# =====================================================================
# Round 1 (run.py) selects SELECTION_K conformers with the unsupervised
# KERNEL_CHANNELS above; their DFT energies (ENERGIES_CSV) then supervise
# a second grid search scored by CKA (reoptimise.py), which lands on a
# *different* representation (ROUND2_KERNEL_CHANNELS). To keep both rounds
# reproducible from this one file, round 2 lives in its own output
# namespace and uses its own channels — the round2/ scripts apply these
# at runtime (round2._common.activate_round2); run.py is unaffected.

ROUND2_ANALYSIS_NAME: str = "round2"

ENERGIES_CSV: Path = USE_CASE_DIR / "input" / "energies_round1.csv"
ENERGY_COL: str = "E (Ha)"  # absolute DFT energy column in ENERGIES_CSV

TOTAL_BUDGET: int = 56  # total relaxed structures wanted (round 1 + round 2)
ZOOM_K: int = 19  # picks for zoom_select / nearest_select

# Focused selection within a kPCA region of interest (zoom_select). Set
# the bounds after viewing the round-2 kPCA; None = unbounded on that side.
ZOOM_BOX: dict = {"kpc1": (None, None), "kpc2": (None, None)}
ZOOM_METHOD: str = "fps"  # "fps" (seeded by round-1 in-box) or "kmedoids"

# Point of interest for nearest_select. None → lowest-energy round-1 conformer.
ZOOM_CENTER: int | None = None

# Re-optimised channels chosen from the CKA grid search (linear kernels).
# Same structure as KERNEL_CHANNELS: the chosen soap/kernel are what the
# selection uses; soap_grid/kernel_grid define the space reoptimise sweeps.
ROUND2_KERNEL_CHANNELS: list[dict] = [
    {
        "name": "core_kernel",
        "soap": dict(
            r_cut=2.0,
            centers=[10, 55, 52, 0, 1],
            species=_SHARED_SPECIES,
            sigma=0.1,
            n_max=8,
            l_max=8,
            average="off",
            normalize=True,
            n_jobs=-1,
            periodic=False,
        ),
        "kernel": dict(method="average", metric="linear"),
        "soap_grid": _SOAP_GRID_SHARED,
        "kernel_grid": [
            dict(method="average", metric="linear"),
            dict(method="average", metric="rbf", gamma="median"),
        ],
    },
    {
        "name": "periphery_kernel",
        "soap": dict(
            r_cut=5.0,
            centers=[7, 30, 36, 14, 23, 9, 17, 19],
            species=_SHARED_SPECIES,
            sigma=0.5,
            n_max=8,
            l_max=8,
            average="off",
            normalize=True,
            n_jobs=-1,
            periodic=False,
        ),
        "kernel": dict(method="rematch", metric="linear", alpha=0.5),
        "soap_grid": _SOAP_GRID_SHARED,
        "kernel_grid": [
            dict(method="rematch", metric="linear", alpha=0.5),
            dict(method="rematch", metric="rbf", gamma="median", alpha=0.1),
        ],
    },
]

ROUND2_KERNEL_COMBINE: str = "product"


# =====================================================================
#  USE-CASE-SPECIFIC — Rh complex conformer analysis
# =====================================================================

SDF_FILE: Path = USE_CASE_DIR / "input" / "open_babel_Rh_conformers.sdf"

# FLEXIBLE_SMARTS: list[str] = ["..."]  # uncomment to use SMARTS-derived centres
FLEXIBLE_INCLUDE_H: bool = False


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
    """Short identifier for the active SOAP centre selection.

    :returns: Tag like ``"c-Rh"``, ``"c-Cu-Zn"``, ``"flex-<hash>"``,
        or ``"c-all"``.
    """
    import hashlib

    ca = SOAP_PARAMS.get("center_atoms")
    if ca:
        return "c-" + "-".join(sorted(ca))
    fs = globals().get("FLEXIBLE_SMARTS")
    if fs:
        h = hashlib.sha256(str(fs).encode()).hexdigest()[:8]
        return f"flex-{h}"
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

    Hashes the same parameter set as :func:`grid_search_tag` (channel
    branch), so that different combine modes / weightings produce
    distinct directories. The field set is kept identical across the
    use cases for cross-use-case consistency.

    :returns: ``f"{combine}_{8-char-hash}"``.
    """
    import hashlib
    import json

    blob = json.dumps(
        {
            "channels": KERNEL_CHANNELS,
            "combine": KERNEL_COMBINE,
            "weights": globals().get("KERNEL_WEIGHTS"),
            "flexible_smarts": globals().get("FLEXIBLE_SMARTS"),
            "flexible_include_h": globals().get("FLEXIBLE_INCLUDE_H", True),
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

    flex_smarts = globals().get("FLEXIBLE_SMARTS")
    flex_include_h = globals().get("FLEXIBLE_INCLUDE_H", True)
    if _use_channels():
        blob = json.dumps(
            {
                "channels": KERNEL_CHANNELS,
                "combine": KERNEL_COMBINE,
                "weights": globals().get("KERNEL_WEIGHTS"),
                "flexible_smarts": flex_smarts,
                "flexible_include_h": flex_include_h,
                "scorer": globals().get("CKA_TARGET_KERNEL"),
                "random_seed": SEED,
                "number_subsamples": GRID_SEARCH_N_SAMPLES,
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
                "flexible_smarts": flex_smarts,
                "flexible_include_h": flex_include_h,
                "scorer": globals().get("CKA_TARGET_KERNEL"),
                "centers": _centers_tag(),
                "random_seed": SEED,
                "number_subsamples": GRID_SEARCH_N_SAMPLES,
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
    configurations sharing the prefix but differing in, say,
    ``species`` or ``centers`` still produce distinct paths.

    :param ch: One entry from ``KERNEL_CHANNELS``.
    :returns: A directory name like ``rcut2.0_sig0.5_n8_l8_aabbccdd``.
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

    Hashes ``ch["kernel"]`` as written in config (without resolving
    ``gamma="median"``); resolution of ``gamma="median"`` depends on
    the SOAP, which is already handled by the nested-directory layout
    (the kernel directory lives inside the corresponding SOAP
    directory, so a SOAP change automatically produces a fresh kernel
    cache).

    :param ch: One entry from ``KERNEL_CHANNELS``.
    :returns: A directory name like ``rematch_rbf_eeff0011``.
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

    Contains one or more SOAP-hashed subdirectories, each in turn
    containing one or more kernel-hashed subdirectories.

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

    Nested inside :func:`channel_soap_dir` because the kernel value
    depends on both the SOAP and the kernel parameters; sharing a
    parent directory keeps related caches together and lets one
    SOAP file serve every kernel variant computed from it.

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


def kpca_analysis_dir() -> Path:
    """Analysis output subdirectory next to the active kPCA outputs.

    Holds script-generated kPCA diagnostics (e.g. the energy-coloured
    plots from ``analysis/plot_energy_kpca.py``), kept out of the
    pipeline-generated files.

    :returns: ``kernel_dir() / "analysis"`` (created if missing).
    """
    d = kernel_dir() / "analysis"
    d.mkdir(parents=True, exist_ok=True)
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
    """CSV of kPCA projections (kpc1, kpc2, kpc3 per conformer)."""
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


def grid_search_analysis_dir(grid_dir: Path) -> Path:
    """Analysis output subdirectory for one grid-search directory.

    Keeps script-generated diagnostics (rankings, top-N plot) separate
    from the pipeline-generated grid-search files.

    :param grid_dir: A grid-search directory under
        ``analysis_dir() / "grid_search"``.
    :returns: ``grid_dir / "analysis"`` (created if missing).
    """
    d = grid_dir / "analysis"
    d.mkdir(parents=True, exist_ok=True)
    return d


def selection_csv_path() -> Path:
    """CSV of selected conformer metadata."""
    return selection_dir() / "selected_structures.csv"


def selection_plot_path() -> Path:
    """2-D kPCA plot with selected conformers highlighted."""
    return selection_dir() / "selection.png"
