"""Scored multi-channel grid search.

Companion to :func:`tastet.pipeline.grid_search_step` (single kernel) for
the tensor-product / multi-channel case. Sweeps each channel's optional
``soap_grid`` × ``kernel_grid``, combines the per-channel kernels via
``KERNEL_COMBINE`` / ``KERNEL_WEIGHTS``, and — when a scorer and target
are supplied — scores each combined kernel (e.g. CKA against energies)
and writes the same artifacts as the single-kernel sweep: a ranked
``results.csv`` and score heatmaps. Distance-distribution diagnostics are
produced in every case, so this also serves as a drop-in for the
unsupervised multi-channel sweep.
"""

from __future__ import annotations

import json
import sys
from itertools import product as iproduct
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from tastet.soap_utils import compute_soap
from tastet.kernel import compute_kernel, resolve_kernel_params, combine_kernels
from tastet.distance import pairwise_dataframe, pairwise_distances
from tastet.plotting.distance import plot_grid_histograms
from tastet.plotting.heatmap import plot_grid_heatmaps
from tastet.sweep.results import save_results


def grid_search_multichannel_step(
    cfg,
    atoms_list: list,
    ids: np.ndarray | list,
    channels: list[dict],
    *,
    scorer=None,
    target: np.ndarray | None = None,
    resolve_channel_soap: Callable[[dict], dict] | None = None,
) -> None:
    """Sweep per-channel SOAP × kernel grids, combine, score, and plot.

    The total sweep is the cartesian product of each channel's
    ``soap_grid`` × ``kernel_grid`` (a channel with neither contributes
    its single ``soap`` / ``kernel`` as-is), subject to
    ``cfg.MAX_GRID_COMBINATIONS``.

    When *scorer* and *target* are both provided (supervised), each
    combined kernel is scored and a ranked ``results.csv`` plus score
    heatmaps are written — identical in form to
    :func:`tastet.pipeline.grid_search_step`. Distance-distribution
    outputs (histogram grid, pairwise CSV, summary CSV) are written in
    every case.

    :param cfg: Config module (needs ``grid_search_*`` paths,
        ``KERNEL_COMBINE``, optionally ``KERNEL_WEIGHTS`` /
        ``MAX_GRID_COMBINATIONS``).
    :param atoms_list: Structures to featurise (e.g. the labelled subset).
    :param ids: Structure identifiers, one per element of *atoms_list*.
    :param channels: Channel definitions (e.g. ``cfg.KERNEL_CHANNELS``),
        each optionally carrying ``soap_grid`` / ``kernel_grid``.
    :param scorer: A :class:`~tastet.metrics.Scorer` instance, or ``None``
        for an unsupervised sweep.
    :param target: Target array passed to the scorer (e.g. energies),
        aligned with *atoms_list*. ``None`` for unsupervised.
    :param resolve_channel_soap: Callable mapping a channel dict to its
        SOAP kwargs (resolving SMARTS centres etc.). Defaults to
        ``dict(channel["soap"])``.
    :returns: ``None``.
    """
    if resolve_channel_soap is None:
        resolve_channel_soap = lambda ch: dict(ch["soap"])

    supervised = scorer is not None and target is not None
    out_dir = cfg.grid_search_dir()
    config_path = cfg.grid_search_config_path()
    if config_path.exists():
        print(f"Grid search already exists -> {out_dir}")
        print("  Change channel grids in config.py to run a new sweep.")
        return

    # ── Expand per-channel combinations ──────────────────────────────
    channel_options: list[list[tuple]] = []
    for ch in channels:
        name = ch["name"]
        base_soap = resolve_channel_soap(ch)
        base_kernel = dict(ch["kernel"])

        soap_grid = ch.get("soap_grid", {})
        kernel_grid = ch.get("kernel_grid", [base_kernel])

        soap_keys = list(soap_grid.keys())
        soap_combos = (
            [
                dict(zip(soap_keys, vals))
                for vals in iproduct(*(soap_grid[k] for k in soap_keys))
            ]
            if soap_keys
            else [{}]
        )

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

    all_combos = list(iproduct(*channel_options))
    n_total = len(all_combos)
    max_combos = getattr(cfg, "MAX_GRID_COMBINATIONS", 500)
    if n_total > max_combos:
        per_ch = " × ".join(
            f"{ch['name']}({len(opts)})" for ch, opts in zip(channels, channel_options)
        )
        sys.exit(
            f"Multi-channel grid search has {n_total} combinations "
            f"({per_ch}), exceeding MAX_GRID_COMBINATIONS={max_combos}.  "
            f"Reduce per-channel soap_grid / kernel_grid entries."
        )

    label = f" (scored by {scorer.name})" if supervised else ""
    print(f"Multi-channel grid search: {n_total} combinations{label} ...")

    # ── Sweep ────────────────────────────────────────────────────────
    mode = getattr(cfg, "KERNEL_COMBINE", "product")
    weights = getattr(cfg, "KERNEL_WEIGHTS", None)

    soap_cache: dict[str, list] = {}
    result_rows: list[dict] = []
    kernel_entries: list[dict] = []
    pair_frames: list[pd.DataFrame] = []

    for combo in tqdm(all_combos, desc="Multi-channel sweep"):
        channel_Ks: list[np.ndarray] = []
        params: dict = {}

        for ch_name, soap_kw, kern_kw, swept in combo:
            cache_key = json.dumps({ch_name: soap_kw}, sort_keys=True, default=str)
            if cache_key not in soap_cache:
                soap_cache[cache_key] = compute_soap(atoms_list, **soap_kw)
            soap_list = soap_cache[cache_key]

            resolved = resolve_kernel_params(soap_list, kern_kw, verbose=False)
            K_ch = compute_kernel(soap_list, **resolved, verbose=False)
            channel_Ks.append(K_ch)

            for k, v in resolved.items():
                swept[f"{ch_name}__{k}"] = v
            params.update(swept)

        K = combine_kernels(channel_Ks, mode=mode, weights=weights)
        kernel_entries.append({"K": K, "params": params})

        # ── Score row (supervised) ──────────────────────────────────
        if supervised:
            row = dict(params)
            try:
                score = scorer(K, target)
                row["status"] = "OK" if score is not None else "FAILED"
                row[scorer.name] = float(score) if score is not None else float("nan")
            except Exception as exc:  # noqa: BLE001 — record, don't abort the sweep
                row["status"] = "FAILED"
                row[scorer.name] = float("nan")
                row["error"] = str(exc)
            result_rows.append(_jsonable(row))

        # ── Pairwise distances frame ────────────────────────────────
        df_pairs = pairwise_dataframe(K, ids)
        for col, val in params.items():
            df_pairs[col] = _scalarise(val)
        pair_frames.append(df_pairs)

    # ── Score results + heatmaps (supervised) ───────────────────────
    if supervised:
        df = pd.DataFrame(result_rows)
        csv_path = save_results(df, cfg.grid_search_csv())
        print(f"  Results      -> {csv_path}")
        plot_grid_heatmaps(
            df,
            value=scorer.name,
            out_path=cfg.grid_search_heatmap_path(),
            suptitle=f"Multi-channel grid search — {scorer.name}",
            show=getattr(cfg, "SHOW", False),
        )
        print(f"  Heatmaps     -> {cfg.grid_search_heatmap_path()}")

    # ── Distance distributions (always) ─────────────────────────────
    hist_path = out_dir / "distance_distributions.png"
    plot_grid_histograms(
        kernel_entries,
        out_path=hist_path,
        suptitle="Distance distributions — multi-channel grid search",
        show=getattr(cfg, "SHOW", False),
    )
    print(f"  Dist plots   -> {hist_path}")

    combined = pd.concat(pair_frames, ignore_index=True)
    pair_csv = out_dir / "pairwise_distances.csv"
    combined.to_csv(pair_csv, index=False)
    print(f"  Pairwise CSV -> {pair_csv}  ({len(combined)} rows)")

    summary_rows: list[dict] = []
    for entry in kernel_entries:
        d = pairwise_distances(entry["K"])
        row = {k: _scalarise(v) for k, v in entry["params"].items()}
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
    all_species = sorted({s for a in atoms_list for s in a.get_chemical_symbols()})
    snapshot_channels = []
    for ch in channels:
        soap_with_species = dict(ch["soap"])
        soap_with_species.setdefault("species", all_species)
        snapshot_channels.append(
            {
                "name": ch["name"],
                "centers_from_smarts": ch.get("centers_from_smarts", False),
                "soap": soap_with_species,
                "kernel": ch["kernel"],
                "soap_grid": ch.get("soap_grid", {}),
                "kernel_grid": ch.get("kernel_grid", [ch["kernel"]]),
            }
        )
    snapshot = {
        "use_tensor_product": True,
        "combine_mode": mode,
        "weights": weights,
        "scorer": getattr(scorer, "name", None),
        "channels": snapshot_channels,
        "random_seed": getattr(cfg, "SEED", None),
        "number_subsamples": getattr(cfg, "GRID_SEARCH_N_SAMPLES", None),
    }
    with open(config_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    print(f"  Config       -> {config_path}")


def _scalarise(val):
    """Collapse a swept value to something CSV/JSON-friendly."""
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, (np.ndarray, list, tuple)):
        return val[0] if len(val) == 1 else str(val)
    return val


def _jsonable(row: dict) -> dict:
    """Scalarise every value in a results row."""
    return {k: _scalarise(v) for k, v in row.items()}
