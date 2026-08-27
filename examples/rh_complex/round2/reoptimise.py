"""Round-2 step 1 — CKA-scored grid search supervised by round-1 energies.

Builds the CKA target from the round-1 DFT energies and runs a grid
search (single- or multi-channel, per ``USE_TENSOR_PRODUCT``) scoring
each SOAP × kernel combination by its agreement with the energies. The
ranking justifies the re-optimized representation recorded in
``config.ROUND2_KERNEL_CHANNELS``.

Outputs land under the round-2 namespace
(``output/<ROUND2_ANALYSIS_NAME>/grid_search/<hash>/``). Run with::

    python round2/reoptimise.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tastet.io import load_atoms_and_meta
from tastet.metrics.cka import CKAScorer
from tastet.pipeline import grid_search_step
from tastet.sweep.multichannel import grid_search_multichannel_step

# config.py and prepare.py live in the example root, one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg  # noqa: E402
from prepare import (  # noqa: E402
    ensure_database,
    resolve_soap_centers,
    resolve_channel_soap,
)
from _common import activate_round2, load_round1_energies, position_of  # noqa: E402


def cka_grid() -> None:
    """CKA-scored grid search using round-1 DFT energies as the target.

    Loads ``cfg.ENERGIES_CSV``, builds the target as
    ``E (Ha) − min(E (Ha))`` — offset purely for numerical safety, CKA
    is invariant to affine target transforms because the centered target
    kernel is unchanged — and dispatches to either
    :func:`tastet.pipeline.grid_search_step` (single-kernel mode) or
    :func:`tastet.sweep.multichannel.grid_search_multichannel_step`
    (multi-channel mode), passing a :class:`~tastet.metrics.cka.CKAScorer`
    parameterised by :attr:`config.CKA_TARGET_KERNEL`.

    The supervised sweep restricts itself to the labeled subset
    (round-1 conformers) so the heatmap reflects only what the energies
    can score.
    """
    activate_round2()
    ensure_database()
    atoms, meta = load_atoms_and_meta(cfg.db_path())

    e_df = load_round1_energies()
    pos_of = position_of(meta)

    e = e_df[cfg.ENERGY_COL].to_numpy(dtype=float)
    target_full = np.full(len(meta), np.nan)
    for cid, e_i in zip(e_df["configuration_id"], e):
        target_full[pos_of[int(cid)]] = e_i

    mask = ~np.isnan(target_full)
    labeled_idx = np.where(mask)[0]
    labeled_atoms = [atoms[i] for i in labeled_idx]
    labeled_ids = meta["configuration_id"].iloc[labeled_idx].to_numpy()
    target = target_full[mask]
    target = target - target.min()

    print(
        f"CKA target: {len(labeled_ids)} labeled conformers from {cfg.ENERGIES_CSV.name}"
    )

    scorer = CKAScorer(target_kernel=cfg.CKA_TARGET_KERNEL)

    if cfg._use_channels():
        grid_search_multichannel_step(
            cfg,
            labeled_atoms,
            labeled_ids,
            channels=cfg.KERNEL_CHANNELS,
            scorer=scorer,
            target=target,
            resolve_channel_soap=resolve_channel_soap,
        )
    else:
        centers = resolve_soap_centers(
            center_atoms=cfg.FIXED_SOAP_KW.get("center_atoms"),
        )
        fixed_kw = dict(cfg.FIXED_SOAP_KW)
        if centers is not None:
            fixed_kw["centers"] = centers
        grid_search_step(
            cfg=cfg,
            atoms_list=labeled_atoms,
            ids=labeled_ids,
            scorer=scorer,
            target=target,
            fixed_soap_kw=fixed_kw,
        )

    csv_path = cfg.grid_search_csv()
    if csv_path.exists():
        df = pd.read_csv(csv_path).sort_values(scorer.name, ascending=False).head(5)
        print(f"\nTop 5 (sorted by {scorer.name}):")
        print(df.to_string(index=False))


def main() -> None:
    """Run the CKA-scored grid search."""
    print(f"\n{'=' * 60}\n  Round 2: cka_grid\n{'=' * 60}")
    cka_grid()


if __name__ == "__main__":
    main()
