"""Distil the round-2 CKA grid search into a four-panel publication figure.

The pipeline heatmap written by the round-2 grid search
(``output/round2/grid_search/<hash>/heatmaps.png``) has one panel per
combination of the non-axis kernel knobs — far too many panels for a
paper figure. This script reads the same ``results.csv`` and renders just
four panels, chosen for spread rather than at random:

* the panel holding the single **highest**-CKA cell;
* the panel holding the single **lowest**-CKA cell;
* two intermediate panels whose *mean* CKA is closest to evenly spaced
  targets across the range of panel means, so the four panels differ in
  overall colour instead of clustering at one end of the scale.

Axes (core vs. periphery :math:`r_{\\mathrm{cut}}`) and the shared colour
scale are taken from the full grid via
:func:`tastet.plotting.heatmap.infer_heatmap_layout`; the colour limits
span the **whole** grid-search CKA range so the two extreme panels show
the true global maximum and minimum. Styling follows
:mod:`tastet.plotting.heatmap` / :mod:`tastet.plotting.style`, matching
the kPCA heatmaps.

Run with::

    python analysis/plot_cka_heatmaps.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tastet.plotting.heatmap import (
    infer_heatmap_layout,
    plot_grid_heatmaps,
    select_spread_panels,
)
from tastet.plotting.style import savefig

# config.py lives in the example root, one level up from analysis/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg  # noqa: E402

# Score column produced by the CKA-scored grid search (see
# tastet.metrics.cka.CKAScorer.name); it is the only value column.
VALUE: str = "cka"

# Number of panels in the distilled figure.
N_PANELS: int = 4


# Display names for the kernel notation $K^{\mathrm{method}}_{\mathrm{metric}}$.
_METHOD_NAMES: dict[str, str] = {"average": "Average", "rematch": "REMatch"}
_METRIC_NAMES: dict[str, str] = {"linear": "Linear", "rbf": "RBF", "polynomial": "Poly"}


def _channel_short(prefix: str) -> str:
    """Shorten a channel prefix for display (``core_kernel`` → ``core``).

    :param prefix: Channel prefix as it appears before ``__`` in a
        results column (e.g. ``periphery_kernel``).
    :returns: The channel name with a trailing ``_kernel`` removed.
    """
    return prefix.removesuffix("_kernel")


def _rcut_label(col: str) -> str:
    """Axis label for a per-channel ``r_cut`` column.

    Distinguishes the two cutoff axes (which would otherwise both read
    ``r_cut``) by superscripting the channel, e.g.
    ``$r_{\\mathrm{cut}}^{\\mathrm{core}}$``.

    :param col: Column name of the form ``<channel>__r_cut``.
    :returns: LaTeX axis label carrying the channel name.
    """
    ch = _channel_short(col.split("__", 1)[0])
    return rf"$r_{{\mathrm{{cut}}}}^{{\mathrm{{{ch}}}}}$"


def _kernel_symbol(method: str, metric: str, sigma: float) -> str:
    """Compact ``K`` notation for one channel's kernel (no ``$`` delimiters).

    Renders as :math:`K^{\\mathrm{method}}_{\\mathrm{metric}}(\\sigma=\\dots)`,
    e.g. ``K^{\\mathrm{Average}}_{\\mathrm{Linear}}(\\sigma=0.1)``. The
    cutoff is omitted because it is mapped to the heatmap axes.

    :param method: Global kernel method (``average`` / ``rematch``).
    :param metric: Local-environment metric (``linear`` / ``rbf``).
    :param sigma: SOAP broadening for the channel.
    :returns: A mathtext fragment ready to be wrapped in ``$...$``.
    """
    m = _METHOD_NAMES.get(method, method)
    k = _METRIC_NAMES.get(metric, metric)
    return rf"K^{{\mathrm{{{m}}}}}_{{\mathrm{{{k}}}}}(\sigma = {sigma:g})"


def _panel_title(
    params: dict[str, object], channels: list[str], methods: dict[str, str]
) -> str:
    """Build a panel title joining each channel's ``K`` symbol with ``∘``.

    :param params: This panel's varying parameters, keyed
        ``<channel>__<knob>`` (metric and sigma per channel).
    :param channels: Ordered channel prefixes (core first, to match the
        x-axis).
    :param methods: Constant kernel method per channel (not part of
        *params* because it does not vary across panels).
    :returns: A complete ``$...$`` mathtext title string.
    """
    symbols = [
        _kernel_symbol(methods[ch], params[f"{ch}__metric"], params[f"{ch}__sigma"])
        for ch in channels
    ]
    return "$" + r" \circ ".join(symbols) + "$"


def _render(results_csv: Path, out_path: Path) -> None:
    """Render the four-panel CKA heatmap for one results file.

    :param results_csv: Path to a grid-search ``results.csv``.
    :param out_path: Destination ``.png`` (a sibling ``.pdf`` is also
        written for publication).
    """
    df = pd.read_csv(results_csv)
    x, y, group_by = infer_heatmap_layout(df, VALUE)

    keys = select_spread_panels(df, VALUE, group_by, n_panels=N_PANELS)
    sub = df[pd.MultiIndex.from_frame(df[group_by]).isin(keys)]
    sub_index = pd.MultiIndex.from_frame(sub[group_by])

    print(f"  selected {len(keys)} of {df.groupby(group_by).ngroups} panels")
    for k in keys:
        cells = sub.loc[sub_index.isin([k]), VALUE]
        print(
            f"    {dict(zip(group_by, k))}: "
            f"mean={cells.mean():.3f}  min={cells.min():.3f}  max={cells.max():.3f}"
        )

    # Channels (core first, matching the x-axis) and their constant
    # methods, which are not in group_by because they do not vary.
    channels: list[str] = []
    for col in group_by:
        prefix = col.split("__", 1)[0]
        if prefix not in channels:
            channels.append(prefix)
    methods = {ch: df[f"{ch}__method"].iloc[0] for ch in channels}

    # Colour limits from the full grid so the extreme panels show the
    # true global maximum and minimum.
    fig = plot_grid_heatmaps(
        sub,
        value=VALUE,
        x=x,
        y=y,
        group_by=group_by,
        out_path=None,
        vmin=float(df[VALUE].min()),
        vmax=float(df[VALUE].max()),
        n_cols=2,
        colorbar_label="CKA",
        label_map={x: _rcut_label(x), y: _rcut_label(y)},
    )

    # Replace the default titles with the compact K notation. The panel
    # axes are the imshow axes (the colorbar axis has none), in the same
    # groupby order plot_grid_heatmaps used to fill them.
    panel_axes = [ax for ax in fig.axes if ax.images]
    group_keys = list(sub.groupby(group_by, dropna=False).groups.keys())
    for ax, key in zip(panel_axes, group_keys):
        key_tuple = key if isinstance(key, tuple) else (key,)
        params = dict(zip(group_by, key_tuple))
        ax.set_title(_panel_title(params, channels, methods), fontsize=10, pad=6)

    savefig(fig, out_path, dpi=300, also_pdf=True)
    plt.close(fig)
    print(f"  figure      -> {out_path}")


def main() -> None:
    """Render a four-panel CKA figure for each round-2 grid search.

    Discovers every ``grid_search/<hash>/results.csv`` under the round-2
    output namespace and writes one ``<hash>.png`` (plus ``.pdf``) into
    ``output/round2/cka_heatmaps/``.
    """
    round2_root = cfg.OUTPUT_ROOT / cfg.ROUND2_ANALYSIS_NAME
    grid_root = round2_root / "grid_search"
    if not grid_root.exists():
        sys.exit(f"No round-2 grid_search root: {grid_root}")

    grid_dirs = sorted(
        d for d in grid_root.iterdir() if d.is_dir() and (d / "results.csv").exists()
    )
    if not grid_dirs:
        sys.exit(f"No grid-search directories with results.csv under {grid_root}")

    out_dir = round2_root / "cka_heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(grid_dirs)} round-2 grid-search directories.")
    for grid_dir in grid_dirs:
        print(f"\n[{grid_dir.name}] reading results.csv")
        _render(grid_dir / "results.csv", out_dir / f"{grid_dir.name}.png")


if __name__ == "__main__":
    main()
