"""Distil the nanoclusters CKA grid search into a four-panel figure.

The pipeline heatmap (``output/production/grid_search/<hash>/heatmaps.png``)
has one panel per combination of the non-axis SOAP/kernel knobs — far too
many panels for a paper figure. This script reads the same ``results.csv``
and renders just four panels, chosen for spread rather than at random:

* the panel holding the single **highest**-CKA cell;
* the panel holding the single **lowest**-CKA cell;
* two intermediate panels whose *mean* CKA is closest to evenly spaced
  targets across the range of panel means, so the four panels differ in
  overall colour instead of clustering at one end of the scale.

Axes and the shared colour scale follow the pipeline heatmap (this example
runs in single-kernel mode, so the axes are two SOAP knobs); the colour
limits span the **whole** grid-search CKA range so the two extreme panels
show the true global maximum and minimum. Panel titles use the compact
:math:`K^{\\mathrm{method}}_{\\mathrm{metric}}(\\dots)` notation, and
styling follows :mod:`tastet.plotting.heatmap` / :mod:`tastet.plotting.style`.

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

# SOAP knobs that may appear in a panel title, with display order and symbol.
_SOAP_SYMBOLS: dict[str, str] = {
    "sigma": r"\sigma",
    "n_max": r"n_{\max}",
    "l_max": r"l_{\max}",
    "r_cut": r"r_{\mathrm{cut}}",
}


def _kernel_symbol(method: str, metric: str, soap: dict[str, object]) -> str:
    """Compact ``K`` notation for the kernel (no ``$`` delimiters).

    Renders as :math:`K^{\\mathrm{method}}_{\\mathrm{metric}}(\\dots)`,
    e.g. ``K^{\\mathrm{Average}}_{\\mathrm{Linear}}(\\sigma=0.1, l_{\\max}=4)``.
    The two SOAP knobs mapped to the heatmap axes are not in *soap* and
    so are omitted from the argument.

    :param method: Global kernel method (``average`` / ``rematch``).
    :param metric: Local-environment metric (``linear`` / ``rbf``).
    :param soap: Remaining swept SOAP knobs distinguishing the panel.
    :returns: A mathtext fragment ready to be wrapped in ``$...$``.
    """
    m = _METHOD_NAMES.get(method, method)
    k = _METRIC_NAMES.get(metric, metric)
    ordered = [key for key in _SOAP_SYMBOLS if key in soap]
    ordered += [key for key in soap if key not in _SOAP_SYMBOLS]
    args = ", ".join(
        rf"{_SOAP_SYMBOLS.get(key, key)} = {soap[key]:g}"
        if isinstance(soap[key], (int, float))
        else rf"{_SOAP_SYMBOLS.get(key, key)} = {soap[key]}"
        for key in ordered
    )
    return rf"K^{{\mathrm{{{m}}}}}_{{\mathrm{{{k}}}}}({args})"


def _panel_title(params: dict[str, object]) -> str:
    """Build a panel title from one panel's varying parameters.

    :param params: This panel's parameters (``method``, ``metric`` and
        the non-axis SOAP knobs).
    :returns: A complete ``$...$`` mathtext title string.
    """
    soap = {k: v for k, v in params.items() if k not in ("method", "metric")}
    return "$" + _kernel_symbol(params["method"], params["metric"], soap) + "$"


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
    )

    # Replace the default titles with the compact K notation. The panel
    # axes are the imshow axes (the colorbar axis has none), in the same
    # groupby order plot_grid_heatmaps used to fill them.
    panel_axes = [ax for ax in fig.axes if ax.images]
    group_keys = list(sub.groupby(group_by, dropna=False).groups.keys())
    for ax, key in zip(panel_axes, group_keys):
        key_tuple = key if isinstance(key, tuple) else (key,)
        params = dict(zip(group_by, key_tuple))
        ax.set_title(_panel_title(params), fontsize=10, pad=6)

    savefig(fig, out_path, dpi=300, also_pdf=True)
    plt.close(fig)
    print(f"  figure      -> {out_path}")


def main() -> None:
    """Render a four-panel CKA figure for each production grid search.

    Discovers every ``grid_search/<hash>/results.csv`` under the
    production output namespace and writes one ``<hash>.png`` (plus
    ``.pdf``) into ``output/production/cka_heatmaps/``.
    """
    grid_root = cfg.analysis_dir() / "grid_search"
    if not grid_root.exists():
        sys.exit(f"No grid_search root: {grid_root}")

    grid_dirs = sorted(
        d for d in grid_root.iterdir() if d.is_dir() and (d / "results.csv").exists()
    )
    if not grid_dirs:
        sys.exit(f"No grid-search directories with results.csv under {grid_root}")

    out_dir = cfg.analysis_dir() / "cka_heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(grid_dirs)} production grid-search directories.")
    for grid_dir in grid_dirs:
        print(f"\n[{grid_dir.name}] reading results.csv")
        _render(grid_dir / "results.csv", out_dir / f"{grid_dir.name}.png")


if __name__ == "__main__":
    main()
