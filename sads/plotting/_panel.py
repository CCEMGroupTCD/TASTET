"""Shared panel-title helpers for grid-search plots.

Lifted from :mod:`sads.plotting.distance` so that
:mod:`sads.plotting.heatmap` and any future grid-search plot use the
same compact LaTeX formatting:

* kernel formula expressed as a math expression
  (``$p_i \\cdot q_j$``, ``$(p_i \\cdot q_j)^d$``, ``$\\exp(-\\gamma\\,\\|p_i - q_j\\|^2)$``);
* SOAP knobs rendered with their conventional symbols
  (``$\\sigma$=0.1``, ``$r_{\\mathrm{cut}}$=4``);
* multi-channel panels stacked across lines joined by ``$\\otimes$``.
"""

from __future__ import annotations

from collections import OrderedDict


# Keys that belong to the kernel specification (everything else is a
# SOAP sweep parameter and shown separately).
KERNEL_KEYS = frozenset({
    "method", "metric", "alpha", "gamma", "degree", "coef0", "threshold",
})


def kernel_formula(params: dict) -> str:
    """Compact LaTeX formula for the local-environment kernel metric.

    :param params: Kernel parameter dict (``method``, ``metric``,
        and any metric-specific knobs like ``gamma`` / ``degree``).
    :returns: LaTeX string suitable for matplotlib titles.
    """
    metric = params.get("metric", "linear")
    dot = r"p_i \!\cdot\! q_j"

    if metric == "linear":
        return rf"${dot}$"

    if metric == "polynomial":
        d = params.get("degree", 2)
        g = params.get("gamma", 1.0)
        c = params.get("coef0", 0.0)
        inner = dot if g == 1.0 else rf"{g:g}\,{dot}"
        if c != 0.0:
            inner = rf"{inner} + {c:g}"
        if d == 1:
            return rf"${inner}$"
        return rf"$({inner})^{{{d}}}$"

    if metric == "rbf":
        g = params.get("gamma", 1.0)
        return rf"$\exp(-{g:g}\,\|p_i - q_j\|^2)$"

    return metric


def method_label(params: dict) -> str:
    """Compact label for the global kernel method (``Avg`` or ``RE``).

    :param params: Kernel parameter dict.
    :returns: Short string for the panel-title prefix.
    """
    method = params.get("method", "average")
    if method == "average":
        return "Avg"
    if method == "rematch":
        a = params.get("alpha", 0.5)
        return rf"RE $\alpha$={a:g}"
    return method


def soap_label(params: dict) -> str:
    """Format swept SOAP parameters (anything not in :data:`KERNEL_KEYS`).

    :param params: Mixed parameter dict.
    :returns: Comma-separated string of SOAP knobs in conventional
        notation, or empty when no SOAP knobs are present.
    """
    parts = []
    for k, v in params.items():
        if k in KERNEL_KEYS:
            continue
        val = f"{v:g}" if isinstance(v, float) else str(v)
        if k == "sigma":
            parts.append(rf"$\sigma$={val}")
        elif k == "r_cut":
            parts.append(rf"$r_{{\mathrm{{cut}}}}$={val}")
        elif k == "n_max":
            parts.append(rf"$n_{{\max}}$={val}")
        elif k == "l_max":
            parts.append(rf"$l_{{\max}}$={val}")
        else:
            parts.append(f"{k}={val}")
    return ",  ".join(parts)


def panel_title(params: dict) -> str:
    """Build a compact panel title from single- or multi-channel params.

    Single-kernel params have bare keys (``method``, ``sigma``, ...).
    Multi-channel params are prefixed with ``channel_name__key``.
    Both forms are detected automatically.

    :param params: Parameter dict for the panel.
    :returns: LaTeX-formatted multi-line title string.
    """
    if any("__" in k for k in params):
        return _multichannel_title(params)

    parts = [method_label(params), kernel_formula(params)]
    soap = soap_label(params)
    if soap:
        parts.append(soap)
    return ",  ".join(parts)


def _multichannel_title(params: dict) -> str:
    r"""Build a title for multi-channel (tensor-product) grid panels.

    Groups ``channel__key`` entries by channel, builds a compact
    label per channel, and stacks them on separate lines joined by
    ``$\otimes$``.

    :param params: Parameter dict whose keys are
        ``channel_name__key``.
    :returns: Multi-line LaTeX title string.
    """
    channels: OrderedDict[str, dict] = OrderedDict()
    for full_key, val in params.items():
        if "__" not in full_key:
            continue
        ch_name, key = full_key.split("__", 1)
        channels.setdefault(ch_name, {})[key] = val

    parts = []
    for ch_name, ch_params in channels.items():
        label = method_label(ch_params)
        formula = kernel_formula(ch_params)
        soap = soap_label(ch_params)
        ch_parts = [label, formula]
        if soap:
            ch_parts.append(soap)
        parts.append(f"{ch_name}: {', '.join(ch_parts)}")

    return r"$\otimes$".join(f"\n{p}\n" for p in parts).strip()