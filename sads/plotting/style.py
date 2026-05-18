from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
from matplotlib.markers import MarkerStyle

palette = {
    "dark orange": "#D55E00",
    "orange":  "#E69F00",
    "yellow":  "#F0E442",
    "green":   "#019E74",
    "blue":    "#57B4E9",
    "dark blue": "#0072B2",
    "magenta": "#CC79A7",
    "pink": "#EE6677",
    "black":   "#000000",
}

cmap = LinearSegmentedColormap.from_list(
    "allowed_gradient",
    [palette["dark blue"], palette["blue"], palette["green"], palette["yellow"], palette["orange"]],
    N=256,
)

_MINUS = "\N{MINUS SIGN}"

def _make_minus_formatter(fmt: str = ".1f"):
    def _formatter(x, _pos=None) -> str:
        s = format(x, fmt)
        return s.replace("-", _MINUS)
    return _formatter

def set_mpl_style(
    font_family: str = "sans-serif",
    font_sans: tuple[str, ...] = ("Arial",),
    base_fontsize: int = 12,
    pdf_fonttype: int = 42,
) -> None:
    """Apply project-wide matplotlib rcParams."""
    plt.rcParams["font.family"] = font_family
    plt.rcParams["font.sans-serif"] = list(font_sans)
    plt.rcParams["pdf.fonttype"] = pdf_fonttype

    plt.rcParams.update({
        "font.size": base_fontsize,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize,
        "ytick.labelsize": base_fontsize,
        "legend.fontsize": base_fontsize,
        "axes.titlesize": base_fontsize,
    })

def apply_axis_style(
    ax,
    use_minor_x: bool = True,
    use_minor_y: bool = True,
    xfmt: str = "%.2f",
    yfmt: str = "%.2f",
) -> None:
    ax.tick_params(which="major", direction="out", top=False, right=False, length=5, width=1.0)
    ax.tick_params(which="minor", direction="out", length=3, width=1.0)

    if use_minor_x:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    if use_minor_y:
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Apply numeric format + long minus
    ax.xaxis.set_major_formatter(FuncFormatter(_make_minus_formatter(xfmt.replace("%", ""))))
    ax.yaxis.set_major_formatter(FuncFormatter(_make_minus_formatter(yfmt.replace("%", ""))))

    # Round caps for tick marks (major & minor)
    for axis in (ax.xaxis, ax.yaxis):
        majors = axis.get_ticklines()
        if majors:
            m = majors[0].get_marker()
            plt.setp(majors, marker=MarkerStyle(m, capstyle="round"))
        minors = axis.get_minorticklines()
        if minors:
            m0 = minors[0].get_marker()
            plt.setp(minors, marker=MarkerStyle(m0, capstyle="round"))

def savefig(fig, path: Path, dpi: int = 450) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # fig.savefig(path, dpi=dpi, transparent=False, bbox_inches="tight", pad_inches=0.0)
    fig.savefig(path, dpi=dpi, transparent=False, pad_inches=0.0)
