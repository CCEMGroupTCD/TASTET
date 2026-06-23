"""Plot the energy-vs-run profile of the raw GOFFE trajectory.

The committed source of truth is ``input/all_runs.traj`` — the
concatenation of every GOFFE global-optimization run.  This optional,
publication-only script detects run boundaries from energy spikes (each
new run restarts from a high-energy configuration) and saves the
energy-vs-run profile figure used in the article.  It is **not** part of
the pipeline: the ``db`` step builds the database directly from
``all_runs.traj`` and does not require this script.

Run with::

    python analysis/plot_run_energy_profile.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import read

from tastet.plotting.style import apply_axis_style, palette, savefig, set_mpl_style

# config.py lives in the example root, one level up from analysis/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg  # noqa: E402

# A jump larger than this between consecutive frames signals a new run.
SPIKE_THRESHOLD: float = 5.0  # eV
# Minimum number of frames required between two run boundaries.
MIN_RUN_LENGTH: int = 1000


def load_trajectory() -> list[Atoms]:
    """Read every frame of the raw concatenated trajectory.

    :returns: All structures in ``cfg.ALL_RUNS_TRAJ``, in file order.
    :raises SystemExit: If the trajectory file is missing.
    """
    if not cfg.ALL_RUNS_TRAJ.exists():
        sys.exit(f"Raw trajectory not found: {cfg.ALL_RUNS_TRAJ}")
    print(f"Reading {cfg.ALL_RUNS_TRAJ} ...")
    images: list[Atoms] = read(str(cfg.ALL_RUNS_TRAJ), index=":")
    print(f"  {len(images)} images loaded.")
    return images


def get_energies(images: list[Atoms]) -> np.ndarray:
    """Extract the potential energy of every frame.

    :param images: Structures read from the trajectory.
    :returns: Array of potential energies, one per frame.
    """
    return np.array([atoms.get_potential_energy() for atoms in images])


def detect_run_boundaries(energies: np.ndarray, threshold: float) -> list[int]:
    """Locate run start indices from energy spikes.

    Each new GOFFE run restarts from a high-energy configuration, so a
    frame-to-frame jump greater than *threshold* marks a boundary.
    Spikes closer than :data:`MIN_RUN_LENGTH` to the previous boundary
    are ignored (intra-run fluctuations).

    :param energies: Per-frame potential energies.
    :param threshold: Minimum energy jump (eV) that marks a new run.
    :returns: Sorted list of 0-based run start indices (always incl. 0).
    """
    boundaries: list[int] = [0]
    diffs: np.ndarray = np.diff(energies)
    last_boundary: int = 0

    for i in range(1, len(energies)):
        if diffs[i - 1] > threshold:
            if i - last_boundary >= MIN_RUN_LENGTH:
                boundaries.append(i)
                last_boundary = i
            else:
                print(
                    f"  Spike at {i} ignored (only {i - last_boundary} "
                    f"steps since last boundary at {last_boundary})"
                )

    boundaries = sorted(set(boundaries))
    print(f"  {len(boundaries)} run boundaries detected: {boundaries}")
    return boundaries


def plot_energy_profile(energies: np.ndarray, boundaries: list[int]) -> None:
    """Plot ``E - E_gm`` across the detected runs and save the figure.

    Each run occupies one equal-width interval on the x-axis, labeled
    by integer run number.  The figure is written to
    ``cfg.OUTPUT_ROOT / "energy_profile_runs.png"``.

    :param energies: Per-frame potential energies.
    :param boundaries: Run start indices from :func:`detect_run_boundaries`.
    """
    set_mpl_style(base_fontsize=12)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))

    relative_energies = energies - np.min(energies)

    run_starts = sorted({b for b in boundaries if 0 <= b < len(relative_energies)})
    if not run_starts or run_starts[0] != 0:
        run_starts.insert(0, 0)
    boundaries_ext = run_starts + [len(relative_energies)]
    n_runs = len(run_starts)

    # Put each detected run into one equal-width x-axis interval,
    # centered at integer run labels: 1, 2, ..., n_runs.
    run_positions = np.empty(len(relative_energies), dtype=float)
    for run_id, (start, end) in enumerate(
        zip(boundaries_ext[:-1], boundaries_ext[1:]), start=1
    ):
        run_length = end - start
        if run_length <= 0:
            continue
        run_positions[start:end] = (
            run_id - 0.5 + (np.arange(run_length) + 0.5) / run_length
        )

    ax.plot(
        run_positions,
        relative_energies,
        lw=1.1,
        color=palette["blue"],
        solid_capstyle="round",
    )

    for run_id in range(1, n_runs):
        ax.axvline(
            run_id + 0.5,
            color=palette["orange"],
            ls="--",
            lw=1.0,
            alpha=0.65,
            zorder=0,
        )

    ax.set_xlim(0.5, n_runs + 0.5)
    ax.set_xticks(np.arange(1, n_runs + 1))
    ax.set_xlabel("GOFFE run")
    ax.set_ylabel(r"$E - E_{\mathrm{gm}}$ (eV)")
    ax.margins(x=0.0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    apply_axis_style(ax, xfmt="%.0f", yfmt="%.2f")

    fig.tight_layout()
    cfg.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = cfg.OUTPUT_ROOT / "energy_profile_runs.png"
    savefig(fig, out_path)
    print(f"  Energy profile saved to {out_path}")
    if getattr(cfg, "SHOW", False):
        plt.show()
    plt.close(fig)


def main() -> None:
    """Produce the energy-vs-run profile figure from the raw trajectory."""
    images = load_trajectory()
    energies = get_energies(images)

    boundaries = detect_run_boundaries(energies, SPIKE_THRESHOLD)
    plot_energy_profile(energies, boundaries)


if __name__ == "__main__":
    main()
