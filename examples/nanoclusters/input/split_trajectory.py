"""Split the raw GOFFE trajectory into per-run trajectories.

The committed source of truth is ``input/all_runs.traj`` — the
concatenation of every GOFFE global-optimisation run.  This script
detects run boundaries from energy spikes (each new run restarts from a
high-energy configuration), then writes one flat trajectory per run,
``input/<run_name>.traj``, which :mod:`prepare` reads to build the
database.  It also saves the energy-vs-run profile figure used in the
publication.

Run this once before the pipeline::

    python input/split_trajectory.py
    python run.py db subsample grid_search soap kernel kpca select
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import read, write

from tastet.plotting.style import apply_axis_style, palette, savefig, set_mpl_style

# config.py lives in the example root, one level up from input/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg  # noqa: E402

# A jump larger than this between consecutive frames signals a new run.
SPIKE_THRESHOLD: float = 5.0  # eV
# Minimum number of frames required between two run boundaries.
MIN_RUN_LENGTH: int = 1000
# All committed runs are single-layer slabs.
N_LAYERS: int = 1


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

    Each run occupies one equal-width interval on the x-axis, labelled
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


def split_and_save(images: list[Atoms], boundaries: list[int]) -> None:
    """Write one flat trajectory per detected run into ``cfg.RUNS_DIR``.

    Run names follow ``run_<id>_n<frames>_<layers>L`` (the ``<layers>L``
    suffix is provenance only) and match the entries of
    ``cfg.TARGET_RUNS``.

    :param images: All structures from the raw trajectory.
    :param boundaries: Run start indices from :func:`detect_run_boundaries`.
    """
    cfg.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    boundaries_ext: list[int] = boundaries + [len(images)]

    for run_id in range(len(boundaries)):
        start: int = boundaries_ext[run_id]
        end: int = boundaries_ext[run_id + 1]
        run_images: list[Atoms] = images[start:end]

        run_name: str = f"run_{run_id:03d}_n{len(run_images)}_{N_LAYERS}L"
        out_path: Path = cfg.RUNS_DIR / f"{run_name}.traj"
        write(str(out_path), run_images)
        print(
            f"  run {run_id:3d}: [{start:5d}:{end:5d}] "
            f"({len(run_images):4d} frames) -> {out_path.name}"
        )

    print(f"\n  {len(boundaries)} per-run trajectories saved to {cfg.RUNS_DIR}/")


def save_summary_csv(
    images: list[Atoms], energies: np.ndarray, boundaries: list[int]
) -> None:
    """Write a provenance CSV of the per-frame run assignment.

    :param images: All structures from the raw trajectory.
    :param energies: Per-frame potential energies.
    :param boundaries: Run start indices from :func:`detect_run_boundaries`.
    """
    boundaries_ext: list[int] = boundaries + [len(images)]
    cfg.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path: Path = cfg.OUTPUT_ROOT / "run_summary.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_index", "run_id", "n_layers", "energy_eV", "n_cu"])
        for run_id in range(len(boundaries)):
            start: int = boundaries_ext[run_id]
            end: int = boundaries_ext[run_id + 1]
            for i in range(start, end):
                n_cu: int = images[i].get_chemical_symbols().count("Cu")
                writer.writerow([i, run_id, N_LAYERS, f"{energies[i]:.6f}", n_cu])
    print(f"  Run summary written to {csv_path}")


def main() -> None:
    """Split the raw trajectory and produce the energy-vs-run figure."""
    images = load_trajectory()
    energies = get_energies(images)

    boundaries = detect_run_boundaries(energies, SPIKE_THRESHOLD)
    plot_energy_profile(energies, boundaries)
    save_summary_csv(images, energies, boundaries)
    split_and_save(images, boundaries)


if __name__ == "__main__":
    main()
