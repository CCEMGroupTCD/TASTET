Cu Nanoclusters on a Surface
============================

This use case analyses copper nanoclusters adsorbed on a metal surface,
selecting a structurally diverse, energy-aware subset for DFT relaxation.
It runs in **single-kernel mode** (``USE_TENSOR_PRODUCT = False``).

The example lives in ``examples/nanoclusters/``.

Data Ingestion
--------------

The committed source of truth is ``input/all_runs.traj`` — the raw
concatenated GOFFE trajectory.  Before running the pipeline, split it into
flat per-run trajectories with the one-time ingestion script::

   python input/split_trajectory.py

This writes the (untracked) ``input/run_*.traj`` files and saves the
energy-vs-run publication figure.  DFT energies of the eventual FPS picks
are committed in ``input/energies_selected.csv``.

Pipeline
--------

.. code-block:: bash

   python run.py db                  # 1. build the database from the per-run trajectories
   python run.py grid_search         # 2. (optional) sweep SOAP × kernel on an energy-balanced subset
   python run.py soap kernel kpca    # 3. single-config pipeline on all structures
   python run.py select              # 4. select structures for DFT

The grid search draws an **energy-balanced** subset (inverse-density over
``NUM_BINS``, size ``GRID_SEARCH_N_SAMPLES``) so the abundant low-energy
basin does not swamp the search.

Configuration
-------------

All settings live in ``config.py``: SOAP and kernel parameters, the grid
search grids, the scorer choice, selection parameters, and the path
helpers that key every output directory.

Output Tree
-----------

Single-kernel mode places the production kernel under
``soap_dir() / kernel_tag()``:

.. code-block:: text

   output/production/
   ├── structures.db
   ├── grid_search/<hash>/
   │   ├── results.csv
   │   ├── distance_distributions.png
   │   └── distance_summary.csv
   └── <soap_tag>/                        # e.g. rcut4.0_sig0.1_n8_l4_c-Cu
       └── <kernel_tag>/                  # e.g. average_linear
           ├── kernel.npz
           ├── distance_distribution.png
           ├── kde_distance_distribution.png
           ├── pairwise_distances.csv
           ├── kpca.png
           ├── kpca_projections.csv
           ├── kpca_meta.json
           └── selection/
               ├── selected_structures.csv
               ├── selection.png
               └── xyz/

Post-hoc Analysis
-----------------

Scripts under ``analysis/`` (not part of the core pipeline):

- ``energy_profile.py`` — surrogate-vs-DFT validation of the selected picks.
- ``plot_subsample_distribution.py`` — full-vs-subsampled energy histograms.
- ``plot_cka_heatmaps.py`` — CKA heatmaps across the grid-search channels.
