Cu Nanoclusters on a ZnO Surface
================================

This use case analyzes :math:`\mathrm{Cu}_{12}` nanoclusters adsorbed on a
:math:`\mathrm{ZnO}(10\bar{1}0)` surface, selecting a structurally diverse
subset for DFT relaxation.  It corresponds to the second case study explained
in the accompanying research article (DOI: ...).

The example lives in ``examples/nanoclusters/``.

Data Ingestion
--------------

The committed source of truth is ``input/all_runs.traj`` — a dataset of
configurations produced by multiple independent runs of a machine-learning
surrogate-driven global optimization algorithm.  Before running the pipeline,
split it into flat per-run trajectories with the one-time ingestion script::

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
