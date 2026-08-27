Cu Nanoclusters on a ZnO Surface
================================

This use case analyzes :math:`\mathrm{Cu}_{12}` nanoclusters adsorbed on a
:math:`\mathrm{ZnO}(10\bar{1}0)` surface, selecting a structurally diverse
subset for DFT relaxation.  It runs in **single-kernel mode** — one
SOAP/kernel channel, rather than the tensor product used by the Rh catalyst
example.  It corresponds to the second case study in the accompanying
research article (DOI: ...).

This page shows how a complete pipeline is assembled from ``tastet``
modules.  The example is fully reproducible from the committed data, so it
is described here at a high level rather than line by line.  Readers who
want to build their own pipelines are encouraged to read the example's
Python files directly: they are meant as a template for how the individual
``tastet`` modules fit together.

The example lives in ``examples/nanoclusters/`` and is driven by three
Python files:

- ``config.py`` — the central configuration file; every tunable parameter
  and output path is defined here.  In normal use it is the only file you
  need to edit (see `Configuration`_ below).
- ``prepare.py`` — data ingestion: builds the ASE database directly from
  ``input/all_runs.traj``.
- ``run.py`` — the command-line driver that runs the pipeline steps
  (``db``, ``grid_search``, ``soap``, ``kernel``, ``kpca``, ``select``).

Configuration
-------------

Everything the pipeline does is controlled from ``config.py``, which is
organized top-to-bottom into labeled blocks.  The ones you are most likely
to touch are:

- **General settings** — output paths, the analysis name (which names the
  output namespace), the random ``SEED``, and the master
  ``USE_TENSOR_PRODUCT`` toggle that switches between single-kernel and
  tensor-product (multi-channel) mode.  This example uses single-kernel
  mode.
- **Single-kernel mode** — ``SOAP_PARAMS`` and ``KERNEL_PARAMS``, the lone
  SOAP/kernel channel used here (active because ``USE_TENSOR_PRODUCT =
  False``).
- **Grid search** — the hyperparameter sweep: the search space
  (``SOAP_GRID``, ``KERNEL_GRID``), the cap on combinations
  (``MAX_GRID_COMBINATIONS``), and the CKA scorer target
  (``CKA_TARGET_KERNEL``), which ranks each candidate representation
  against the surrogate-predicted energies.
- **Multi-channel kernel** — ``KERNEL_CHANNELS``, for tensor-product mode.
  Not used here.
- **Structure selection** — an optional energy pre-filter
  (``SELECTION_ENERGY_MAX``, relative to the lowest surrogate-predicted
  energy), how many
  representatives to pick (``SELECTION_K``), and with which algorithm
  (``SELECTION_METHOD``): Furthest Point Sampling (FPS) or k-medoids.
- **Use-case-specific** — settings unique to this system: the grid-search
  subsampling (``GRID_SEARCH_N_SAMPLES``, ``NUM_BINS``) and the input
  trajectory path (``ALL_RUNS_TRAJ``).
- **Path helpers** — functions that assemble the output directory tree from
  the settings above.  You should not need to edit these.

To adapt the example, edit the relevant block; the steps below read their
parameters from it.

Data Ingestion
--------------

Data ingestion is the most application-specific part of the workflow: it
depends entirely on the format and number of your input files.  It is
handled by ``prepare.py``, which is therefore the script you are most
likely to rewrite for a new dataset.

The committed source of truth is ``input/all_runs.traj`` — a dataset of
configurations produced by multiple independent runs of a machine-learning
surrogate-driven global optimization algorithm.  The ``db`` step builds the
database directly from ``input/all_runs.traj``; no pre-splitting is
required.  DFT energies of the FPS picks are committed in
``input/energies_selected.csv``.

The optional ``analysis/plot_run_energy_profile.py`` script reads the same
trajectory to produce the energy-vs-run publication figure; it detects run
boundaries from energy spikes but is not part of the pipeline.

Pipeline
--------

.. code-block:: text

   python run.py db                  # 1. build the database from input/all_runs.traj
   python run.py grid_search         # 2. (optional) hyperparameter sweep on a subsample of structures
   python run.py soap kernel kpca    # 3. single-config pipeline on all structures
   python run.py select              # 4. select structures for DFT

The grid search runs on an **energy-balanced** subset of the structures:
inverse-density sampling over ``NUM_BINS`` bins of the
**surrogate-predicted** energy, of size ``GRID_SEARCH_N_SAMPLES``,
so that the abundant low-energy basin does not swamp the search.  These are
the global-optimization surrogate's energies, not DFT — the only DFT
energies in this example are those of the final FPS picks
(``energies_selected.csv``).

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
           ├── kpca_3d.png
           ├── kpca_projections.csv
           ├── kpca_meta.json
           └── selection/
               ├── selected_structures.csv
               ├── selection.png
               ├── selection_3d.png
               ├── selection_full.png
               ├── xyz/
               └── poscars/

Selection Output
----------------

Each ``select`` step records its picks in several complementary forms:

- ``selected_structures.csv`` lists the chosen **configuration IDs** — the
  1-based, gap-free row indices into the ASE database (``structures.db``),
  so each ID points straight back at its structure.
- ``xyz/`` holds the picks as ``.xyz`` files, named via
  ``SELECTION_XYZ_TEMPLATE`` (e.g. ``structure_5.xyz``).  This is the
  pipeline's default export format, identical across every use case.
- ``poscars/`` holds the same picks as VASP ``POSCAR`` files
  (``POSCAR_<id>``), ready for a periodic DFT relaxation.  These are an
  add-on specific to this example (see the note below).

.. note::

   The pipeline always writes ``.xyz`` so that its output stays
   consistent across use cases.  If ``.xyz`` does not match the input
   format your quantum-chemistry code expects (e.g. VASP), you can
   convert any pick with ASE:

   .. code-block:: python

      from ase.io import read, write

      atoms = read("structure_5.xyz")
      write("POSCAR", atoms, format="vasp")   # any format ASE supports

   This example automates exactly that conversion for the periodic slab
   system: on top of the shared ``.xyz`` export, its ``select`` step
   (the ``_select`` function in ``examples/nanoclusters/run.py``) writes
   each pick as a VASP ``POSCAR`` into the ``poscars/`` directory above,
   so you do not have to convert them by hand.
