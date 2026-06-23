Rh Catalyst Conformers
======================

This use case analyzes a library of conformers of a rhodium catalyst,
selecting representatives for DFT.  It runs in **tensor-product
(multi-channel) mode**, combining two SOAP/kernel channels into one
product kernel.  It corresponds to the first case study in the
accompanying research article (DOI: ...).

This page shows how a complete pipeline is assembled from ``tastet``
modules.  The example is fully reproducible from the committed data, so it
is described here at a high level rather than line by line.  Readers who
want to build their own pipelines are encouraged to read the example's
Python files directly: they are meant as a template for how the individual
``tastet`` modules fit together.

Selection proceeds in **two rounds** so that energies from the first round
can supervise the kernel optimization in the second.  Round 1 selects a
diverse set blindly — with no energy information — and submits it for DFT.
Round 2 then feeds those round-1 energies into a second, supervised grid
search that re-optimizes the kernel channels, rewarding the representation
whose kernel best aligns with the energy ordering, before selecting a
second batch.

The example lives in ``examples/rh_complex/`` and is driven by three Python
files:

- ``config.py`` — the central configuration file; every tunable parameter
  and output path is defined here.  In normal use it is the only file you
  need to edit (see `Configuration`_ below).
- ``prepare.py`` — data ingestion: reads the input SDF and energy CSVs and
  builds the ASE database.
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
  tensor-product (multi-channel) mode.  This example uses tensor-product
  mode.
- **Single-kernel mode** — ``SOAP_PARAMS`` and ``KERNEL_PARAMS``, the lone
  SOAP/kernel channel used when ``USE_TENSOR_PRODUCT = False``.  Not used
  here.
- **Grid search** — the hyperparameter sweep: the search space
  (``SOAP_GRID``, ``KERNEL_GRID``), the cap on combinations
  (``MAX_GRID_COMBINATIONS``), and ``GRID_SEARCH_N_SAMPLES``, the number of
  conformers the sweep runs on (``None`` = all conformers).
- **Multi-channel kernel** — ``KERNEL_CHANNELS``, the list of SOAP/kernel
  channels combined into the product kernel (``KERNEL_COMBINE``).
- **Structure selection** — how many representatives to pick
  (``SELECTION_K``) and with which algorithm (``SELECTION_METHOD``):
  Furthest Point Sampling (FPS) or k-medoids.
- **Round 2** — the supervised re-optimization and re-selection settings
  (the ``ROUND2_*`` and ``ZOOM_*`` variables, the energy-CSV paths, and
  ``TOTAL_BUDGET``).  These are described under `Round 2 (supervised)`_.
- **Use-case-specific** — settings unique to this system, such as the input
  SDF path.
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

Here the dataset is ``input/open_babel_Rh_conformers.sdf``, a conformer
library read directly by ``prepare.ensure_database()``.  The DFT energies
are committed input data, each a two-column ``file,E (Ha)`` CSV:

- ``input/energies_round1.csv`` — round-1 FPS picks.
- ``input/energies_round2.csv`` — the union of the three round-2
  strategies' picks.

ΔE is always recomputed from ``E (Ha)``; round-2 plots reference the
study-wide found minimum, so round 1 and round 2 share one zero.

Round 1 (unsupervised)
----------------------

.. code-block:: text

   python run.py db                  # 1. build the master database from the SDF
   python run.py grid_search         # 2. (optional) hyperparameter sweep on a subsample / all conformers
   python run.py soap kernel kpca    # 3. single-config pipeline on all conformers
   python run.py select              # 4. select representatives for DFT

The grid search runs on a random subsample of the conformers (or all of
them) sized by ``GRID_SEARCH_N_SAMPLES``.

Round 2 (supervised)
--------------------

The round-2 workflow lives under ``round2/`` and is activated at runtime by
``activate_round2()`` so ``config.py`` stays reproducible for round 1.  It
re-optimizes the kernel channels under supervision of the round-1 energies,
then re-selects with three strategies:

.. code-block:: text

   python round2/reoptimise.py       # CKA-scored grid search supervised by round-1 energies
   python round2/reselect.py select          # FPS, truncated from TOTAL_BUDGET
   python round2/reselect.py zoom_select      # focus a kPCA region (ZOOM_BOX)
   python round2/reselect.py nearest_select   # center on the lowest-energy round-1 conformer

All three strategies yield the same number of round-2 picks, given by ``ZOOM_K``.  Round 2 uses its own output namespace
(``ROUND2_ANALYSIS_NAME``) and re-optimized channels via the ``ROUND2_*``
config section.

Output Tree
-----------

Tensor-product mode places the production kernel under a hash-keyed
``product_<hash>/`` directory:

.. code-block:: text

   output/production/
   ├── structures.db
   ├── grid_search/<hash>/
   │   ├── results.csv
   │   ├── distance_distributions.png
   │   └── distance_summary.csv
   ├── channels/<name>/<soap_tag>/<kernel_tag>/   # per-channel SOAP + kernel caches
   └── product_<hash>/
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
           └── xyz/

Selection Output
----------------

Each ``select`` step records its picks in two complementary forms:

- ``selected_structures.csv`` lists the chosen **configuration IDs** — the
  1-based, gap-free row indices into the ASE database (``structures.db``),
  so each ID points straight back at its conformer.
- ``xyz/`` holds the same structures exported as ``.xyz`` files, named via
  ``SELECTION_XYZ_TEMPLATE`` (e.g. ``conformer_5.xyz``).

.. note::

   If ``.xyz`` does not match the input format your quantum-chemistry code
   expects (e.g. VASP), you can convert it with ASE:

   .. code-block:: python

      from ase.io import read, write

      atoms = read("conformer_5.xyz")
      write("POSCAR", atoms, format="vasp")
