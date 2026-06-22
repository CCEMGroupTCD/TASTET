Rh Catalyst Conformers
======================

This use case analyzes a library of conformers of a rhodium catalyst,
selecting representatives for DFT.  It runs in **tensor-product
(multi-channel) mode**, combining several SOAP/kernel channels into one
product kernel.  It corresponds to the first case study explained in the
accompanying research article (DOI: ...).

Selection proceeds in **two rounds** so that energy information can supervise
the kernel optimization.  Round 1 selects a diverse set blindly (no energies
yet) and sends it for DFT.  Round 2 then feeds those round-1 energies back in
to re-optimize the kernel channels — rewarding the representation that best
separates structures by energy — before selecting a second batch.

The example lives in ``examples/rh_complex/``.

Data Ingestion
--------------

The committed source of truth is ``input/open_babel_Rh_conformers.sdf``,
a conformer library read directly by ``prepare.ensure_database()``.  DFT
energies are committed input data, two columns ``file,E (Ha)`` each:

- ``input/energies_round1.csv`` — round-1 FPS picks.
- ``input/energies_round2.csv`` — the union of the three round-2 strategies'
  picks.

ΔE is always recomputed from ``E (Ha)``; round-2 plots reference the
study-wide found minimum so round-1 and round-2 share one zero.

Round 1
-------

.. code-block:: bash

   python run.py db                  # 1. build the master database from the SDF
   python run.py grid_search         # 2. (optional) sweep on a random subsample / all conformers
   python run.py soap kernel kpca    # 3. single-config pipeline on all conformers
   python run.py select              # 4. select representatives for DFT

The grid search draws a random subsample (or all conformers) sized by
``GRID_SEARCH_N_SAMPLES``.

Round 2 (supervised)
--------------------

The round-2 workflow lives under ``round2/`` and is activated at runtime by
``activate_round2()`` so ``config.py`` stays reproducible for round 1.  It
re-optimizes the kernel channels under supervision of the round-1 energies,
then re-selects with three strategies:

.. code-block:: bash

   python round2/reoptimise.py       # CKA-scored grid search supervised by round-1 energies
   python round2/reselect.py select          # FPS, truncated from TOTAL_BUDGET
   python round2/reselect.py zoom_select      # focus a kPCA region (ZOOM_BOX)
   python round2/reselect.py nearest_select   # center on the lowest-energy round-1 conformer

All three strategies yield the same number of round-2 picks (``ZOOM_K``)
for paper consistency.  Round 2 uses its own output namespace
(``ROUND2_ANALYSIS_NAME``) and re-optimized channels via the ``ROUND2_*``
config section.

Configuration
-------------

All settings live in ``config.py``, including a dedicated ``ROUND2_*``
section (``ROUND2_ANALYSIS_NAME``, ``ENERGIES_CSV``, ``ROUND2_ENERGIES_CSV``,
``TOTAL_BUDGET``, ``ZOOM_*``, ``ROUND2_KERNEL_CHANNELS``).

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
       ├── kpca_projections.csv
       ├── kpca_meta.json
       └── selection/
           ├── selected_structures.csv
           ├── selection.png
           └── xyz/
