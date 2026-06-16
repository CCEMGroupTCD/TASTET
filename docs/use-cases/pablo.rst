Cu Clusters on Surface
======================

This use case analyses Cu cluster adsorption on a metal surface,
selecting representative low-energy structures for DFT relaxation.

Pipeline Overview
-----------------

.. code-block:: bash

   # 1. Build the master database from trajectory files
   python run.py db

   # 2. (Optional) Subsample for faster grid search
   python run.py subsample

   # 3. (Optional) Sweep SOAP × kernel hyperparameters
   python run.py grid_search

   # 4. Compute SOAP, kernel, and kPCA with chosen parameters
   python run.py soap kernel kpca

   # 5. Select representative structures
   python run.py select

   # 6. Export as POSCAR files for DFT
   python export_poscars.py

Configuration
-------------

All settings live in ``config.py``, split into two sections:

**General settings** (reusable across use cases): paths, SOAP/kernel
parameters, grid search grids, scorer choice, and selection parameters.

**Use-case-specific settings**: reference energies for formation energy
computation, trajectory run directories, and directory naming helpers.

Structure Selection
-------------------

Two sampling strategies are available, configured via
``SELECTION_METHOD`` in ``config.py``:

- **kmedoids** — clusters the filtered structures in kernel space and
  picks the medoid of each cluster.  Good when you want uniform
  coverage of distinct structural motifs.

- **fps** (furthest point sampling) — greedily picks the structure
  most distant from the current selection.  Good when you want maximum
  diversity across the entire space.

Both operate on the full kernel matrix (not the 2D kPCA projection),
preserving all structural similarity information.

Output Tree
-----------

.. code-block:: text

   output/<ANALYSIS_NAME>/
   ├── structures.db
   ├── structures.csv
   ├── grid_search/<hash>/
   │   ├── results.csv
   │   ├── heatmaps.png
   │   └── config.json
   └── <soap_tag>/
       └── <kernel_tag>/
           ├── kernel.npz
           ├── kpca.png
           ├── kpca_projections.csv
           ├── kpca_meta.json
           └── selection/
               ├── selected_structures.csv
               ├── selection.png
               └── poscars/