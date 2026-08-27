Use Cases
=========

TASTET ships with two complete, self-contained example pipelines under
``examples/``.  Each is driven by three top-level scripts — ``config.py``
(hyperparameters and path helpers), ``prepare.py`` (data ingestion), and
``run.py`` (the CLI pipeline) — and shares the same six-step command
structure:

- ``db`` — build the master ASE database from the raw input data.
- ``grid_search`` — sweep SOAP/kernel hyperparameters and rank them by their
  kernel-induced distance distributions.
- ``soap`` — compute SOAP descriptors for every structure.
- ``kernel`` — build the global similarity kernel from those descriptors.
- ``kpca`` — project the kernel into 2-D with kernel PCA for visualization.
- ``select`` — pick a diverse subset of representative structures for DFT.

.. toctree::
   :maxdepth: 1

   rh_complex
   nanoclusters
