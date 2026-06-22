Use Cases
=========

TASTET ships with two complete, self-contained example pipelines under
``examples/``.  Each is driven by three top-level scripts — ``config.py``
(hyperparameters and path helpers), ``prepare.py`` (data ingestion), and
``run.py`` (the CLI pipeline) — and shares the same six-step command
structure: ``db``, ``grid_search``, ``soap``, ``kernel``, ``kpca``,
``select``.

Both examples need the ``examples`` extra (installs RDKit)::

   pip install -e ".[examples]"

.. toctree::
   :maxdepth: 1

   nanoclusters
   rh_complex
