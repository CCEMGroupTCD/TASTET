TASTET — Tool for Atomistic Structure selection Through Efficient Triage
========================================================================

TASTET is a Python toolkit for the selection of representative structures
in molecular and materials datasets.  It provides a modular pipeline:

1. Compute SOAP descriptors for a set of structures.
2. Build global similarity kernels.
3. Optimize kernel representation via grid search.
4. Visualize kernel space with kernel PCA.
5. Select representative structures for high-fidelity calculations.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting-started
   api/index
   use-cases/index