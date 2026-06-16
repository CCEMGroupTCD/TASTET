TASTET — Tool for Atomistic Structure selection Through Efficient Triage
========================================================================

TASTET is a Python toolkit for analysing atomic structures through SOAP
descriptors and kernel methods.  It provides a modular pipeline:

1. Compute SOAP descriptors for a set of structures.
2. Build global similarity kernels (average or REMatch).
3. Evaluate kernel quality via grid search (CKA, dissimilarity).
4. Visualise structure space with kernel PCA.
5. Select representative structures for downstream calculations.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting-started
   api/index
   use-cases/index