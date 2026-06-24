API Reference
=============

TASTET does three things: it **parses** input structures into an ASE database,
**represents** them in a similarity kernel space encoded by SOAP descriptors,
and **selects** a space-filling subset of structures in that space. Two
secondary tools support this workflow — kernel-PCA visualization and
hyperparameter grid search.

Parsing structures into databases
---------------------------------

.. toctree::
   :maxdepth: 1

   io

Representing structures: SOAP and kernels
-----------------------------------------

.. toctree::
   :maxdepth: 1

   soap
   kernel

Selecting structures
--------------------

.. toctree::
   :maxdepth: 1

   selection

Visualizing kernel space with kernel PCA
----------------------------------------

.. toctree::
   :maxdepth: 1

   kpca

Hyperparameter optimization
---------------------------

The grid search tunes the SOAP and kernel representation, scored either without
labels (the kernel-induced distance spread) or against a target property
(centered kernel alignment).

.. toctree::
   :maxdepth: 1

   sweep
   distance
   cka
   metrics

Pipeline
--------

.. toctree::
   :maxdepth: 1

   pipeline