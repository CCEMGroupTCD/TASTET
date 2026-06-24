TASTET — Tool for Atomistic Structure selection Through Efficient Triage
========================================================================

TASTET is a Python toolkit for selecting representative structures from
molecular and materials datasets. At its core it does three things:

1. **Parse** input structures into an ASE database.
2. **Represent** each structure in a similarity kernel space encoded by SOAP
   descriptors.
3. **Select** a space-filling subset of structures for high-fidelity
   calculations.

Two secondary tools support this workflow: kernel-PCA **visualization** of the
kernel space, and unsupervised/supervised **hyperparameter grid searches** that
tune the SOAP and kernel representation.

TASTET is built on `ASE <https://wiki.fysik.dtu.dk/ase/>`_ (structures and
databases) and `DScribe <https://singroup.github.io/dscribe/>`_ (SOAP
descriptors).

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting-started
   api/index
   use-cases/index