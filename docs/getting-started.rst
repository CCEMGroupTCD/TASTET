Getting Started
===============

Installation
------------

TASTET runs on **macOS** and **Linux** and requires **Python 3.10 or newer**.
We recommend installing it into a fresh virtual environment (``conda`` or
``venv``) so its dependencies stay isolated:

.. code-block:: text

   conda create -n tastet python=3.11
   conda activate tastet

TASTET is not on PyPI yet, so install it from source in editable mode:

.. code-block:: text

   git clone https://github.com/CCEMGroupTCD/TASTET.git
   cd TASTET
   pip install -e .

This pulls in the core scientific stack automatically (NumPy, ASE, DScribe,
scikit-learn, …). Optional extras add tooling for specific tasks:

.. code-block:: text

   pip install -e ".[examples]"   # run the bundled examples (adds RDKit)
   pip install -e ".[docs]"       # build this documentation (adds Sphinx)
   pip install -e ".[dev]"        # linting and tests (adds ruff, pytest)

Verify the installation
-----------------------

Check that TASTET imports and reports its version:

.. code-block:: text

   python -c "import tastet; print(tastet.__version__)"

This should print the installed version number. If this command displays any
errors, please open an issue on our
`issue tracker <https://github.com/CCEMGroupTCD/TASTET/issues>`_ so we can help
and fix it.

Quick Example
-------------

.. code-block:: python

   import pandas as pd
   from tastet.soap_utils import compute_soap
   from tastet.kernel import compute_kernel
   from tastet.selection import select_structures

   # 1. Compute SOAP descriptors
   soap_list = compute_soap(atoms_list, r_cut=4.0, sigma=0.1,
                            n_max=8, l_max=4, center_atoms=["Cu"])

   # 2. Build kernel matrix
   K = compute_kernel(soap_list, method="rematch", metric="linear", alpha=0.5)

   # 3. Select k diverse representatives directly from the kernel
   meta = pd.DataFrame({"configuration_id": range(1, len(atoms_list) + 1)})
   selected, idx_pool, selected_indices = select_structures(
       K, meta, k=10, method="fps")

   # selected_indices are the row positions of the chosen structures
   print(selected_indices)

:func:`~tastet.selection.select_structures` works in memory on the kernel
matrix alone — it needs no ASE database and writes nothing.  Here *meta* is a
minimal per-structure table aligned to the kernel rows; in a real campaign it
would carry energies (for the optional ``energy_max`` filter) and any other
metadata.  The :doc:`use-cases/index` pipelines wrap this call with the
database bookkeeping (recording the selected ``configuration_id``\ s, exporting
structures, and plotting).

Use Cases
---------

See :doc:`use-cases/index` for complete worked examples.