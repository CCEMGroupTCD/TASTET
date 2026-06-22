Getting Started
===============

Installation
------------

Clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/youruser/tastet.git
   cd tastet
   pip install -e .

To run the bundled examples, add the ``examples`` extra — it installs
RDKit, used by the Rh-complex example to read its SDF conformer library:

.. code-block:: bash

   pip install -e ".[examples]"

Each example is then driven by ``python run.py <step>``; see
:doc:`use-cases/index` for the full walkthroughs.

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