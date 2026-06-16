Getting Started
===============

Installation
------------

Clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/youruser/tastet.git
   cd tastet
   pip install -e ".[docs]"

The ``[docs]`` extra installs Sphinx and the ReadTheDocs theme so you
can build documentation locally.

Quick Example
-------------

.. code-block:: python

   from tastet.soap_utils import compute_soap
   from tastet.kernel import compute_kernel
   from tastet.kpca import fit_kpca
   from tastet.plotting import plot_kpca

   # 1. Compute SOAP descriptors
   soap_list = compute_soap(atoms_list, r_cut=4.0, sigma=0.1,
                            n_max=8, l_max=4, center_atoms=["Cu"])

   # 2. Build kernel matrix
   K = compute_kernel(soap_list, method="average", metric="linear")

   # 3. Run kPCA
   result = fit_kpca(K, n_components=2)

   # 4. Plot
   plot_kpca(result, color_values=energies,
             color_label="Formation energy (eV)")

Use Cases
---------

See :doc:`use-cases/index` for complete worked examples including
grid search, hyperparameter selection, and structure selection for DFT.