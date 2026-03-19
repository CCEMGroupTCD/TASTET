# SADS — Selection by Atomic Density Similarity

Explore atomic structures via SOAP descriptors, global similarity kernels, and kernel PCA.

## Installation

```bash
pip install -e .
```

## Quickstart

```python
from ase.io import read
from sads import compute_soap, compute_kernel, fit_kpca
from sads.plotting import plot_kpca

structures = read("structures.traj", index=":")

soap_list = compute_soap(structures, r_cut=4.0, n_max=6, l_max=6, sigma=0.1)
K = compute_kernel(soap_list, method="rematch", metric="linear", alpha=0.5)
result = fit_kpca(K, n_components=2)

plot_kpca(result, color_values=energies, color_label="Energy (eV)", show=True)
```

## Use cases

Each subdirectory under `use_cases/` is a self-contained analysis with its own `config.py` and `run.py`. See `use_cases/pablo_clusters/` for a worked example.

## Package structure

```
sads/
├── soap.py          # compute_soap — public API
├── soap_utils.py    # generate_environment_soap internals
├── kernel.py        # compute_kernel — average / REMatch kernels
├── kpca.py          # fit_kpca — KernelPCA wrapper + KPCAResult dataclass
├── io.py            # save/load SOAP and kernel arrays
├── plotting.py      # plot_kpca scatter helper
└── plot_style.py    # matplotlib style config (set_mpl_style, etc.)
```
