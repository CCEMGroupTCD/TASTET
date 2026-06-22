# TASTET — Tool for Atomistic Structure selection Through Efficient Triage

Explore atomic structures via SOAP descriptors, global similarity kernels, and kernel PCA.

## Installation

```bash
pip install -e .                # library only
pip install -e ".[examples]"    # also run the bundled examples (RDKit)
pip install -e ".[docs]"        # also build the Sphinx docs
```

## Quickstart

```python
from ase.io import read
from tastet.soap_utils import compute_soap
from tastet.kernel import compute_kernel
from tastet.kpca import fit_kpca
from tastet.plotting import plot_kpca

structures = read("structures.traj", index=":")

soap_list = compute_soap(structures, r_cut=4.0, n_max=6, l_max=6, sigma=0.1)
K = compute_kernel(soap_list, method="rematch", metric="linear", alpha=0.5)
result = fit_kpca(K, n_components=2)

plot_kpca(result, color_values=energies, color_label="Energy (eV)", show=True)
```

## Use cases

Two complete, self-contained example pipelines live under `examples/`:

- `examples/nanoclusters/` — Cu nanoclusters on a surface (single-kernel mode).
- `examples/rh_complex/` — Rh complex conformers (tensor-product mode, with a
  supervised round-2 workflow).

Each is driven by three scripts — `config.py`, `prepare.py`, and `run.py` — sharing
the same CLI steps (`db`, `grid_search`, `soap`, `kernel`, `kpca`, `select`). See the
[Use Cases](docs/use-cases/index.rst) docs for worked walkthroughs.

## Package structure

```
tastet/
├── soap_utils.py    # compute_soap — SOAP descriptors
├── kernel.py        # compute_kernel — average / REMatch kernels
├── kpca.py          # fit_kpca — KernelPCA wrapper + KPCAResult dataclass
├── io.py            # database, SOAP, and kernel save/load helpers
├── distance.py      # kernel-induced distance distributions
├── selection.py     # diverse structure selection (FPS / k-medoids)
├── pipeline.py      # shared db/soap/kernel/kpca/grid_search/select steps
├── cka.py           # centred kernel alignment scoring
├── metrics/         # scorer protocol + CKA scorer
├── sweep/           # SOAP × kernel grid search (single- and multi-channel)
└── plotting/        # kPCA scatter, heatmaps, distance plots, styling
```
