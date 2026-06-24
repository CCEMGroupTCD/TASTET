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
import pandas as pd
from ase.io import read
from tastet.soap_utils import compute_soap
from tastet.kernel import compute_kernel
from tastet.selection import select_structures

structures = read("structures.traj", index=":")

# Represent the structures in a SOAP-encoded kernel space
soap_list = compute_soap(structures, r_cut=4.0, n_max=6, l_max=6, sigma=0.1)
K = compute_kernel(soap_list, method="rematch", metric="linear", alpha=0.5)

# Select 10 space-filling representatives directly from the kernel
meta = pd.DataFrame({"configuration_id": range(1, len(structures) + 1)})
selected, pool, selected_indices = select_structures(K, meta, k=10, method="fps")
print(selected_indices)
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
├── cka.py           # centered kernel alignment scoring
├── metrics/         # scorer protocol + CKA scorer
├── sweep/           # SOAP × kernel grid search (single- and multi-channel)
└── plotting/        # kPCA scatter, heatmaps, distance plots, styling
```

## License

TASTET is released under the MIT License — see [`LICENSE`](LICENSE).
© 2025 Alejandro Cañete-Arché and the CCEM Group.
