# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SADS (Selection by Atomic Density Similarity) — Python research package and companion code for a publication. Provides structure-space analysis via SOAP descriptors, kernel matrices, and kernel PCA.

- `sads/` — the library
- `examples/nanoclusters/` and `examples/rh_complex/` — two public use cases
- `docs/` — Sphinx documentation

## Files to ignore

Never read, modify, or reference these private research files:

- `MEA/`, `MEA_backup/`, `Flow_cell_backup/`
- `all_data.csv`, `cual_tests.py`, `final_batch.csv`

## Code style

- No dead or redundant code. Delete rather than comment out.
- Type hints on all public functions (Python ≥ 3.10, PEP 484).
- `pathlib.Path` for all filesystem paths.
- Dataclasses for result containers (see `KPCAResult` in `sads/kpca.py`).

## Docstrings

Sphinx/reST style throughout. Every public function, class, and module gets a docstring.

```python
def func(x: int, *, flag: bool = False) -> float:
    """One-line summary.

    :param x: Description.
    :param flag: Description.
    :returns: Description.
    :raises ValueError: When x is negative.
    """
```

Use `:ivar name:` / `:vartype name:` for dataclass fields.

## Examples — isomorphism rule

`nanoclusters/` and `rh_complex/` must stay structurally parallel: same CLI steps in `run.py`, same function signatures in `prepare.py`, same docstring coverage. Each example has exactly:

- `config.py` — hyperparameters and path helpers
- `prepare.py` — data ingestion; must expose `ensure_database()` and `resolve_channel_soap()`
- `run.py` — CLI pipeline: `db`, [`subsample`], `grid_search`, `soap`, `kernel`, `kpca`, `select`

When modifying one example, check whether the same change applies to the other. (`subsample` is nanoclusters-only — that is expected.)

## Key invariant

`configuration_id` is 1-based and gap-free. Array index = `configuration_id - 1`. Subsampling must re-key 1..N.

## Commands

```bash
pip install -e ".[dev,docs,examples]"   # editable install with all extras
cd docs && make html                     # build Sphinx HTML
ruff check sads/ examples/              # lint
ruff format sads/ examples/             # format
```

No automated test suite.