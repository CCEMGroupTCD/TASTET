---
name: verify-examples
description: Check that examples/nanoclusters/ and examples/rh_complex/ are isomorphic in CLI structure, prepare.py interface, config.py keys, and docstring coverage.
---

Compare the two public examples for structural parity. Read both `prepare.py`, `run.py`, and `config.py` files from each example, then check:

1. **CLI steps** — Both `run.py` files must define the same pipeline steps: `db`, `grid_search`, `soap`, `kernel`, `kpca`, `select`. (`subsample` is nanoclusters-only — expected.) Report any step present in one but missing from the other.

2. **prepare.py interface** — Both must expose `ensure_database()` and `resolve_channel_soap()`. Compare signatures and docstrings. Report any mismatch.

3. **config.py structure** — Both must define `SOAP_PARAMS`, `KERNEL_PARAMS`, `USE_TENSOR_PRODUCT`, `ANALYSIS_NAME`, and path helpers (`db_path`, `soap_path`, `kernel_path`, etc.). List any keys missing from either.

4. **Docstring coverage** — Every public function in `prepare.py`, `run.py`, and `config.py` must have a Sphinx/reST docstring with `:param:`, `:returns:` where applicable. Report any missing or incomplete docstrings.

Report differences concisely. If the examples are in sync, say so explicitly.
