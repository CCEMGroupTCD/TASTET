"""Round-2 supervised CKA re-optimisation and incremental selection.

Submodules:

* :mod:`round2._common` — shared helpers (energy loading, kPCA-space
  geometry, three-layer plotting) and :func:`round2._common.activate_round2`,
  which points :mod:`config` at the round-2 output namespace and channels.
* :mod:`round2.reoptimise` — the CKA-scored grid search (``cka_grid``).
* :mod:`round2.reselect` — the three further-selection strategies
  (``select``, ``zoom_select``, ``nearest_select``).
"""
