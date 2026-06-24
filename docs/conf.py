"""Sphinx configuration for TASTET documentation."""

project = "TASTET"
copyright = "2026, Alejandro Cañete-Arché and the CCEM Group"
author = "Alejandro Cañete-Arché"

# ── Extensions ────────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",  # pull docstrings from code
    "sphinx.ext.viewcode",  # add [source] links to API docs
    "sphinx.ext.intersphinx",  # cross-link to numpy/sklearn docs
]

# ── Autodoc ───────────────────────────────────────────────────────────
autodoc_member_order = "bysource"  # keep source order, not alphabetical
autodoc_typehints = "description"  # show type hints in the description
autodoc_default_options = {
    "members": True,
    "undoc-members": False,  # skip functions without docstrings
    "show-inheritance": True,
}

# ── Intersphinx (cross-references to external docs) ──────────────────
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# ── Theme ─────────────────────────────────────────────────────────────
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}

# ── Paths ─────────────────────────────────────────────────────────────
import os, sys

sys.path.insert(0, os.path.abspath(".."))  # so autodoc can import tastet
