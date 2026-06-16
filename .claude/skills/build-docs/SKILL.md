---
name: build-docs
description: Build Sphinx HTML documentation and report autodoc warnings or errors. Use after changing docstrings, module structure, or rst files.
---

Run the Sphinx build and report any problems:

1. Run `cd docs && make html 2>&1` from the project root and capture all output.
2. Report any WARNING or ERROR lines (autodoc failures, missing references, intersphinx misses).
3. If the build succeeds cleanly, confirm and give the output path.
4. Do not open a browser or take further action.