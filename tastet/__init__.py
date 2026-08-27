"""TASTET — Tool for Atomistic Structure selection Through Efficient Triage."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tastet")
except PackageNotFoundError:  # running from a source tree, not installed
    __version__ = "0.0.0"

__all__ = ["__version__"]
