"""Sweep result persistence and filtering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_results(df: pd.DataFrame, path: Path | str) -> Path:
    """Write sweep results to CSV, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_results(path: Path | str) -> pd.DataFrame:
    """Read sweep results from CSV."""
    return pd.read_csv(path)


def valid_rows(
    df: pd.DataFrame,
    score_col: str | None = None,
) -> pd.DataFrame:
    """Return only rows with ``status == "OK"`` and a finite score.

    Parameters
    ----------
    df : DataFrame
        Output of :func:`~sads.sweep.engine.run_sweep` (or loaded CSV).
    score_col : str, optional
        Name of the score column to filter on.  When *None*, only the
        status filter is applied.
    """
    out = df[df["status"] == "OK"].copy()
    if score_col and score_col in out.columns:
        out = out.dropna(subset=[score_col])
        out[score_col] = out[score_col].astype(float)
    return out


def top_results(
    df: pd.DataFrame,
    score_col: str,
    n: int = 10,
    ascending: bool = False,
    use_abs: bool = False,
) -> pd.DataFrame:
    """Return the *n* best rows sorted by *score_col*.

    Parameters
    ----------
    df : DataFrame
        Usually the output of :func:`valid_rows`.
    score_col : str
        Column to sort by.
    n : int
        How many rows to return.
    ascending : bool
        Sort order.  ``False`` (default) puts the highest scores first.
    use_abs : bool
        Sort by absolute value of *score_col*.
    """
    out = valid_rows(df, score_col)
    key = out[score_col].abs() if use_abs else out[score_col]
    out = out.assign(_sort_key=key).sort_values("_sort_key", ascending=ascending)
    return out.drop(columns="_sort_key").head(n)