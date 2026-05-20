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

    :param df: Output of :func:`~sads.sweep.engine.run_sweep`, or a loaded CSV.
    :param score_col: Name of the score column to filter on. When ``None``, only
        the status filter is applied.
    :returns: Filtered DataFrame containing only successful rows with finite scores
        when ``score_col`` is provided.
    :rtype: pandas.DataFrame
    """
    out = df[df["status"] == "OK"].copy()
    if score_col and score_col in out.columns:
        out = out.dropna(subset=[score_col])
        out[score_col] = out[score_col].astype(float)
    return out