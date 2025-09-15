"""Tests for the :mod:`library.data_profiling` module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from library.data_profiling import analyze_table_quality


def test_analyze_table_quality_creates_reports(tmp_path: Path) -> None:
    """Profile a small data frame and ensure reports are written."""

    df = pd.DataFrame({"num": [1, 2, 3], "text": ["a", "b", None]})
    prefix = tmp_path / "sample"
    quality, corr = analyze_table_quality(df, table_name=str(prefix))

    assert quality["column"].tolist() == ["num", "text"]
    assert (tmp_path / "sample_quality_report_table.csv").exists()
    assert (tmp_path / "sample_data_correlation_report_table.csv").exists()
    assert isinstance(corr, pd.DataFrame)
