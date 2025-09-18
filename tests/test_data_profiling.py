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


def test_analyze_table_quality_uses_custom_separator(tmp_path: Path) -> None:
    """Ensure semicolon separated CSV files are handled when a separator is set."""

    csv_path = Path(__file__).parent / "data" / "sample_semicolon.csv"
    prefix = tmp_path / "semicolon_sample"

    quality, _ = analyze_table_quality(
        str(csv_path),
        table_name=str(prefix),
        separator=";",
    )

    assert quality["column"].tolist() == ["num", "text"]
    assert quality.loc[quality["column"] == "num", "non_null"].item() == 3
    assert (tmp_path / "semicolon_sample_quality_report_table.csv").exists()
    assert (tmp_path / "semicolon_sample_data_correlation_report_table.csv").exists()
