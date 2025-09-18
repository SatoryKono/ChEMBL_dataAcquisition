"""Tabular data profiling utilities.

This module provides :func:`analyze_table_quality` which profiles each
column in a :class:`pandas.DataFrame` and computes pairwise correlations
for numeric columns.  The function writes two CSV reports: a quality report
and a correlation matrix.  It has no third-party dependencies beyond
``pandas`` and ``numpy`` and is suitable for offline use.
"""

from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Sized
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import DtypeWarning

LOGGER = logging.getLogger(__name__)

# Precompiled regular expressions for pattern coverage
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
_ISSN_RE = re.compile(r"^\d{4}-\d{3}[\dXx]$")
_URL_RE = re.compile(r"^(?:https?|ftp)://|^www\.")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

BOOL_LIKE = {"true", "false", "yes", "no", "y", "n", "1", "0", "t", "f"}


def _load_table(
    table: pd.DataFrame | str | Path,
    separator: str = ",",
    encoding: str | None = None,
) -> pd.DataFrame:
    """Load a table from ``table``.

    Parameters
    ----------
    table:
        Either an existing :class:`pandas.DataFrame` or path to a CSV file.
    separator:
        Field delimiter used when reading CSV files. Defaults to a comma.
    encoding:
        Encoding to use when reading the CSV file. When ``None`` (the
        default) the function attempts several common encodings until one
        succeeds.

    Returns
    -------
    pandas.DataFrame
        Loaded data frame.
    """

    if isinstance(table, pd.DataFrame):
        return table.copy()

    path = Path(table)
    encodings: list[str]
    if encoding:
        encodings = [encoding]
    else:
        encodings = ["utf-8-sig", "utf-8", "cp1251", "latin-1"]
    for enc in encodings:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DtypeWarning)
                return pd.read_csv(
                    path,
                    sep=separator,
                    encoding=enc,
                    low_memory=False,
                )
        except UnicodeDecodeError:
            LOGGER.debug("failed to decode %s with %s", path, enc)
            continue
    raise UnicodeDecodeError(
        encodings[0],
        b"",
        0,
        1,
        f"Unable to decode CSV using encodings {', '.join(encodings)}",
    )


def _is_isbn(value: str) -> bool:
    """Return ``True`` if ``value`` is a valid ISBN10/13."""

    digits = re.sub(r"[-\s]", "", value)
    if re.fullmatch(r"\d{9}[\dXx]", digits):
        total = sum(
            (10 - i) * (10 if ch.upper() == "X" else int(ch))
            for i, ch in enumerate(digits)
        )
        return total % 11 == 0
    if re.fullmatch(r"\d{13}", digits):
        total = sum(
            (1 if i % 2 == 0 else 3) * int(ch) for i, ch in enumerate(digits[:-1])
        )
        check = (10 - total % 10) % 10
        return check == int(digits[-1])
    return False


def _non_empty_mask(series: pd.Series) -> pd.Series:
    """Boolean mask of values that are not empty by content."""

    def is_non_empty(val: Any) -> bool:
        if isinstance(val, str):
            return bool(val.strip())
        if isinstance(val, Sized) and not isinstance(val, (bytes, bytearray)):
            # For sequences and other sized containers, consider length
            return len(val) > 0
        try:
            return not pd.isna(val)
        except (TypeError, ValueError):
            # Objects that cannot be evaluated by pandas are treated as non-empty
            return True

    return series.map(is_non_empty)


def _string_values(series: pd.Series, mask: pd.Series) -> pd.Series:
    """Return non-empty string values from ``series`` according to ``mask``."""

    return (
        series[mask & series.map(lambda x: isinstance(x, str))].astype(str).str.strip()
    )


def _pattern_cov(strings: pd.Series, pattern: re.Pattern[str]) -> float:
    """Fraction of ``strings`` matching ``pattern``."""

    if strings.empty:
        return 0.0
    return float(strings.str.match(pattern).mean())


def _isbn_cov(strings: pd.Series) -> float:
    if strings.empty:
        return 0.0
    return float(strings.map(_is_isbn).mean())


def _bool_like_cov(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return float(values.astype(str).str.strip().str.lower().isin(BOOL_LIKE).mean())


def _numeric_stats(series: pd.Series) -> tuple[pd.Series, float, dict[str, float]]:
    """Convert ``series`` to numeric values and compute summary statistics."""

    numeric = pd.to_numeric(series, errors="coerce").astype(float)

    coverage = float(numeric.notna().mean())
    stats = {
        "numeric_min": float(numeric.min()) if coverage else np.nan,
        "numeric_p50": float(numeric.quantile(0.5)) if coverage else np.nan,
        "numeric_p95": float(numeric.quantile(0.95)) if coverage else np.nan,
        "numeric_max": float(numeric.max()) if coverage else np.nan,
        "numeric_mean": float(numeric.mean()) if coverage else np.nan,
        "numeric_std": float(numeric.std(ddof=0)) if coverage else np.nan,
    }

    return numeric, coverage, stats


def _parse_dates(series: pd.Series) -> tuple[float, dict[str, pd.Timestamp | float]]:
    """Attempt to parse ``series`` as dates and compute summary statistics."""

    def normalise(val: object) -> object:
        if isinstance(val, str) and re.fullmatch(r"\d{4}", val.strip()):
            return f"{val.strip()}-07-01"
        return val

    normalised = series.map(normalise)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Parsed string.*included an un-recognized timezone",
            category=FutureWarning,
        )
        dates = pd.to_datetime(normalised, errors="coerce", utc=True, format="mixed")

    coverage = float(dates.notna().mean())
    if coverage:
        dt = dates.dropna().dt.tz_convert(None)
        stats: dict[str, pd.Timestamp | float] = {
            "date_min": dt.min(),
            "date_p50": dt.quantile(0.5),
            "date_max": dt.max(),
        }
    else:
        stats = {"date_min": np.nan, "date_p50": np.nan, "date_max": np.nan}
    return coverage, stats


def _text_length_stats(series: pd.Series) -> dict[str, float]:
    values = series.dropna().map(lambda x: str(x).strip())
    if values.empty:
        return {
            "text_len_min": np.nan,
            "text_len_p50": np.nan,
            "text_len_p95": np.nan,
            "text_len_max": np.nan,
        }
    lengths = values.map(len)
    return {
        "text_len_min": float(lengths.min()),
        "text_len_p50": float(lengths.quantile(0.5)),
        "text_len_p95": float(lengths.quantile(0.95)),
        "text_len_max": float(lengths.max()),
    }


def _top_values(series: pd.Series) -> str:
    counts = series.dropna().map(lambda x: str(x).strip()).value_counts().head(3)
    parts = [f"{str(val)[:60]} ({cnt})" for val, cnt in counts.items()]
    return "; ".join(parts)


def analyze_table_quality(
    table: pd.DataFrame | str | Path,
    table_name: str,
    separator: str = ",",
    encoding: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profile ``table`` and compute correlations for numeric columns.

    Parameters
    ----------
    table:
        :class:`pandas.DataFrame` or path to a CSV file.
    table_name:
        Base name used for output files.  The generated reports append
        ``"_quality_report_table.csv"`` and ``"_data_correlation_report_table.csv"``
        to this value.
    separator:
        Field delimiter used when reading CSV files provided via ``table``.
        Defaults to a comma.
    encoding:
        Encoding to use when reading the CSV file. When ``None`` (the
        default) the reader attempts several common encodings.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Quality report and correlation matrix.
    """

    df = _load_table(table, separator=separator, encoding=encoding)
    rows: list[dict[str, object]] = []
    numeric_candidates: dict[str, pd.Series] = {}
    for column in df.columns:
        series = df[column]
        non_null = int(series.notna().sum())
        mask = _non_empty_mask(series)
        non_empty = int(mask.sum())
        empty_pct = float(1 - non_empty / len(series)) if len(series) else 0.0
        try:
            unique_cnt = int(series.dropna().nunique())
        except TypeError:
            # ``nunique`` requires hashable values; fall back to string representation
            unique_cnt = int(series.dropna().map(str).nunique())
        unique_pct = float(unique_cnt / non_empty) if non_empty else np.nan

        strings = _string_values(series, mask)
        pattern_cov_doi = _pattern_cov(strings, _DOI_RE)
        pattern_cov_issn = _pattern_cov(strings, _ISSN_RE)
        pattern_cov_isbn = _isbn_cov(strings)
        pattern_cov_url = _pattern_cov(strings, _URL_RE)
        pattern_cov_email = _pattern_cov(strings, _EMAIL_RE)

        bool_like_cov = _bool_like_cov(strings)

        numeric_series, numeric_cov, num_stats = _numeric_stats(series)
        if numeric_cov >= 0.8:
            numeric_candidates[column] = numeric_series

        date_cov, date_stats = _parse_dates(series)

        text_stats = _text_length_stats(series)

        roles: list[str] = []
        if pattern_cov_url > 0:
            roles.append("url")
        if pattern_cov_doi > 0:
            roles.append("doi")
        if pattern_cov_issn > 0:
            roles.append("issn")
        if pattern_cov_isbn > 0:
            roles.append("isbn")
        if pattern_cov_email > 0:
            roles.append("email")
        if bool_like_cov >= 0.8:
            roles.append("boolean")
        if date_cov >= 0.8:
            roles.append("date")
        if numeric_cov >= 0.8:
            roles.append("numeric")
        if non_empty and unique_cnt / non_empty >= 0.98:
            roles.append("identifier-like")
        elif unique_cnt <= min(100, 0.05 * non_empty):
            roles.append("categorical")
        else:
            roles.append("free-text")

        row: dict[str, object] = {
            "column": column,
            "non_null": non_null,
            "non_empty": non_empty,
            "empty_pct": empty_pct,
            "unique_cnt": unique_cnt,
            "unique_pct_of_non_empty": unique_pct,
            "pattern_cov_doi": pattern_cov_doi,
            "pattern_cov_issn": pattern_cov_issn,
            "pattern_cov_isbn": pattern_cov_isbn,
            "pattern_cov_url": pattern_cov_url,
            "pattern_cov_email": pattern_cov_email,
            "bool_like_cov": bool_like_cov,
            "numeric_cov": numeric_cov,
            **num_stats,
            "date_cov": date_cov,
            **date_stats,
            **text_stats,
            "guessed_roles": "|".join(roles),
            "top_values": _top_values(series),
        }
        rows.append(row)

    column_order = [
        "column",
        "non_null",
        "non_empty",
        "empty_pct",
        "unique_cnt",
        "unique_pct_of_non_empty",
        "pattern_cov_doi",
        "pattern_cov_issn",
        "pattern_cov_isbn",
        "pattern_cov_url",
        "pattern_cov_email",
        "bool_like_cov",
        "numeric_cov",
        "numeric_min",
        "numeric_p50",
        "numeric_p95",
        "numeric_max",
        "numeric_mean",
        "numeric_std",
        "date_cov",
        "date_min",
        "date_p50",
        "date_max",
        "text_len_min",
        "text_len_p50",
        "text_len_p95",
        "text_len_max",
        "guessed_roles",
        "top_values",
    ]
    quality_report = pd.DataFrame(rows, columns=column_order)
    quality_path = f"{table_name}_quality_report_table.csv"
    quality_report.to_csv(quality_path, index=False, encoding="utf-8-sig")

    corr_path = f"{table_name}_data_correlation_report_table.csv"
    if numeric_candidates:
        corr_report = pd.DataFrame(numeric_candidates).corr(method="pearson")
        corr_report.reset_index().to_csv(corr_path, index=False, encoding="utf-8-sig")
    else:
        corr_report = pd.DataFrame()
        corr_report.to_csv(corr_path, index=False, encoding="utf-8-sig")

    return quality_report, corr_report


if __name__ == "__main__":  # pragma: no cover - illustrative usage
    demo = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "url": ["https://example.com", "http://test.com", None, ""],
            "value": [10, "20", "30", "not"],
            "year": ["2020", "1999", None, "no"],
        }
    )
    analyze_table_quality(demo, table_name="demo")
