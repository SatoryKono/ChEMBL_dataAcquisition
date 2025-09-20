"""Performance regression tests for :mod:`library.normalize_activities`."""

from __future__ import annotations

import time
from typing import Any, Callable

import pandas as pd

from library.normalize_activities import (
    _FLOAT_COLUMNS,
    _INT_COLUMNS,
    normalize_activities,
)


def _legacy_normalise_numeric(value: Any) -> float | None:
    """Mirror the element-wise numeric normalisation used before vectorisation."""

    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    if pd.isna(value):  # Handles pd.NA and numpy NaN values.
        return None
    numeric = pd.to_numeric([value], errors="coerce")[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _legacy_normalise_integer(value: Any) -> int | None:
    """Mirror the previous integer normalisation used during mapping."""

    numeric = _legacy_normalise_numeric(value)
    if numeric is None:
        return None
    return int(numeric)


def _legacy_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the original mapping-based normalisation for benchmarking."""

    result = df.copy()
    for column in _FLOAT_COLUMNS:
        if column in result.columns:
            result[column] = (
                result[column].map(_legacy_normalise_numeric).astype("Float64")
            )
    for column in _INT_COLUMNS:
        if column in result.columns:
            result[column] = (
                result[column].map(_legacy_normalise_integer).astype("Int64")
            )
    return result


def _build_float_payload(factory: Callable[[int], Any], size: int) -> list[Any]:
    """Create a list of heterogeneous numeric values for float columns."""

    return [factory(index) for index in range(size)]


def _float_factory(index: int) -> Any:
    """Generate diverse numeric payloads including strings and missing values."""

    remainder = index % 12
    if remainder == 0:
        return None
    if remainder == 1:
        return ""
    if remainder == 2:
        return "  "
    if remainder == 3:
        return float(index)
    if remainder == 4:
        return f"{index}"
    if remainder == 5:
        return f"{index}.5"
    if remainder == 6:
        return f" {-index}.25 "
    if remainder == 7:
        return index
    if remainder == 8:
        return -index / 3
    if remainder == 9:
        return pd.NA
    if remainder == 10:
        return "not-a-number"
    return f"{index / 7:.5f}"


def _int_factory(index: int) -> Any:
    """Generate values that require truncation, coercion and null handling."""

    remainder = index % 10
    if remainder == 0:
        return None
    if remainder == 1:
        return ""
    if remainder == 2:
        return " 42 "
    if remainder == 3:
        return str(index)
    if remainder == 4:
        return index
    if remainder == 5:
        return f"{index}.9"
    if remainder == 6:
        return -index
    if remainder == 7:
        return pd.NA
    if remainder == 8:
        return "not-a-number"
    return -index / 2


def test_normalize_activities_numeric_performance() -> None:
    """Vectorised normalisation must outperform the previous map-based variant."""

    row_count = 25_000
    float_values_primary = _build_float_payload(_float_factory, row_count)
    float_values_secondary = _build_float_payload(
        lambda idx: _float_factory(row_count - idx - 1), row_count
    )
    int_values_primary = [_int_factory(index) for index in range(row_count)]
    int_values_secondary = [
        _int_factory(row_count - index - 1) for index in range(row_count)
    ]

    raw = pd.DataFrame(
        {
            "activity_chembl_id": [f" CHEMBL{index} " for index in range(row_count)],
            "standard_value": float_values_primary,
            "activity_value": float_values_secondary,
            "standard_flag": int_values_primary,
            "activity_id": int_values_secondary,
        }
    )

    start_vectorised = time.perf_counter()
    vectorised = normalize_activities(raw)
    vectorised_duration = time.perf_counter() - start_vectorised

    start_legacy = time.perf_counter()
    legacy = _legacy_normalize(raw)
    legacy_duration = time.perf_counter() - start_legacy

    pd.testing.assert_series_equal(
        vectorised["standard_value"], legacy["standard_value"], check_names=False
    )
    pd.testing.assert_series_equal(
        vectorised["activity_value"], legacy["activity_value"], check_names=False
    )
    pd.testing.assert_series_equal(
        vectorised["standard_flag"], legacy["standard_flag"], check_names=False
    )
    pd.testing.assert_series_equal(
        vectorised["activity_id"], legacy["activity_id"], check_names=False
    )

    assert vectorised_duration < legacy_duration
