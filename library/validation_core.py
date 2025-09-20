"""Shared utilities for tabular validation routines."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_scalar

LOGGER = logging.getLogger(__name__)

ERROR_URL = "https://pandera.readthedocs.io/en/stable/error_handling.html"


@dataclass(frozen=True)
class ValidationResult:
    """Container for validation results.

    Attributes
    ----------
    valid:
        DataFrame containing rows that passed validation.
    errors:
        Aggregated error report with one row per failure.
    """

    valid: pd.DataFrame
    errors: pd.DataFrame


def _is_missing_scalar(value: Any) -> bool:
    """Return ``True`` when ``value`` is recognised as a missing scalar."""

    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    return False


def coerce_value(value: Any) -> Any:
    """Normalise ``value`` so it is JSON serialisable and handles nulls."""

    if value is None:
        return None
    if isinstance(value, np.ndarray):
        dtype_kind = value.dtype.kind
        if dtype_kind in {"O", "U", "S"}:
            if value.size == 0:
                return []
            converted = [coerce_value(item) for item in value.tolist()]
            if len(converted) == 1:
                single = converted[0]
                if isinstance(single, dict):
                    return converted
                return single
            return converted
        if value.size == 0:
            return None
        if value.size == 1:
            return coerce_value(value.item())
        return [coerce_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        scalar_value = value.item()
        return None if _is_missing_scalar(scalar_value) else scalar_value
    if is_scalar(value):
        return None if _is_missing_scalar(value) else value
    return value


def coerce_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with values coerced via :func:`coerce_value`."""

    if df.empty:
        return df.copy()
    return df.applymap(coerce_value)


def build_error_frame(
    source: pd.Series,
    mask: pd.Series,
    *,
    column: str,
    message: str,
    error_type: str = "value_error",
) -> pd.DataFrame:
    """Construct a standard error report slice for a column.

    Parameters
    ----------
    source:
        Series containing the original values to report.
    mask:
        Boolean mask identifying failing rows.
    column:
        Column name that triggered the error.
    message:
        Human readable message describing the failure.
    error_type:
        Short error classification identifier.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``index``, ``column``, ``message``, ``value`` and
        ``error_type``. Empty when ``mask`` selects no rows.
    """

    failing = source.loc[mask]
    if failing.empty:
        return pd.DataFrame(
            columns=["index", "column", "message", "value", "error_type"]
        )
    frame = (
        failing.to_frame(name="value")
        .reset_index(names="index")
        .assign(column=column, message=message, error_type=error_type)
    )
    return frame[["index", "column", "message", "value", "error_type"]]


def combine_error_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Combine error frames ensuring canonical ordering."""

    materialised = [frame for frame in frames if not frame.empty]
    if not materialised:
        return pd.DataFrame(
            columns=["index", "column", "message", "value", "error_type"]
        )
    combined = pd.concat(materialised, ignore_index=True)
    combined = combined.sort_values(["index", "column", "message"]).reset_index(
        drop=True
    )
    return combined


def serialise_errors(
    errors: pd.DataFrame,
    data: pd.DataFrame,
    *,
    errors_path: Path,
    error_url: str = ERROR_URL,
) -> None:
    """Write the aggregated error report to disk using the legacy JSON layout."""

    if errors.empty:
        if errors_path.exists():
            errors_path.unlink()
        return

    payload: list[dict[str, Any]] = []
    for index, group in errors.groupby("index", sort=True):
        row = data.loc[index]
        row_payload = {column: coerce_value(value) for column, value in row.items()}
        error_entries = [
            {
                "type": str(item["error_type"]),
                "loc": [str(item["column"])],
                "msg": str(item["message"]),
                "input": coerce_value(item["value"]),
                "ctx": {"error": str(item["message"])},
                "url": error_url,
            }
            for item in group.to_dict("records")
        ]
        payload.append(
            {"index": int(index), "errors": error_entries, "row": row_payload}
        )

    errors_path.parent.mkdir(parents=True, exist_ok=True)
    with errors_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    LOGGER.info("Validation produced %d error records", len(payload))


__all__ = [
    "ValidationResult",
    "build_error_frame",
    "coerce_dataframe",
    "coerce_value",
    "combine_error_frames",
    "serialise_errors",
]
