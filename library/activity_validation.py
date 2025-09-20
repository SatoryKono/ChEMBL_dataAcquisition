"""Vectorised validation utilities for normalised activity tables."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .validation_core import (
    ValidationResult,
    build_error_frame,
    coerce_dataframe,
    combine_error_frames,
    serialise_errors,
)

LOGGER = logging.getLogger(__name__)

_REQUIRED_STR_COLUMNS = ("activity_chembl_id", "assay_chembl_id")
_INT_COLUMNS = ("record_id", "activity_id")
_FLOAT_COLUMNS = (
    "standard_value",
    "standard_upper_value",
    "standard_lower_value",
    "pchembl_value",
)


class ActivitiesSchema:
    """Schema shim providing the legacy column ordering."""

    @staticmethod
    def ordered_columns() -> list[str]:
        """Return the ordered column list from the historic schema."""

        return [
            "activity_chembl_id",
            "assay_chembl_id",
            "molecule_chembl_id",
            "parent_molecule_chembl_id",
            "document_chembl_id",
            "target_chembl_id",
            "record_id",
            "activity_id",
            "standard_type",
            "standard_relation",
            "standard_units",
            "standard_value",
            "standard_upper_value",
            "standard_lower_value",
            "pchembl_value",
            "potential_duplicate",
            "data_validity_comment",
            "data_validity_warning",
            "activity_comment",
            "type",
            "relation",
            "units",
        ]


def _ensure_required_strings(df: pd.DataFrame) -> Iterable[pd.DataFrame]:
    for column in _REQUIRED_STR_COLUMNS:
        if column not in df.columns:
            continue
        original = df[column]
        flattened = original.apply(
            lambda value: (
                value[0] if isinstance(value, list) and len(value) == 1 else value
            )
        )

        def _is_falsey(value: Any) -> bool:
            if pd.isna(value):
                return True
            try:
                return not bool(value)
            except TypeError:
                return False

        mask = flattened.apply(_is_falsey)
        series = flattened.astype("string")
        df[column] = series.mask(mask, other=pd.NA)
        yield build_error_frame(
            original,
            mask.fillna(False),
            column=column,
            message="value must not be empty",
        )


def _coerce_optional_int(df: pd.DataFrame, column: str) -> Iterable[pd.DataFrame]:
    if column not in df.columns:
        return []
    original = df[column]
    numeric = pd.to_numeric(original, errors="coerce")
    mask_invalid = original.notna() & numeric.isna()
    result = pd.Series(pd.NA, index=df.index, dtype="Int64")
    valid_numeric = numeric.dropna()
    if not valid_numeric.empty:
        result.loc[valid_numeric.index] = valid_numeric.astype(int)
    df[column] = result
    return [
        build_error_frame(
            original,
            mask_invalid.fillna(False),
            column=column,
            message="value must be an integer",
        )
    ]


def _coerce_optional_float(df: pd.DataFrame, column: str) -> Iterable[pd.DataFrame]:
    if column not in df.columns:
        return []
    original = df[column]
    numeric = pd.to_numeric(original, errors="coerce")
    mask_invalid = original.notna() & numeric.isna()
    df[column] = numeric.astype("Float64")
    return [
        build_error_frame(
            original,
            mask_invalid.fillna(False),
            column=column,
            message="value must be numeric",
        )
    ]


def validate_activities(df: pd.DataFrame, *, errors_path: Path) -> ValidationResult:
    """Validate the activity table and emit aggregated diagnostics.

    Parameters
    ----------
    df:
        Normalised activity DataFrame.
    errors_path:
        Output path for the JSON error report.

    Returns
    -------
    ValidationResult
        Container with validated rows and the aggregated error DataFrame.
    """

    if df.empty:
        LOGGER.info("Validation skipped because the DataFrame is empty")
        empty_errors = combine_error_frames([])
        return ValidationResult(valid=df.copy(), errors=empty_errors)

    missing_required = sorted(set(_REQUIRED_STR_COLUMNS) - set(df.columns))
    if missing_required:
        LOGGER.warning(
            "Input data is missing required columns: %s", ", ".join(missing_required)
        )

    coerced = coerce_dataframe(df)
    error_frames: list[pd.DataFrame] = []
    if missing_required:
        missing_mask = pd.Series(True, index=coerced.index)
        for column in missing_required:
            placeholder = pd.Series(pd.NA, index=coerced.index)
            error_frames.append(
                build_error_frame(
                    placeholder,
                    missing_mask,
                    column=column,
                    message="column is required",
                    error_type="missing_error",
                )
            )
    error_frames.extend(_ensure_required_strings(coerced))
    for column in _INT_COLUMNS:
        error_frames.extend(_coerce_optional_int(coerced, column))
    for column in _FLOAT_COLUMNS:
        error_frames.extend(_coerce_optional_float(coerced, column))

    errors = combine_error_frames(error_frames)
    serialise_errors(errors, coerced, errors_path=errors_path)

    if errors.empty:
        valid = coerced.reset_index(drop=True)
    else:
        invalid_indices = pd.Index(errors["index"].unique())
        mask_valid = ~coerced.index.isin(invalid_indices)
        valid = coerced.loc[mask_valid].reset_index(drop=True)

    return ValidationResult(valid=valid, errors=errors)


__all__ = ["validate_activities", "ValidationResult", "ActivitiesSchema"]
