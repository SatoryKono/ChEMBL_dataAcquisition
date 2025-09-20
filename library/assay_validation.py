"""Vectorised validation utilities for normalised assay tables."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from .validation_core import (
    ValidationResult,
    build_error_frame,
    coerce_dataframe,
    combine_error_frames,
    serialise_errors,
)

LOGGER = logging.getLogger(__name__)

_REQUIRED_COLUMNS = ("assay_chembl_id", "document_chembl_id")
_OPTIONAL_INT_COLUMNS = ("confidence_score",)


class AssaysSchema:
    """Schema stub exposing the legacy column order for downstream code."""

    @staticmethod
    def ordered_columns() -> list[str]:
        """Return the ordered column list from the historic Pydantic schema."""

        return [
            "assay_chembl_id",
            "document_chembl_id",
            "target_chembl_id",
            "assay_category",
            "assay_group",
            "assay_type",
            "assay_type_description",
            "assay_organism",
            "assay_test_type",
            "assay_cell_type",
            "assay_tissue",
            "assay_tax_id",
            "assay_with_same_target",
            "confidence_score",
            "confidence_description",
            "relationship_type",
            "relationship_description",
            "bao_format",
            "bao_label",
        ]


def _coerce_assay_identifier(df: pd.DataFrame) -> Iterable[pd.DataFrame]:
    column = "assay_chembl_id"
    if column not in df.columns:
        return []
    original = df[column]
    flattened = original.apply(
        lambda value: value[0] if isinstance(value, list) and len(value) == 1 else value
    )
    series = flattened.astype("string")
    stripped = series.str.strip()
    mask = stripped.isna() | stripped.eq("")
    df[column] = stripped.mask(mask, other=pd.NA)
    message = "assay_chembl_id must not be empty"
    return [
        build_error_frame(original, mask.fillna(False), column=column, message=message)
    ]


def _coerce_document_identifier(df: pd.DataFrame) -> Iterable[pd.DataFrame]:
    column = "document_chembl_id"
    if column not in df.columns:
        return []
    original = df[column]
    flattened = original.apply(
        lambda value: value[0] if isinstance(value, list) and len(value) == 1 else value
    )
    series = flattened.astype("string")
    mask = series.isna()
    df[column] = series
    message = "value must not be missing"
    return [
        build_error_frame(original, mask.fillna(False), column=column, message=message)
    ]


def _coerce_required_assay_count(df: pd.DataFrame) -> Iterable[pd.DataFrame]:
    column = "assay_with_same_target"
    if column not in df.columns:
        if df.empty:
            return []
        placeholder = pd.Series(pd.NA, index=df.index)
        missing_mask = pd.Series(True, index=df.index)
        return [
            build_error_frame(
                placeholder,
                missing_mask,
                column=column,
                message="assay_with_same_target is required",
                error_type="missing_error",
            )
        ]
    original = df[column]
    numeric = pd.to_numeric(original, errors="coerce")
    mask_missing = original.isna()
    mask_invalid = original.notna() & numeric.isna()
    ints = pd.Series(pd.NA, index=df.index, dtype="Int64")
    valid_numeric = numeric.dropna()
    if not valid_numeric.empty:
        ints.loc[valid_numeric.index] = valid_numeric.astype(int)
    mask_negative = ints < 0
    df[column] = ints.mask(mask_negative, other=pd.NA)
    return [
        build_error_frame(
            original,
            mask_missing.fillna(False),
            column=column,
            message="assay_with_same_target is required",
        ),
        build_error_frame(
            original,
            mask_invalid.fillna(False),
            column=column,
            message="value must be an integer",
        ),
        build_error_frame(
            original,
            mask_negative.fillna(False),
            column=column,
            message="value must be non-negative",
        ),
    ]


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


def validate_assays(df: pd.DataFrame, *, errors_path: Path) -> ValidationResult:
    """Validate the assay table using vectorised checks.

    Parameters
    ----------
    df:
        Normalised assay DataFrame.
    errors_path:
        Output path for the JSON error report.

    Returns
    -------
    ValidationResult
        Container with the filtered DataFrame and aggregated error report.
    """

    if df.empty:
        LOGGER.info("Validation skipped because the DataFrame is empty")
        empty_errors = combine_error_frames([])
        return ValidationResult(valid=df.copy(), errors=empty_errors)

    missing_required = sorted(set(_REQUIRED_COLUMNS) - set(df.columns))
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
    error_frames.extend(_coerce_assay_identifier(coerced))
    error_frames.extend(_coerce_document_identifier(coerced))
    error_frames.extend(_coerce_required_assay_count(coerced))
    for column in _OPTIONAL_INT_COLUMNS:
        error_frames.extend(_coerce_optional_int(coerced, column))

    errors = combine_error_frames(error_frames)
    serialise_errors(errors, coerced, errors_path=errors_path)

    if errors.empty:
        valid = coerced.reset_index(drop=True)
    else:
        invalid_indices = pd.Index(errors["index"].unique())
        mask_valid = ~coerced.index.isin(invalid_indices)
        valid = coerced.loc[mask_valid].reset_index(drop=True)

    return ValidationResult(valid=valid, errors=errors)


__all__ = ["validate_assays", "ValidationResult", "AssaysSchema"]
