"""Vectorised validation schema for normalised ChEMBL test item tables."""

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

_REQUIRED_STR_COLUMNS = ("molecule_chembl_id", "canonical_smiles", "standard_inchi_key")
_INT_COLUMNS = (
    "chembl_num_ro5_violations",
    "pubchem_cid",
    "pubchem_h_bond_donor_count",
    "pubchem_h_bond_acceptor_count",
    "pubchem_rotatable_bond_count",
)
_FLOAT_COLUMNS = (
    "chembl_full_mwt",
    "chembl_alogp",
    "pubchem_molecular_weight",
    "pubchem_tpsa",
    "pubchem_x_log_p",
)


class TestitemsSchema:
    """Lightweight schema placeholder retaining the historical column order."""

    @staticmethod
    def ordered_columns() -> list[str]:
        """Return the ordered column list from the legacy schema."""

        return [
            "molecule_chembl_id",
            "canonical_smiles",
            "standard_inchi_key",
            "pref_name",
            "molecule_type",
            "structure_type",
            "salt_chembl_id",
            "parent_chembl_id",
            "max_phase",
            "chembl_full_mwt",
            "chembl_alogp",
            "chembl_num_ro5_violations",
            "chembl_molecular_species",
            "pubchem_cid",
            "pubchem_molecular_formula",
            "pubchem_molecular_weight",
            "pubchem_tpsa",
            "pubchem_x_log_p",
            "pubchem_h_bond_donor_count",
            "pubchem_h_bond_acceptor_count",
            "pubchem_rotatable_bond_count",
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
        series = flattened.astype("string")
        stripped = series.str.strip()
        mask = stripped.isna() | stripped.eq("")
        df[column] = stripped.mask(mask, other=pd.NA)
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


def _coerce_max_phase(df: pd.DataFrame) -> Iterable[pd.DataFrame]:
    column = "max_phase"
    if column not in df.columns:
        return []
    original = df[column]
    numeric = pd.to_numeric(original, errors="coerce")
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


def validate_testitems(df: pd.DataFrame, *, errors_path: Path) -> ValidationResult:
    """Validate the provided DataFrame and write a legacy JSON error report.

    Parameters
    ----------
    df:
        DataFrame containing normalised ChEMBL test item records.
    errors_path:
        Destination for the JSON error report. Existing files are overwritten on
        success and removed when the validation is clean.

    Returns
    -------
    ValidationResult
        Object containing the filtered DataFrame with valid rows and the
        aggregated error report.
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
    error_frames.extend(_coerce_max_phase(coerced))
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


__all__ = ["validate_testitems", "ValidationResult", "TestitemsSchema"]
