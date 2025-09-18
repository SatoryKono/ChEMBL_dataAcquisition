"""Validation utilities for normalised assay tables."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, List, Mapping

import numpy as np
import pandas as pd
from pandas.api.types import is_scalar
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

LOGGER = logging.getLogger(__name__)


class AssaysSchema(BaseModel):
    """Pydantic model describing a validated assay record."""

    model_config = ConfigDict(extra="allow")

    assay_chembl_id: str
    document_chembl_id: str
    target_chembl_id: str | None = None
    assay_category: str | None = None
    assay_group: str | None = None
    assay_type: str | None = None
    assay_type_description: str | None = None
    assay_organism: str | None = None
    assay_test_type: str | None = None
    assay_cell_type: str | None = None
    assay_tissue: str | None = None
    assay_tax_id: str | None = None
    assay_with_same_target: int
    confidence_score: int | None = None
    confidence_description: str | None = None
    relationship_type: str | None = None
    relationship_description: str | None = None
    bao_format: str | None = None
    bao_label: str | None = None

    @field_validator("assay_chembl_id", mode="before")
    @classmethod
    def _ensure_non_empty(cls, value: Any) -> str:
        if not value:
            raise ValueError("assay_chembl_id must not be empty")
        return str(value)

    @field_validator("assay_with_same_target", mode="before")
    @classmethod
    def _ensure_positive(cls, value: Any) -> int:
        if value is None:
            raise ValueError("assay_with_same_target is required")
        int_value = int(value)
        if int_value < 0:
            raise ValueError("assay_with_same_target must be non-negative")
        return int_value

    @classmethod
    def ordered_columns(cls) -> List[str]:
        """Return the columns defined by the schema in declaration order."""

        return list(cls.model_fields.keys())


def _is_missing_scalar(value: Any) -> bool:
    """Return ``True`` when ``value`` represents a missing scalar."""

    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    return False


def _coerce_value(value: Any) -> Any:
    """Normalise ``value`` so it is JSON-serialisable and handles nulls."""

    if value is None:
        return None

    if isinstance(value, np.ndarray):
        dtype_kind = value.dtype.kind
        if dtype_kind in {"O", "U", "S"}:
            if value.size == 0:
                return []
            return [_coerce_value(item) for item in value.tolist()]
        if value.size == 0:
            return None
        if value.size == 1:
            return _coerce_value(value.item())
        return [_coerce_value(item) for item in value.tolist()]

    if isinstance(value, np.generic):
        scalar_value = value.item()
        return None if _is_missing_scalar(scalar_value) else scalar_value

    if is_scalar(value):
        return None if _is_missing_scalar(value) else value

    return value


def _coerce_record(row: pd.Series) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in row.items():
        clean[key] = _coerce_value(value)
    return clean


def validate_assays(
    df: pd.DataFrame,
    schema: type[AssaysSchema] = AssaysSchema,
    *,
    errors_path: Path,
) -> pd.DataFrame:
    """Validates rows in a DataFrame against the given schema and writes failures to a file.

    Args:
        df: The pandas DataFrame to validate.
        schema: The Pydantic schema to validate against.
        errors_path: The path to the file where validation errors will be written.

    Returns:
        A new DataFrame containing only the valid rows.
    """

    if df.empty:
        LOGGER.info("Validation skipped because the DataFrame is empty")
        return df

    valid_rows: List[dict[str, Any]] = []
    errors: List[dict[str, Any]] = []

    def _normalise_error_details(
        details: Iterable[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        normalised: list[dict[str, Any]] = []
        for entry in details:
            clean_entry = dict(entry)
            ctx = clean_entry.get("ctx")
            if isinstance(ctx, dict):
                clean_entry["ctx"] = {key: str(value) for key, value in ctx.items()}
            normalised.append(clean_entry)
        return normalised

    for index, row in df.iterrows():
        payload = _coerce_record(row)
        try:
            record = schema(**payload)
        except ValidationError as exc:
            LOGGER.warning("Validation error for row %s: %s", index, exc)
            errors.append(
                {
                    "index": int(index),
                    "errors": _normalise_error_details(exc.errors()),
                    "row": payload,
                }
            )
            continue
        valid_rows.append(record.model_dump())

    if errors:
        errors_path.parent.mkdir(parents=True, exist_ok=True)
        with errors_path.open("w", encoding="utf-8") as handle:
            json.dump(errors, handle, ensure_ascii=False, indent=2)
        LOGGER.info("Validation produced %d error records", len(errors))
    elif errors_path.exists():
        errors_path.unlink()

    return pd.DataFrame(valid_rows)
