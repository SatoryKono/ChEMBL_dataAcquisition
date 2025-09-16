"""Validation utilities for normalised assay tables."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, List, Mapping

import numpy as np
import pandas as pd
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


def _is_missing(value: Any) -> bool:
    """Return ``True`` when ``value`` should be treated as missing.

    Args:
        value: Arbitrary value taken from a pandas ``Series``.

    Returns:
        ``True`` if the value should be converted to ``None`` during validation,
        otherwise ``False``.
    """

    if value is None:
        return True

    if isinstance(value, (pd.Series, np.ndarray)):
        if value.size == 0:
            return True
        missing = pd.isna(value)
        if isinstance(missing, (pd.Series, np.ndarray)):
            return bool(missing.all())
        return bool(missing)

    try:
        missing = pd.isna(value)
    except TypeError:
        return False

    if isinstance(missing, (pd.Series, np.ndarray)):
        if missing.size == 0:
            return True
        return bool(missing.all())
    return bool(missing)


def _coerce_record(row: pd.Series) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in row.items():
        if _is_missing(value):
            clean[key] = None
        else:
            clean[key] = value
    return clean


def validate_assays(
    df: pd.DataFrame,
    schema: type[AssaysSchema] = AssaysSchema,
    *,
    errors_path: Path,
) -> pd.DataFrame:
    """Validate and coerce assay records.

    Args:
        df: Normalised assay data to validate.
        schema: Pydantic model used to validate each row.
        errors_path: Location where validation errors are persisted as JSON.

    Returns:
        A DataFrame containing only the records that passed validation.
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
