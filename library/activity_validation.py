"""Validation utilities for normalised activity tables."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, Type

import numpy as np
import pandas as pd
from pandas.api.types import is_scalar
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

LOGGER = logging.getLogger(__name__)


class ActivitiesSchema(BaseModel):
    """Pydantic model describing a validated activity record."""

    model_config = ConfigDict(extra="allow")

    activity_chembl_id: str
    assay_chembl_id: str
    molecule_chembl_id: str | None = None
    parent_molecule_chembl_id: str | None = None
    document_chembl_id: str | None = None
    target_chembl_id: str | None = None
    record_id: int | None = None
    activity_id: int | None = None
    standard_type: str | None = None
    standard_relation: str | None = None
    standard_units: str | None = None
    standard_value: float | None = None
    standard_upper_value: float | None = None
    standard_lower_value: float | None = None
    pchembl_value: float | None = None
    potential_duplicate: bool | None = None
    data_validity_comment: str | None = None
    data_validity_warning: bool | None = None
    activity_comment: str | None = None
    type: str | None = None
    relation: str | None = None
    units: str | None = None

    @field_validator("activity_chembl_id", "assay_chembl_id", mode="before")
    @classmethod
    def _ensure_non_empty(cls, value: Any) -> str:
        if not value:
            msg = "value must not be empty"
            raise ValueError(msg)
        return str(value)

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
        clean[str(key)] = _coerce_value(value)
    return clean


def validate_activities(
    df: pd.DataFrame,
    schema: Type[ActivitiesSchema] = ActivitiesSchema,
    *,
    errors_path: Path,
) -> pd.DataFrame:
    """Validate rows in ``df`` against ``schema`` and write failures."""

    if df.empty:
        LOGGER.info("Validation skipped because the DataFrame is empty")
        return df

    def _is_required(field: Any) -> bool:
        method: Callable[[], bool] | None = getattr(field, "is_required", None)
        if method is None:
            return False
        return method()

    required_fields = [
        name for name, field in schema.model_fields.items() if _is_required(field)
    ]
    missing_required = [field for field in required_fields if field not in df.columns]
    if missing_required:
        LOGGER.warning(
            "Input data is missing required columns: %s",
            ", ".join(sorted(missing_required)),
        )

    valid_rows: List[dict[str, Any]] = []
    errors: List[dict[str, Any]] = []

    def _normalise_error_details(
        details: Iterable[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        normalised: list[dict[str, Any]] = []
        for entry in details:
            clean_entry = dict(entry)
            ctx = clean_entry.get("ctx")
            if isinstance(ctx, Mapping):
                clean_ctx: dict[str, str] = {}
                for key, value in ctx.items():
                    clean_ctx[str(key)] = str(value)
                clean_entry["ctx"] = clean_ctx
            normalised.append(clean_entry)
        return normalised

    for index, row in df.iterrows():
        payload = _coerce_record(row)
        try:
            record = schema(**payload)
        except ValidationError as exc:
            LOGGER.warning("Validation error for row %s: %s", index, exc)
            if isinstance(index, (int, np.integer, float, np.floating)):
                row_index = int(index)
            else:
                row_index = int(str(index))
            errors.append(
                {
                    "index": row_index,
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
