"""Validation utilities for normalised activity tables."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Type

import pandas as pd
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


def _coerce_record(row: pd.Series) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            clean[key] = None
        else:
            clean[key] = value
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

    required_fields = [
        name
        for name, field in schema.model_fields.items()
        if field.is_required()  # type: ignore[attr-defined]
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
