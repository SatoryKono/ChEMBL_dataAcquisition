"""Validation schema for normalised ChEMBL test item tables."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from pandas.api.types import is_scalar
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

LOGGER = logging.getLogger(__name__)


class TestitemsSchema(BaseModel):
    """Pydantic model describing a validated molecule record."""

    model_config = ConfigDict(extra="allow")

    molecule_chembl_id: str
    canonical_smiles: str
    standard_inchi_key: str
    pref_name: str | None = None
    molecule_type: str | None = None
    structure_type: str | None = None
    salt_chembl_id: str | None = None
    parent_chembl_id: str | None = None
    max_phase: int | None = None
    chembl_full_mwt: float | None = None
    chembl_alogp: float | None = None
    chembl_num_ro5_violations: int | None = None
    chembl_molecular_species: str | None = None
    pubchem_cid: int | None = None
    pubchem_molecular_formula: str | None = None
    pubchem_molecular_weight: float | None = None
    pubchem_tpsa: float | None = None
    pubchem_x_log_p: float | None = None
    pubchem_h_bond_donor_count: int | None = None
    pubchem_h_bond_acceptor_count: int | None = None
    pubchem_rotatable_bond_count: int | None = None

    @field_validator(
        "molecule_chembl_id", "canonical_smiles", "standard_inchi_key", mode="before"
    )
    @classmethod
    def _ensure_non_empty(cls, value: Any) -> str:
        if value is None:
            raise ValueError("value must not be empty")
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("value must not be empty")
            return stripped
        return str(value)

    @field_validator("max_phase", mode="before")
    @classmethod
    def _coerce_phase(cls, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                int_value = int(float(stripped))
            except ValueError as exc:  # pragma: no cover - defensive branch
                raise ValueError("max_phase must be an integer") from exc

            return int_value
        try:
            int_value = int(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive branch
            raise ValueError("max_phase must be an integer") from exc
        if int_value < 0:
            raise ValueError("max_phase must be non-negative")
        return int_value

    @field_validator(
        "chembl_full_mwt",
        "chembl_alogp",
        "chembl_num_ro5_violations",
        "pubchem_cid",
        "pubchem_molecular_weight",
        "pubchem_tpsa",
        "pubchem_x_log_p",
        "pubchem_h_bond_donor_count",
        "pubchem_h_bond_acceptor_count",
        "pubchem_rotatable_bond_count",
        mode="before",
    )
    @classmethod
    def _coerce_numeric(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            if isinstance(value, float) and np.isnan(value):
                return None
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                if any(char in stripped for char in [".", "e", "E"]):
                    return float(stripped)
                return int(stripped)
            except ValueError:
                return stripped
        return value

    @classmethod
    def ordered_columns(cls) -> list[str]:
        """Return the columns defined by the schema in declaration order."""

        return list(cls.model_fields.keys())


def _is_missing_scalar(value: Any) -> bool:
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    return False


def _coerce_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return []
        if value.ndim > 1:
            flattened = value.flatten()
            if flattened.size == 0:
                return []
            if flattened.size == 1:
                return _coerce_value(flattened.item())
            return [_coerce_value(item) for item in flattened.tolist()]
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


def validate_testitems(
    df: pd.DataFrame,
    schema: type[TestitemsSchema] = TestitemsSchema,
    *,
    errors_path: Path,
) -> pd.DataFrame:
    """Validate rows in ``df`` against ``schema`` and write failures."""

    if df.empty:
        LOGGER.info("Validation skipped because the DataFrame is empty")
        return df

    valid_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

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


__all__ = ["validate_testitems", "TestitemsSchema"]
