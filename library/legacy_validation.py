"""Legacy Pydantic-based validators for benchmarking and regression tests."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Type

import numpy as np
import pandas as pd
from pandas.api.types import is_scalar
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

LOGGER = logging.getLogger(__name__)


class LegacyActivitiesSchema(BaseModel):
    """Historical schema describing a validated activity record."""

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


class LegacyAssaysSchema(BaseModel):
    """Historical schema describing a validated assay record."""

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


class LegacyTestitemsSchema(BaseModel):
    """Historical schema describing a validated molecule record."""

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
            except ValueError as exc:
                raise ValueError("max_phase must be an integer") from exc
            return int_value
        try:
            int_value = int(value)
        except (TypeError, ValueError) as exc:
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


def legacy_validate_activities(
    df: pd.DataFrame,
    schema: Type[LegacyActivitiesSchema] = LegacyActivitiesSchema,
    *,
    errors_path: Path,
) -> pd.DataFrame:
    if df.empty:
        LOGGER.info("Validation skipped because the DataFrame is empty")
        return df

    required_fields = [
        name for name, field in schema.model_fields.items() if field.is_required()
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


def legacy_validate_assays(
    df: pd.DataFrame,
    schema: Type[LegacyAssaysSchema] = LegacyAssaysSchema,
    *,
    errors_path: Path,
) -> pd.DataFrame:
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


def legacy_validate_testitems(
    df: pd.DataFrame,
    schema: type[LegacyTestitemsSchema] = LegacyTestitemsSchema,
    *,
    errors_path: Path,
) -> pd.DataFrame:
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


__all__ = [
    "LegacyActivitiesSchema",
    "LegacyAssaysSchema",
    "LegacyTestitemsSchema",
    "legacy_validate_activities",
    "legacy_validate_assays",
    "legacy_validate_testitems",
]
