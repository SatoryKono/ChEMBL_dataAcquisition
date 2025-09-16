"""Deterministic normalisation utilities for activity tables."""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable

import pandas as pd

LOGGER = logging.getLogger(__name__)


_STRING_COLUMNS: list[str] = [
    "activity_chembl_id",
    "assay_chembl_id",
    "document_chembl_id",
    "molecule_chembl_id",
    "parent_molecule_chembl_id",
    "target_chembl_id",
    "standard_type",
    "standard_relation",
    "standard_units",
    "type",
    "relation",
    "units",
    "uo_units",
    "qudt_units",
    "activity_comment",
    "data_validity_comment",
]

_FLOAT_COLUMNS: list[str] = [
    "standard_value",
    "standard_upper_value",
    "standard_lower_value",
    "pchembl_value",
    "activity_value",
]

_INT_COLUMNS: list[str] = ["record_id", "activity_id", "standard_flag"]
_BOOLEAN_COLUMNS: list[str] = ["potential_duplicate", "data_validity_warning"]
_MAPPING_COLUMNS: list[str] = [
    "ligand_efficiency",
    "molecule_properties",
    "molecule_hierarchy",
]
_COLLECTION_COLUMNS: list[str] = ["activity_properties", "target_components"]


def _normalise_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if pd.isna(value):  # type: ignore[arg-type]
        return None
    text = str(value).strip()
    return text or None


def _normalise_numeric(value: Any) -> float | None:
    if value in (None, "") or (isinstance(value, float) and pd.isna(value)):
        return None
    numeric = pd.to_numeric([value], errors="coerce")[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _normalise_integer(value: Any) -> int | None:
    numeric = _normalise_numeric(value)
    if numeric is None:
        return None
    return int(numeric)


def _normalise_boolean(value: Any) -> bool | None:
    if value in (None, "") or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):  # type: ignore[arg-type]
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes"}:
        return True
    if text in {"false", "f", "0", "no"}:
        return False
    LOGGER.debug("Unable to normalise boolean value: %s", value)
    return None


def _normalise_mapping(value: Any) -> Any:
    if value in (None, "") or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, dict):
        return {key: value.get(key) for key in sorted(value)}
    return value


def _normalise_collection(value: Any) -> list[Any]:
    if value in (None, "") or (isinstance(value, float) and pd.isna(value)):
        return []
    items: Iterable[Any]
    if isinstance(value, list):
        items = value
    else:
        items = [value]
    normalised: list[Any] = []
    for item in items:
        if isinstance(item, dict):
            normalised.append({key: item.get(key) for key in sorted(item)})
        else:
            normalised.append(item)
    normalised.sort(key=lambda obj: json.dumps(obj, ensure_ascii=False, sort_keys=True))
    return normalised


def normalize_activities(df: pd.DataFrame) -> pd.DataFrame:
    """Return a normalised copy of ``df``.

    Normalisation performs the following steps:

    * String-like columns are stripped of surrounding whitespace and empty
      values are replaced with ``None``.
    * Numeric columns are converted to ``Float64`` or ``Int64`` where
      applicable.
    * Boolean columns are cast to pandas' nullable boolean dtype.
    * Mapping and collection columns are sorted deterministically.
    """

    result = df.copy()

    for column in _STRING_COLUMNS:
        if column in result.columns:
            result[column] = result[column].map(_normalise_string)

    for column in _FLOAT_COLUMNS:
        if column in result.columns:
            result[column] = result[column].map(_normalise_numeric).astype("Float64")

    for column in _INT_COLUMNS:
        if column in result.columns:
            result[column] = result[column].map(_normalise_integer).astype("Int64")

    for column in _BOOLEAN_COLUMNS:
        if column in result.columns:
            result[column] = result[column].map(_normalise_boolean).astype("boolean")

    for column in _MAPPING_COLUMNS:
        if column in result.columns:
            result[column] = result[column].map(_normalise_mapping)

    for column in _COLLECTION_COLUMNS:
        if column in result.columns:
            result[column] = result[column].map(_normalise_collection)

    return result
