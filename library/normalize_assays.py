"""Deterministic normalisation utilities for assay tables."""

from __future__ import annotations

import json
import logging
from typing import Any, List

import pandas as pd

LOGGER = logging.getLogger(__name__)


_STRING_COLUMNS: List[str] = [
    "assay_category",
    "assay_cell_type",
    "assay_chembl_id",
    "assay_group",
    "assay_organism",
    "assay_strain",
    "assay_subcellular_fraction",
    "assay_tax_id",
    "assay_test_type",
    "assay_tissue",
    "assay_type",
    "assay_type_description",
    "bao_format",
    "bao_label",
    "cell_chembl_id",
    "confidence_description",
    "description",
    "document_chembl_id",
    "relationship_description",
    "relationship_type",
    "src_assay_id",
    "target_chembl_id",
    "tissue_chembl_id",
]

_NUMERIC_COLUMNS: List[str] = ["confidence_score", "src_id"]
_COLLECTION_COLUMNS: List[str] = [
    "assay_classifications",
    "assay_parameters",
    "variant_sequence",
]


def _normalise_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _normalise_collection(value: Any) -> list[Any]:
    if value in (None, "") or (isinstance(value, float) and pd.isna(value)):
        return []
    items: list[Any]
    if isinstance(value, list):
        items = list(value)
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


def normalize_assays(df: pd.DataFrame) -> pd.DataFrame:
    """Return a normalised copy of ``df``.

    Normalisation performs the following steps:

    * String-like columns are stripped of surrounding whitespace and empty
      values are replaced with ``None``.
    * Numeric columns are coerced into nullable integer dtype.
    * Collection columns (lists of dictionaries) are sorted deterministically
      with keys ordered alphabetically.
    """

    result = df.copy()

    for column in _STRING_COLUMNS:
        if column in result.columns:
            result[column] = result[column].map(_normalise_string)

    for column in _NUMERIC_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype(
                "Int64"
            )

    if "assay_with_same_target" in result.columns:
        result["assay_with_same_target"] = (
            pd.to_numeric(result["assay_with_same_target"], errors="coerce")
            .fillna(0)
            .astype("int64")
        )

    for column in _COLLECTION_COLUMNS:
        if column in result.columns:
            result[column] = result[column].map(_normalise_collection)

    return result
