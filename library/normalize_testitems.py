"""Normalisation helpers for ChEMBL molecule records.

The :func:`normalize_testitems` function flattens nested JSON structures
returned by the ChEMBL API into a tabular representation suitable for CSV
serialisation and validation.

Algorithm Notes
---------------
1. Coerce identifiers and categorical fields to upper-case, trimmed strings.
2. Extract nested dictionaries such as ``molecule_structures`` and
   ``molecule_properties`` into flat columns.
3. Normalise list-like attributes (synonyms, ATC codes, cross references)
   into deterministic, sorted representations.
4. Preserve numeric metadata as Python ``int``/``float`` types whenever
   possible, leaving textual descriptors untouched.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping

import pandas as pd

LOGGER = logging.getLogger(__name__)

_BOOL_TRUE = {"true", "t", "1", "yes", "y"}
_BOOL_FALSE = {"false", "f", "0", "no", "n"}


def _clean_identifier(value: Any) -> str | None:
    if value is None:
        return None
    candidate = str(value).strip()
    if not candidate:
        return None
    return candidate.upper()


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int,)):
        return int(value)
    if isinstance(value, float):
        if pd.isna(value):
            return None
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(float(stripped))
        except ValueError:
            return None
    return None


def _coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int,)):
        return bool(value)
    if isinstance(value, float):
        if pd.isna(value):
            return None
        return bool(value)
    if isinstance(value, str):
        stripped = value.strip().lower()
        if not stripped:
            return None
        if stripped in _BOOL_TRUE:
            return True
        if stripped in _BOOL_FALSE:
            return False
    return None


def _normalise_sequence(values: Iterable[Any]) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_string(value)
        if text is None:
            continue
        if text in seen:
            continue
        seen.add(text)
        items.append(text)
    items.sort()
    return items


def _normalise_cross_references(
    entries: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalised: list[dict[str, Any]] = []
    for entry in entries:
        xref_id = _clean_string(entry.get("xref_id"))
        xref_src = _clean_string(entry.get("xref_src"))
        xref_name = _clean_string(entry.get("xref_name"))
        if not any([xref_id, xref_src, xref_name]):
            continue
        normalised.append(
            {"xref_id": xref_id, "xref_src": xref_src, "xref_name": xref_name}
        )
    normalised.sort(
        key=lambda row: ((row.get("xref_src") or ""), (row.get("xref_id") or ""))
    )
    return normalised


def _normalise_synonyms(entries: Iterable[Mapping[str, Any]]) -> list[str]:
    synonyms = []
    for entry in entries:
        synonym = _clean_string(entry.get("synonyms"))
        if synonym is None:
            continue
        synonyms.append(synonym)
    return _normalise_sequence(synonyms)


def _coerce_property_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and pd.isna(value):
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


def normalize_testitems(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a normalized copy of the DataFrame with flattened molecule metadata.

    Args:
        df: The pandas DataFrame to normalize.

    Returns:
        A new DataFrame with normalized data.
    """

    if df.empty:
        return df

    records: list[dict[str, Any]] = []
    for raw in df.to_dict("records"):
        identifier = _clean_identifier(raw.get("molecule_chembl_id"))
        if identifier is None:
            LOGGER.warning("Skipping record without molecule_chembl_id: %s", raw)
            continue

        record: dict[str, Any] = {"molecule_chembl_id": identifier}
        record["pref_name"] = _clean_string(raw.get("pref_name"))
        record["molecule_type"] = _clean_string(raw.get("molecule_type"))
        record["structure_type"] = _clean_string(raw.get("structure_type"))
        record["chirality"] = _coerce_int(raw.get("chirality"))
        record["availability_type"] = _coerce_int(raw.get("availability_type"))
        record["max_phase"] = _coerce_int(raw.get("max_phase"))
        record["first_approval"] = _coerce_int(raw.get("first_approval"))
        record["usan_year"] = _coerce_int(raw.get("usan_year"))
        record["black_box_warning"] = _coerce_bool(raw.get("black_box_warning"))
        record["prodrug"] = _coerce_bool(raw.get("prodrug"))
        record["oral"] = _coerce_bool(raw.get("oral"))
        record["parenteral"] = _coerce_bool(raw.get("parenteral"))
        record["topical"] = _coerce_bool(raw.get("topical"))
        record["natural_product"] = _coerce_bool(raw.get("natural_product"))
        record["polymer_flag"] = _coerce_bool(raw.get("polymer_flag"))
        record["first_in_class"] = _coerce_bool(raw.get("first_in_class"))
        record["dosed_ingredient"] = _coerce_bool(raw.get("dosed_ingredient"))
        record["therapeutic_flag"] = _coerce_bool(raw.get("therapeutic_flag"))

        structures = raw.get("molecule_structures") or {}
        if isinstance(structures, Mapping):
            record["canonical_smiles"] = _clean_string(
                structures.get("canonical_smiles")
            )
            record["standard_inchi"] = _clean_string(structures.get("standard_inchi"))
            record["standard_inchi_key"] = _clean_string(
                structures.get("standard_inchi_key")
            )

        hierarchy = raw.get("molecule_hierarchy") or {}
        if isinstance(hierarchy, Mapping):
            parent_id = _clean_identifier(hierarchy.get("parent_chembl_id"))
            salt_id = _clean_identifier(hierarchy.get("molecule_chembl_id"))
            if parent_id and parent_id != identifier:
                record["parent_chembl_id"] = parent_id
            if salt_id and parent_id and salt_id != parent_id:
                record["salt_chembl_id"] = salt_id
            active_id = _clean_identifier(hierarchy.get("active_chembl_id"))
            if active_id and active_id not in {identifier, parent_id, salt_id}:
                record["active_chembl_id"] = active_id

        atc_codes = raw.get("atc_classifications")
        if isinstance(atc_codes, Iterable) and not isinstance(atc_codes, (str, bytes)):
            record["atc_classifications"] = _normalise_sequence(atc_codes)

        synonyms = raw.get("molecule_synonyms")
        if isinstance(synonyms, Iterable):
            synonym_entries = [
                entry for entry in synonyms if isinstance(entry, Mapping)
            ]
            if synonym_entries:
                record["synonyms"] = _normalise_synonyms(synonym_entries)

        cross_refs = raw.get("cross_references")
        if isinstance(cross_refs, Iterable):
            ref_entries = [entry for entry in cross_refs if isinstance(entry, Mapping)]
            if ref_entries:
                record["cross_references"] = _normalise_cross_references(ref_entries)

        properties = raw.get("molecule_properties")
        if isinstance(properties, Mapping):
            for key, value in properties.items():
                clean_key = f"chembl_{key}"
                record[clean_key] = _coerce_property_value(value)

        records.append(record)

    normalised = pd.DataFrame.from_records(records)
    for column in ["canonical_smiles", "standard_inchi", "standard_inchi_key"]:
        if column not in normalised.columns:
            normalised[column] = pd.NA

    return normalised


__all__ = ["normalize_testitems"]
