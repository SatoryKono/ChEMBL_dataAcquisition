"""CLI entry point for the unified target data pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Callable, Dict, List

import pandas as pd
import requests
import yaml
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    from _path_utils import ensure_project_root as _ensure_project_root

    _ensure_project_root()


from library.chembl_targets import TargetConfig, fetch_targets
from library.gtop_client import GtoPClient, GtoPConfig
from library.hgnc_client import HGNCClient, load_config as load_hgnc_config
from library.uniprot_client import (
    NetworkConfig as UniNetworkConfig,
    RateLimitConfig as UniRateConfig,
    UniProtClient,
)
from library.orthologs import EnsemblHomologyClient, OmaClient
from library.http_client import CacheConfig
from library.uniprot_enrich.enrich import (
    UniProtClient as UniProtEnrichClient,
    _collect_ec_numbers,
)
from library.logging_utils import configure_logging

from library.cli_common import (
    analyze_table_quality,
    ensure_output_dir,
    serialise_dataframe,
    write_cli_metadata,
)

from library.protein_classifier import classify_protein


from library.pipeline_targets import (
    PipelineConfig,
    load_pipeline_config,
    run_pipeline,
)

from library.iuphar import ClassificationRecord, IUPHARClassifier, IUPHARData


# Columns produced by :func:`add_iuphar_classification`.
DEFAULT_LOG_FORMAT = "human"


IUPHAR_CLASS_COLUMNS = [
    "iuphar_target_id",
    "iuphar_family_id",
    "iuphar_type",
    "iuphar_class",
    "iuphar_subclass",
    "iuphar_chain",
    "iuphar_name",
    "iuphar_full_id_path",
    "iuphar_full_name_path",
]


def merge_chembl_fields(
    pipeline_df: pd.DataFrame, chembl_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge ChEMBL-specific columns into the pipeline output.

    Parameters
    ----------
    pipeline_df:
        Data frame produced by :func:`run_pipeline`.
    chembl_df:
        Data frame containing raw ChEMBL target information.

    Returns
    -------
    pandas.DataFrame
        Combined data frame with additional ChEMBL columns appended. Existing
        columns in ``pipeline_df`` are preserved; overlapping columns from
        ``chembl_df`` are ignored to avoid duplication.
    """

    extra_cols = [c for c in chembl_df.columns if c not in pipeline_df.columns]
    if extra_cols:
        pipeline_df = pipeline_df.merge(
            chembl_df[["target_chembl_id", *extra_cols]],
            on="target_chembl_id",
            how="left",
        )
    return pipeline_df


def _parse_species_argument(raw_value: str | None) -> list[str]:
    """Return CLI-provided species values as an ordered, de-duplicated list.

    Parameters
    ----------
    raw_value:
        Raw command line value provided via ``--species``. Individual entries
        may be separated by commas. Empty fragments are ignored so accidental
        trailing separators do not introduce blank species names.

    Returns
    -------
    list[str]
        Cleaned species names preserving their original order of appearance.
        Duplicate entries are collapsed to the first occurrence to keep the
        priority list concise.
    """

    if raw_value is None:
        return []
    candidates = [fragment.strip() for fragment in raw_value.split(",")]
    seen: set[str] = set()
    cleaned: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        if candidate in seen:
            continue
        cleaned.append(candidate)
        seen.add(candidate)
    return cleaned


def _merge_species_lists(
    cli_values: Sequence[str], config_values: Sequence[str]
) -> list[str]:
    """Combine species lists from CLI and configuration without duplicates.

    Parameters
    ----------
    cli_values:
        Ordered species preference supplied via the command line.
    config_values:
        Baseline species priority sourced from the YAML configuration file.

    Returns
    -------
    list[str]
        Combined list where CLI-specified species retain precedence but
        configuration values are preserved afterwards. Duplicate entries are
        removed while keeping the order of first appearance across both
        sequences.
    """

    combined: list[str] = []
    seen: set[str] = set()
    for value in (*cli_values, *config_values):
        if value in seen:
            continue
        combined.append(value)
        seen.add(value)
    return combined


def _ensure_mapping(
    value: Any, *, context: str, allow_none: bool = True
) -> dict[str, Any]:
    """Return ``value`` as a mapping with defensive type validation."""

    if value is None:
        if allow_none:
            return {}
        msg = f"Expected '{context}' to be a mapping, not null"
        raise TypeError(msg)
    if isinstance(value, Mapping):
        return dict(value)
    msg = f"Expected '{context}' to be a mapping, not {type(value).__name__!s}"
    raise TypeError(msg)


def _optional_mapping(
    mapping: Mapping[str, Any], key: str, *, context: str
) -> dict[str, Any] | None:
    """Return an optional nested mapping with validation."""

    raw_value = mapping.get(key)
    if raw_value is None:
        return None
    if isinstance(raw_value, Mapping):
        return dict(raw_value)
    msg = f"Expected '{context}' to be a mapping, not {type(raw_value).__name__!s}"
    raise TypeError(msg)


def _ensure_bool(value: Any, *, context: str, default: bool = False) -> bool:
    """Return a boolean configuration value validating its type."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    msg = f"Expected '{context}' to be a boolean, not {type(value).__name__!s}"
    raise TypeError(msg)


def _ensure_str_sequence(value: Any, *, context: str) -> list[str]:
    """Return ``value`` as a list of strings preserving order."""

    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        result: list[str] = []
        for item in value:
            if not isinstance(item, str):
                msg = (
                    f"Expected all entries in '{context}' to be strings, "
                    f"found {type(item).__name__!s}"
                )
                raise TypeError(msg)
            result.append(item)
        return result
    msg = f"Expected '{context}' to be a sequence of strings"
    raise TypeError(msg)


def _load_yaml_mapping(path: str) -> dict[str, Any]:
    """Load a YAML file ensuring it contains a mapping at the top level."""

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return _ensure_mapping(data, context=f"{path} root", allow_none=False)


def _apply_env_overrides(
    data: dict[str, Any], *, section: str | None = None
) -> dict[str, Any]:
    """Return ``data`` updated with values from environment variables.

    Parameters
    ----------
    data:
        Mutable configuration dictionary to update in-place.
    section:
        Optional name of a specific configuration section. When supplied the
        helper assumes that the first component of the environment variable
        path refers to this section and strips it accordingly. This mirrors the
        behaviour implemented for the chembl2uniprot CLI utilities.

    Returns
    -------
    dict[str, Any]
        The dictionary with applied overrides. The same instance that was
        passed in is returned to facilitate fluent usage patterns.
    """

    prefixes = ("CHEMBL_DA__", "CHEMBL_")
    section_lower = section.lower() if section else None
    for raw_key, value in os.environ.items():
        existing_keys = {str(key).lower() for key in data}
        matched_prefix = next((p for p in prefixes if raw_key.startswith(p)), None)
        if matched_prefix is None:
            continue
        tail = raw_key[len(matched_prefix) :]
        if not tail:
            continue
        path = [part.lower() for part in tail.split("__") if part]
        if not path:
            continue
        if matched_prefix == "CHEMBL_DA__":
            if section_lower and path[0] == section_lower:
                path = path[1:]
            elif not section_lower and path[0] not in existing_keys and len(path) > 1:
                path = path[1:]
        else:
            if section_lower:
                if path[0] != section_lower:
                    continue
                path = path[1:]
        if not path:
            continue
        ref: dict[str, Any] | Any = data
        valid_path = True
        for part in path[:-1]:
            if not isinstance(ref, dict):
                valid_path = False
                break
            ref = ref.setdefault(part, {})
        if not valid_path or not isinstance(ref, dict):
            continue
        ref[path[-1]] = value
    return data


@dataclass
class SectionSettings:
    """Normalised network, rate limit and cache settings for a section."""

    timeout_sec: float
    max_retries: int
    backoff_sec: float
    rps: float
    cache: CacheConfig | None


def _coerce_float(value: Any, *, default: float, context: str) -> float:
    """Return ``value`` as a float while honouring ``default`` for nulls."""

    if value is None:
        return float(default)
    if isinstance(value, bool):
        msg = f"Expected '{context}' to be numeric, not {type(value).__name__!s}"
        raise TypeError(msg)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        msg = f"Expected '{context}' to be numeric, received {value!r}"
        raise TypeError(msg) from exc


def _coerce_int(value: Any, *, default: int, context: str) -> int:
    """Return ``value`` as an integer while honouring ``default`` for nulls."""

    if value is None:
        return int(default)
    if isinstance(value, bool):
        msg = f"Expected '{context}' to be an integer, not {type(value).__name__!s}"
        raise TypeError(msg)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        msg = f"Expected '{context}' to be an integer, received {value!r}"
        raise TypeError(msg) from exc


def _resolve_section_settings(
    section_name: str,
    section_cfg: Mapping[str, Any],
    *,
    pipeline_cfg: PipelineConfig,
    global_cache: CacheConfig | None,
    default_backoff: float = 1.0,
) -> SectionSettings:
    """Return normalised timeout, retry, rate limit and cache settings."""

    network_cfg_raw = section_cfg.get("network")
    network_cfg = (
        _ensure_mapping(network_cfg_raw, context=f"{section_name}.network")
        if isinstance(network_cfg_raw, Mapping)
        else None
    )

    def _network_value(*keys: str) -> Any:
        for key in keys:
            if network_cfg is not None and network_cfg.get(key) is not None:
                return network_cfg.get(key)
            if section_cfg.get(key) is not None:
                return section_cfg.get(key)
        return None

    timeout_sec = _coerce_float(
        _network_value("timeout_sec", "timeout"),
        default=pipeline_cfg.timeout_sec,
        context=f"{section_name}.timeout_sec",
    )
    max_retries = _coerce_int(
        _network_value("max_retries", "retries"),
        default=pipeline_cfg.retries,
        context=f"{section_name}.retries",
    )
    backoff_sec = _coerce_float(
        _network_value("backoff_sec", "backoff_base_sec", "backoff"),
        default=default_backoff,
        context=f"{section_name}.backoff",
    )

    rate_limit_raw = section_cfg.get("rate_limit")
    rate_limit_cfg = (
        _ensure_mapping(rate_limit_raw, context=f"{section_name}.rate_limit")
        if isinstance(rate_limit_raw, Mapping)
        else None
    )

    def _rate_value(*keys: str) -> Any:
        for key in keys:
            if rate_limit_cfg is not None and rate_limit_cfg.get(key) is not None:
                return rate_limit_cfg.get(key)
            if section_cfg.get(key) is not None:
                return section_cfg.get(key)
        return None

    rps = _coerce_float(
        _rate_value("rps", "rate_limit_rps"),
        default=pipeline_cfg.rate_limit_rps,
        context=f"{section_name}.rps",
    )

    cache_cfg = _optional_mapping(section_cfg, "cache", context=f"{section_name}.cache")
    cache = CacheConfig.from_dict(cache_cfg) or global_cache

    return SectionSettings(
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        backoff_sec=backoff_sec,
        rps=rps,
        cache=cache,
    )


def _normalise_field_names(value: Any, *, source: str) -> list[str]:
    """Return ``value`` as a list of stripped field names.

    Parameters
    ----------
    value:
        Configuration value representing UniProt field names.
    source:
        Human-readable label used in error messages to indicate the origin of
        the data.

    Returns
    -------
    list[str]
        Field names stripped of surrounding whitespace. Empty strings are
        discarded to avoid passing blank field names to the UniProt API.

    Raises
    ------
    TypeError
        If ``value`` is neither a string nor an iterable of strings.
    """

    if value is None:
        return []
    if isinstance(value, str):
        candidate = value.strip()
        return [candidate] if candidate else []
    if isinstance(value, Mapping):
        msg = f"Expected '{source}' to be a sequence of strings, not a mapping"
        raise TypeError(msg)
    if not isinstance(value, Iterable):
        msg = f"Expected '{source}' to be a string or iterable of strings"
        raise TypeError(msg)
    names: list[str] = []
    for entry in value:
        if entry is None:
            continue
        if not isinstance(entry, str):
            msg = (
                f"Expected all values in '{source}' to be strings, received "
                f"{type(entry).__name__}"
            )
            raise TypeError(msg)
        candidate = entry.strip()
        if candidate:
            names.append(candidate)
    return names


def _resolve_uniprot_fields(uni_cfg: Mapping[str, Any]) -> str:
    """Return the UniProt ``fields`` parameter derived from ``uni_cfg``.

    The configuration may specify the desired UniProt fields explicitly via the
    ``fields`` key. When omitted, the helper falls back to the ``columns``
    configuration so existing deployments that only list output columns continue
    to function without modification.

    Parameters
    ----------
    uni_cfg:
        Mapping containing UniProt configuration options loaded from the YAML
        file.

    Returns
    -------
    str
        Comma separated string suitable for the UniProt REST API ``fields``
        parameter. The string is empty when neither ``fields`` nor ``columns``
        are configured.
    """

    field_names = _normalise_field_names(uni_cfg.get("fields"), source="uniprot.fields")
    if field_names:
        return ",".join(field_names)
    column_names = _normalise_field_names(
        uni_cfg.get("columns"), source="uniprot.columns"
    )
    return ",".join(column_names)


def add_iuphar_classification(
    pipeline_df: pd.DataFrame,
    target_csv: str | Path,
    family_csv: str | Path,
    *,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Append IUPHAR classification columns to ``pipeline_df``.

    Parameters
    ----------
    pipeline_df:
        Data frame produced by :func:`run_pipeline`.
    target_csv:
        Path to the ``_IUPHAR_target.csv`` file.
    family_csv:
        Path to the ``_IUPHAR_family.csv`` file.
    encoding:
        File encoding used when loading the IUPHAR tables.

    Returns
    -------
    pandas.DataFrame
        ``pipeline_df`` extended with classification fields. Existing
        columns are preserved.
    """

    data = IUPHARData.from_files(target_csv, family_csv, encoding=encoding)
    classifier = IUPHARClassifier(data)

    def _classify(row: pd.Series) -> pd.Series:
        # Prioritise an explicit GtoP target ID if available.
        target_id = row.get("gtop_target_id", "")
        if not target_id:
            target_id = data.target_id_by_uniprot(row.get("uniprot_id_primary", ""))
        if not target_id:
            target_id = data.target_id_by_hgnc_name(row.get("hgnc_name", ""))
        if not target_id:
            target_id = data.target_id_by_hgnc_id(row.get("hgnc_id", ""))
        if not target_id:
            target_id = data.target_id_by_gene(row.get("gene_symbol", ""))
        if not target_id:
            synonyms = str(row.get("synonyms_all", "")).split("|")
            mapped = data.target_ids_by_synonyms(synonyms)
            # Ignore ambiguous mappings returning multiple IDs.
            if mapped and "|" not in mapped:
                target_id = mapped
        record = (
            classifier.by_target_id(target_id) if target_id else ClassificationRecord()
        )
        return pd.Series(
            {
                "iuphar_target_id": record.IUPHAR_target_id,
                "iuphar_family_id": record.IUPHAR_family_id,
                "iuphar_type": record.IUPHAR_type,
                "iuphar_class": record.IUPHAR_class,
                "iuphar_subclass": record.IUPHAR_subclass,
                "iuphar_chain": ">".join(record.IUPHAR_tree),
                "iuphar_name": record.IUPHAR_name,
                "iuphar_full_id_path": (
                    data.all_id(record.IUPHAR_target_id)
                    if record.IUPHAR_target_id != "N/A"
                    else ""
                ),
                "iuphar_full_name_path": (
                    data.all_name(record.IUPHAR_target_id)
                    if record.IUPHAR_target_id != "N/A"
                    else ""
                ),
            }
        )

    class_df = pipeline_df.apply(_classify, axis=1)
    return pd.concat([pipeline_df, class_df], axis=1)


def add_protein_classification(
    pipeline_df: pd.DataFrame,
    fetch_entries: Callable[[Iterable[str]], Dict[str, Any]],
) -> pd.DataFrame:
    """Append automated protein classification columns.

    Parameters
    ----------
    pipeline_df:
        Data frame produced by :func:`run_pipeline`.
    fetch_entries:
        Callable returning a mapping of UniProt accession to the corresponding
        JSON entry for a set of accessions.

    Returns
    -------
    pandas.DataFrame
        ``pipeline_df`` extended with predicted classification fields. The
        following columns are added:

        ``protein_class_pred_L1``, ``protein_class_pred_L2``,
        ``protein_class_pred_L3``, ``protein_class_pred_rule_id``,
        ``protein_class_pred_evidence`` and
        ``protein_class_pred_confidence``.
    """

    columns = [
        "protein_class_pred_L1",
        "protein_class_pred_L2",
        "protein_class_pred_L3",
        "protein_class_pred_rule_id",
        "protein_class_pred_evidence",
        "protein_class_pred_confidence",
    ]

    ids = pipeline_df.get("uniprot_id_primary", pd.Series(dtype=str)).astype(str)
    unique_ids = [acc for acc in dict.fromkeys(ids) if acc]
    logger = logging.getLogger(__name__)
    entry_map: Dict[str, Any] = {}
    fetched_entries: Dict[str, Any] = {}
    if unique_ids:
        try:
            fetched_entries = fetch_entries(unique_ids) or {}
        except requests.RequestException as exc:
            sample_ids = ", ".join(unique_ids[:3])
            msg = (
                "Network error while fetching UniProt entries "
                f"for {len(unique_ids)} accessions"
            )
            if sample_ids:
                msg = f"{msg}: {sample_ids}"
            raise RuntimeError(msg) from exc
        except Exception as exc:  # pragma: no cover - logging side effect
            logger.warning("Failed to fetch UniProt entries: %s", exc)
            fetched_entries = {}

    for acc in unique_ids:
        entry = fetched_entries.get(acc)
        if entry is not None:
            entry_map[acc] = entry

    def _classify(acc: str) -> pd.Series:
        entry = entry_map.get(acc)
        result = classify_protein(entry) if entry else {}
        evidence = result.get("evidence", [])
        return pd.Series(
            {
                "protein_class_pred_L1": result.get("protein_class_L1", ""),
                "protein_class_pred_L2": result.get("protein_class_L2", ""),
                "protein_class_pred_L3": result.get("protein_class_L3", ""),
                "protein_class_pred_rule_id": result.get("rule_id", ""),
                "protein_class_pred_evidence": "|".join(evidence),
                "protein_class_pred_confidence": result.get("confidence", ""),
            }
        )

    class_df = ids.apply(_classify)
    for col in columns:
        if col not in class_df.columns:
            class_df[col] = ""
    return pd.concat([pipeline_df, class_df], axis=1)


def add_uniprot_fields(
    pipeline_df: pd.DataFrame,
    fetch_all: Callable[[Iterable[str]], Dict[str, Dict[str, str]]],
) -> pd.DataFrame:
    """Append supplementary UniProt annotations to ``pipeline_df``.

    Parameters
    ----------
    pipeline_df:
        Data frame produced by :func:`run_pipeline` and containing a
        ``uniprot_id_primary`` column.
    fetch_all:
        Callable returning a mapping from UniProt accession to a dictionary of
        annotation fields. Typically this is
        :meth:`library.uniprot_enrich.enrich.UniProtClient.fetch_all`.

    Returns
    -------
    pandas.DataFrame
        ``pipeline_df`` with additional UniProt fields appended. Existing
        columns are left untouched.
    """

    # Mapping of output column names to keys in the annotation dictionary.
    col_map = {
        "uniProtkbId": "uniprotkb_Id",
        "secondaryAccessions": "secondary_uniprot_id",
        "recommendedName": "recommended_name",
        "geneName": "gene_name",
        "secondaryAccessionNames": "secondary_accession_names",
        "molecular_function": "molecular_function",
        "cellular_component": "cellular_component",
        "subcellular_location": "subcellular_location",
        "topology": "topology",
        "transmembrane": "transmembrane",
        "intramembrane": "intramembrane",
        "glycosylation": "glycosylation",
        "lipidation": "lipidation",
        "disulfide_bond": "disulfide_bond",
        "modified_residue": "modified_residue",
        "phosphorylation": "phosphorylation",
        "acetylation": "acetylation",
        "ubiquitination": "ubiquitination",
        "signal_peptide": "signal_peptide",
        "propeptide": "propeptide",
        "GuidetoPHARMACOLOGY": "GuidetoPHARMACOLOGY",
        "family": "family",
        "SUPFAM": "SUPFAM",
        "PROSITE": "PROSITE",
        "InterPro": "InterPro",
        "Pfam": "Pfam",
        "PRINTS": "PRINTS",
        "TCDB": "TCDB",
    }

    ids = pipeline_df.get("uniprot_id_primary", pd.Series(dtype=str)).astype(str)
    mapping = fetch_all([i for i in ids if i])

    for out_col, src_col in col_map.items():
        if out_col in pipeline_df.columns:
            # Respect existing columns to avoid overwriting prior values.
            continue
        pipeline_df[out_col] = [mapping.get(i, {}).get(src_col, "") for i in ids]
    return pipeline_df


def extract_activity(data: Any) -> dict[str, str]:
    """Return catalytic reaction names and EC numbers found in ``data``.

    The UniProt record may list one or more "CATALYTIC ACTIVITY" comments,
    each describing a reaction and an associated EC number. This helper
    aggregates those reactions and numbers as pipe-separated strings.

    Parameters
    ----------
    data:
        A UniProt JSON structure, list of entries, or search results
        containing UniProt entries.

    Returns
    -------
    dict[str, str]
        A dictionary with keys ``reactions`` and ``reaction_ec_numbers``.
        Missing information yields empty strings.
    """

    reactions: list[str] = []
    numbers: list[str] = []
    if isinstance(data, dict) and "results" in data:
        entries = data["results"]
    elif isinstance(data, list):
        entries = data
    else:
        entries = [data]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        comments = entry.get("comments", [])
        if not isinstance(comments, list):
            continue
        for comment in comments:
            if not isinstance(comment, dict):
                continue
            if comment.get("commentType") != "CATALYTIC ACTIVITY":
                continue
            reaction = comment.get("reaction")
            if not isinstance(reaction, dict):
                continue
            name = reaction.get("name")
            if isinstance(name, dict):
                name = name.get("value")
            if isinstance(name, str):
                reactions.append(name)
            numbers.extend(list(_collect_ec_numbers(reaction)))
    return {
        "reactions": "|".join(reactions),
        "reaction_ec_numbers": "|".join(numbers),
    }


def add_activity_fields(
    pipeline_df: pd.DataFrame, fetch_entry: Callable[[str], Any]
) -> pd.DataFrame:
    """Append catalytic activity and EC numbers parsed from UniProt entries.

    Parameters
    ----------
    pipeline_df:
        Data frame produced by :func:`run_pipeline` containing a
        ``uniprot_id_primary`` column.
    fetch_entry:
        Callable returning a UniProt JSON entry for a given accession.

    Returns
    -------
    pandas.DataFrame
        ``pipeline_df`` with ``reactions`` and ``reaction_ec_numbers``
        columns populated. Existing columns are preserved.
    """

    ids = pipeline_df.get("uniprot_id_primary", pd.Series(dtype=str)).astype(str)
    cache: Dict[str, dict[str, str]] = {}
    for acc in ids:
        if not acc or acc in cache:
            continue
        entry = fetch_entry(acc)
        cache[acc] = (
            extract_activity(entry)
            if entry
            else {"reactions": "", "reaction_ec_numbers": ""}
        )
    pipeline_df = pipeline_df.copy()
    pipeline_df["reactions"] = [cache.get(i, {}).get("reactions", "") for i in ids]
    pipeline_df["reaction_ec_numbers"] = [
        cache.get(i, {}).get("reaction_ec_numbers", "") for i in ids
    ]
    return pipeline_df


def extract_isoform(data: Any) -> dict[str, str]:
    """Return isoform information found in ``data``.

    The function inspects ``ALTERNATIVE PRODUCTS`` comments and gathers the
    names, IDs, and synonyms for each isoform. Multiple IDs or synonyms within
    an isoform are joined by ``":"`` while separate isoforms are joined by
    ``"|"``. When no isoform data is available, the strings ``"None"`` are
    returned for all fields.

    Parameters
    ----------
    data:
        A UniProt JSON structure, list of entries, or search results containing
        UniProt entries.

    Returns
    -------
    dict[str, str]
        Mapping with keys ``isoform_names``, ``isoform_ids`` and
        ``isoform_synonyms`` containing pipe-separated strings.
    """

    names: list[str] = []
    ids: list[str] = []
    syns: list[str] = []
    if isinstance(data, dict) and "results" in data:
        entries = data["results"]
    elif isinstance(data, list):
        entries = data
    else:
        entries = [data]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        comments = entry.get("comments", [])
        if not isinstance(comments, list):
            continue
        for comment in comments:
            if (
                not isinstance(comment, dict)
                or comment.get("commentType") != "ALTERNATIVE PRODUCTS"
            ):
                continue
            isoforms = comment.get("isoforms", [])
            if not isinstance(isoforms, list):
                continue
            for iso in isoforms:
                if not isinstance(iso, dict):
                    continue
                name = None
                name_obj = iso.get("name")
                if isinstance(name_obj, dict):
                    name = name_obj.get("value")
                if isinstance(name, str):
                    names.append(name)
                iso_ids: list[str] = []
                for iid in iso.get("isoformIds", []) or []:
                    if isinstance(iid, str):
                        iso_ids.append(iid)
                ids.append(":".join(iso_ids) if iso_ids else "N/A")
                syn_list: list[str] = []
                for syn in iso.get("synonyms", []) or []:
                    if isinstance(syn, dict):
                        value = syn.get("value")
                        if isinstance(value, str):
                            syn_list.append(value)
                syns.append(":".join(syn_list) if syn_list else "N/A")
    result = {
        "isoform_names": "|".join(names) if names else "None",
        "isoform_ids": "|".join(ids) if names else "None",
        "isoform_synonyms": "|".join(syns) if names else "None",
    }
    return result


def add_isoform_fields(
    pipeline_df: pd.DataFrame, fetch_entry: Callable[[str], Any]
) -> pd.DataFrame:
    """Append isoform data parsed from UniProt entries.

    Parameters
    ----------
    pipeline_df:
        Data frame produced by :func:`run_pipeline` containing a
        ``uniprot_id_primary`` column.
    fetch_entry:
        Callable returning a UniProt JSON entry for a given accession.

    Returns
    -------
    pandas.DataFrame
        ``pipeline_df`` with ``isoform_names``, ``isoform_ids`` and
        ``isoform_synonyms`` columns populated. Existing columns are preserved.
    """

    ids = pipeline_df.get("uniprot_id_primary", pd.Series(dtype=str)).astype(str)
    cache: Dict[str, dict[str, str]] = {}
    for acc in ids:
        if not acc or acc in cache:
            continue
        entry = fetch_entry(acc)
        cache[acc] = (
            extract_isoform(entry)
            if entry
            else {
                "isoform_names": "None",
                "isoform_ids": "None",
                "isoform_synonyms": "None",
            }
        )
    pipeline_df = pipeline_df.copy()
    pipeline_df["isoform_names"] = [
        cache.get(i, {}).get("isoform_names", "None") for i in ids
    ]
    pipeline_df["isoform_ids"] = [
        cache.get(i, {}).get("isoform_ids", "None") for i in ids
    ]
    pipeline_df["isoform_synonyms"] = [
        cache.get(i, {}).get("isoform_synonyms", "None") for i in ids
    ]
    return pipeline_df


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the pipeline."""

    argv = list(sys.argv[1:] if argv is None else argv)

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the pipeline configuration YAML file",
    )
    preliminary, _ = config_parser.parse_known_args(argv)
    try:
        config_data = _load_yaml_mapping(preliminary.config)
    except FileNotFoundError:
        config_data = {}
    orthologs_cfg = _ensure_mapping(config_data.get("orthologs"), context="orthologs")
    orthologs_default = _ensure_bool(
        orthologs_cfg.get("enabled"), context="orthologs.enabled", default=False
    )

    parser = argparse.ArgumentParser(
        description="Unified target data pipeline",
        parents=[config_parser],
    )
    parser.set_defaults(with_orthologs=orthologs_default)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--id-column", default="target_chembl_id")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--log-format",
        default=DEFAULT_LOG_FORMAT,
        choices=("human", "json"),
        help="Logging output format (human or json)",
    )
    parser.add_argument("--sep", default=",")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--list-format", default=None)
    parser.add_argument("--species", default=None)
    parser.add_argument("--affinity-parameter", default="pKi")
    parser.add_argument("--approved-only", default=None)
    parser.add_argument("--primary-target-only", default=None)
    parser.add_argument("--with-isoforms", action="store_true")
    parser.add_argument(
        "--with-orthologs",
        action=argparse.BooleanOptionalAction,
        help=(
            "Enable or disable ortholog enrichment. Defaults to the YAML "
            "configuration when neither flag is supplied."
        ),
    )
    parser.add_argument("--iuphar-target", help="Path to _IUPHAR_target.csv")
    parser.add_argument("--iuphar-family", help="Path to _IUPHAR_family.csv")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Maximum number of IDs per network request",
    )
    parser.add_argument(
        "--meta-output",
        default=None,
        help="Optional metadata YAML path",
    )
    return parser.parse_args(argv)


def build_clients(
    cfg_path: str,
    pipeline_cfg: PipelineConfig,
    *,
    with_orthologs: bool = False,
    default_cache: CacheConfig | None = None,
) -> tuple[
    UniProtClient,
    HGNCClient,
    GtoPClient,
    EnsemblHomologyClient | None,
    OmaClient | None,
    list[str],
]:
    """Initialise service clients used by the pipeline.

    Parameters
    ----------
    cfg_path:
        Path to the YAML configuration file.
    pipeline_cfg:
        High-level pipeline configuration controlling retries and rate limits.
    with_orthologs:
        When ``True`` return ortholog clients in addition to the core clients.
    default_cache:
        Optional fallback cache configuration applied when a section does not
        specify its own cache settings.
    """

    data = _load_yaml_mapping(cfg_path)
    _apply_env_overrides(data)
    global_cache = default_cache or CacheConfig.from_dict(
        _optional_mapping(data, "http_cache", context="http_cache")
    )

    uni_cfg = _ensure_mapping(data.get("uniprot"), context="uniprot", allow_none=False)
    uni_settings = _resolve_section_settings(
        "uniprot",
        uni_cfg,
        pipeline_cfg=pipeline_cfg,
        global_cache=global_cache,
    )
    fields = _resolve_uniprot_fields(uni_cfg)
    uni = UniProtClient(
        base_url=str(uni_cfg["base_url"]),
        fields=fields,
        network=UniNetworkConfig(
            timeout_sec=uni_settings.timeout_sec,
            max_retries=uni_settings.max_retries,
            backoff_sec=uni_settings.backoff_sec,
        ),
        rate_limit=UniRateConfig(rps=uni_settings.rps),
        cache=uni_settings.cache,
    )

    # The HGNC configuration is nested under the top-level "hgnc" section
    # in the YAML file. Explicitly select this section to avoid passing the
    # entire configuration dictionary to ``HGNCServiceConfig``, which would
    # otherwise raise ``TypeError`` due to unexpected keys.
    hcfg = load_hgnc_config(cfg_path, section="hgnc")
    hgnc_cfg = _ensure_mapping(data.get("hgnc"), context="hgnc")
    hgnc_settings = _resolve_section_settings(
        "hgnc",
        hgnc_cfg,
        pipeline_cfg=pipeline_cfg,
        global_cache=global_cache,
    )
    hcfg.network.timeout_sec = hgnc_settings.timeout_sec
    hcfg.network.max_retries = hgnc_settings.max_retries
    hcfg.rate_limit.rps = hgnc_settings.rps
    hcfg.cache = hgnc_settings.cache
    hgnc = HGNCClient(hcfg)

    gtop_cfg = _ensure_mapping(data.get("gtop"), context="gtop", allow_none=False)
    gtop_settings = _resolve_section_settings(
        "gtop",
        gtop_cfg,
        pipeline_cfg=pipeline_cfg,
        global_cache=global_cache,
    )
    gcfg = GtoPConfig(
        base_url=str(gtop_cfg["base_url"]),
        timeout_sec=gtop_settings.timeout_sec,
        max_retries=gtop_settings.max_retries,
        rps=gtop_settings.rps,
        backoff=gtop_settings.backoff_sec,
        cache=gtop_settings.cache,
    )
    gtop = GtoPClient(gcfg)

    ens_client: EnsemblHomologyClient | None = None
    oma_client: OmaClient | None = None
    target_species: list[str] = []
    if with_orthologs:
        orth_cfg = _ensure_mapping(data.get("orthologs"), context="orthologs")
        orth_settings = _resolve_section_settings(
            "orthologs",
            orth_cfg,
            pipeline_cfg=pipeline_cfg,
            global_cache=global_cache,
        )
        orth_cache = orth_settings.cache
        ens_client = EnsemblHomologyClient(
            base_url="https://rest.ensembl.org",
            network=UniNetworkConfig(
                timeout_sec=orth_settings.timeout_sec,
                max_retries=orth_settings.max_retries,
                backoff_sec=orth_settings.backoff_sec,
            ),
            rate_limit=UniRateConfig(rps=orth_settings.rps),
            cache=orth_cache,
        )
        oma_client = OmaClient(
            base_url="https://omabrowser.org/api",
            network=UniNetworkConfig(
                timeout_sec=orth_settings.timeout_sec,
                max_retries=orth_settings.max_retries,
                backoff_sec=orth_settings.backoff_sec,
            ),
            rate_limit=UniRateConfig(rps=orth_settings.rps),
            cache=orth_cache,
        )
        target_species = _ensure_str_sequence(
            orth_cfg.get("target_species"), context="orthologs.target_species"
        )
    return uni, hgnc, gtop, ens_client, oma_client, target_species


def main() -> None:
    """Main entry point for the unified target data pipeline."""

    args = parse_args()
    configure_logging(args.log_level, log_format=args.log_format)
    pipeline_cfg = load_pipeline_config(args.config)
    if args.list_format is not None:
        pipeline_cfg.list_format = args.list_format
    cli_species = _parse_species_argument(args.species)
    if cli_species:
        pipeline_cfg.species_priority = _merge_species_lists(
            cli_species, pipeline_cfg.species_priority
        )
    pipeline_cfg.iuphar.affinity_parameter = args.affinity_parameter
    if args.approved_only is not None:
        approved_only = args.approved_only.lower()
        pipeline_cfg.iuphar.approved_only = (
            None if approved_only == "null" else approved_only == "true"
        )
    if args.primary_target_only is not None:
        pipeline_cfg.iuphar.primary_target_only = (
            args.primary_target_only.lower() == "true"
        )
    pipeline_cfg.include_isoforms = pipeline_cfg.include_isoforms or args.with_isoforms
    use_isoforms = pipeline_cfg.include_isoforms

    # Load optional ChEMBL column configuration and ensure required fields
    data = _load_yaml_mapping(args.config)
    use_orthologs = bool(args.with_orthologs)
    global_cache = CacheConfig.from_dict(
        _optional_mapping(data, "http_cache", context="http_cache")
    )
    enrich_cfg = _ensure_mapping(data.get("uniprot_enrich"), context="uniprot_enrich")
    enrich_cache = (
        CacheConfig.from_dict(
            _optional_mapping(enrich_cfg, "cache", context="uniprot_enrich.cache")
        )
        or global_cache
    )
    chembl_section = _ensure_mapping(data.get("chembl"), context="chembl")
    chembl_cols = _ensure_str_sequence(
        chembl_section.get("columns"), context="chembl.columns"
    )
    required_cols = [
        "target_chembl_id",
        "pref_name",
        "protein_name_canonical",
        "target_type",
        "organism",
        "tax_id",
        "species_group_flag",
        "target_components",
        "protein_classifications",
        "cross_references",
        "gene_symbol_list",
        "protein_synonym_list",
        "hgnc_name",
        "hgnc_id",
    ]
    columns = chembl_cols or []
    for col in required_cols:
        if col not in columns:
            columns.append(col)
    chembl_cache = CacheConfig.from_dict(
        _optional_mapping(chembl_section, "cache", context="chembl.cache")
    )
    chembl_cfg: TargetConfig = TargetConfig(
        list_format=pipeline_cfg.list_format,
        columns=columns,
        cache=chembl_cache or global_cache,
    )

    # Merge additional column requirements from other data sources so that the
    # final output includes every requested field. This allows individual
    # sections in the YAML configuration (``uniprot``, ``gtop``, ``hgnc``) to
    # declare their own column lists without having to manually duplicate them
    # under ``pipeline.columns``.
    for section in ("uniprot", "gtop", "hgnc"):
        section_cfg = _ensure_mapping(data.get(section), context=section)
        for col in _ensure_str_sequence(
            section_cfg.get("columns"), context=f"{section}.columns"
        ):
            if col not in pipeline_cfg.columns:
                pipeline_cfg.columns.append(col)

    (
        uni_client,
        hgnc_client,
        gtop_client,
        ens_client,
        oma_client,
        target_species,
    ) = build_clients(
        args.config,
        pipeline_cfg,
        with_orthologs=use_orthologs,
        default_cache=global_cache,
    )

    df = pd.read_csv(args.input, sep=args.sep, encoding=args.encoding)
    if args.id_column not in df.columns:
        raise ValueError(f"Missing required column '{args.id_column}'")
    ids_series = df[args.id_column].astype(str).map(str.strip)
    ids_series = ids_series[ids_series != ""]
    ids: List[str] = list(dict.fromkeys(ids_series))

    # Fetch comprehensive ChEMBL data once and reuse it in the pipeline
    chembl_df = fetch_targets(ids, chembl_cfg, batch_size=args.batch_size)

    def _cached_chembl_fetch(
        _: Sequence[str], __: TargetConfig
    ) -> pd.DataFrame:  # pragma: no cover - simple wrapper
        return chembl_df

    # Run the pipeline with a progress bar to provide user feedback on long
    # operations. The progress bar advances once per processed target.
    with tqdm(total=len(ids), desc="targets") as pbar:
        out_df = run_pipeline(
            ids,
            pipeline_cfg,
            chembl_fetcher=_cached_chembl_fetch,
            chembl_config=chembl_cfg,
            uniprot_client=uni_client,
            hgnc_client=hgnc_client,
            gtop_client=gtop_client,
            ensembl_client=ens_client,
            oma_client=oma_client,
            target_species=target_species,
            progress_callback=pbar.update,
        )
    enrich_client = UniProtEnrichClient(cache_config=enrich_cache)
    out_df = add_uniprot_fields(out_df, enrich_client.fetch_all)
    out_df = merge_chembl_fields(out_df, chembl_df)
    entry_cache: Dict[str, Any] = {}

    def cached_fetch(acc: str) -> Any:
        if acc not in entry_cache:
            entry_cache[acc] = uni_client.fetch_entry_json(acc)
        return entry_cache[acc]

    out_df = add_activity_fields(out_df, cached_fetch)
    if use_isoforms:
        out_df = add_isoform_fields(out_df, cached_fetch)

    # Append optional IUPHAR classification data when both CSV files are provided.
    if args.iuphar_target and args.iuphar_family:
        target_csv = Path(args.iuphar_target)
        family_csv = Path(args.iuphar_family)
        if not target_csv.exists():
            msg = f"IUPHAR target file not found: {target_csv}"
            raise FileNotFoundError(msg)
        if not family_csv.exists():
            msg = f"IUPHAR family file not found: {family_csv}"
            raise FileNotFoundError(msg)
        out_df = add_iuphar_classification(
            out_df,
            target_csv,
            family_csv,
            encoding=args.encoding,
        )
    out_df = add_protein_classification(
        out_df,
        lambda accs: uni_client.fetch_entries_json(accs, batch_size=args.batch_size),
    )
    # Keep classification columns grouped together at the end for clarity.
    cols = [c for c in out_df.columns if c not in IUPHAR_CLASS_COLUMNS]
    out_df = out_df[cols + IUPHAR_CLASS_COLUMNS]

    sort_candidates = [
        "target_chembl_id",
        "uniprot_id_primary",
        "gene_symbol",
        "hgnc_id",
    ]
    sort_columns = [column for column in sort_candidates if column in out_df.columns]
    if sort_columns:
        out_df = out_df.sort_values(sort_columns).reset_index(drop=True)

    output_path = ensure_output_dir(Path(args.output).expanduser().resolve())
    serialised_df = serialise_dataframe(out_df, list_format=pipeline_cfg.list_format)
    serialised_df.to_csv(
        output_path,
        index=False,
        sep=args.sep,
        encoding=args.encoding,
    )

    analyze_table_quality(serialised_df, table_name=str(output_path.with_suffix("")))

    meta_path = Path(args.meta_output) if args.meta_output else None
    write_cli_metadata(
        output_path,
        row_count=int(serialised_df.shape[0]),
        column_count=int(serialised_df.shape[1]),
        namespace=args,
        command_parts=tuple(sys.argv),
        meta_path=meta_path,
    )


if __name__ == "__main__":
    main()
