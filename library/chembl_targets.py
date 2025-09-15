"""Utilities for downloading and normalising ChEMBL target records.

The :func:`fetch_targets` function orchestrates the retrieval of target
information for a list of ChEMBL identifiers and returns a :class:`pandas.DataFrame`
ready for serialisation.  The implementation favours determinism: list-like
fields are sorted and serialised either as JSON arrays or pipe-delimited strings
depending on the configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, Dict, List, Sequence

import pandas as pd

from http_client import HttpClient

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass


@dataclass
class TargetConfig:
    """Configuration controlling data acquisition."""

    base_url: str = "https://www.ebi.ac.uk/chembl/api/data"
    timeout_sec: float = 30.0
    max_retries: int = 3
    rps: float = 2.0
    output_encoding: str = "utf-8-sig"
    output_sep: str = ","
    list_format: str = "json"  # "json" or "pipe"


# ---------------------------------------------------------------------------
# Normalisation helpers


def normalise_ids(ids: Sequence[str]) -> List[str]:
    """Return cleaned and deduplicated ChEMBL identifiers.

    Parameters
    ----------
    ids:
        Raw sequence of identifiers possibly containing duplicates or empty
        entries.
    """

    cleaned = []
    seen = set()
    for raw in ids:
        if raw is None:
            continue
        cid = raw.strip().upper()
        if not cid:
            continue
        if cid not in seen:
            seen.add(cid)
            cleaned.append(cid)
    return cleaned


# ---------------------------------------------------------------------------
# Data extraction utilities


def _serialize(obj: Any, *, list_format: str) -> str:
    """Serialise ``obj`` deterministically according to ``list_format``."""

    if isinstance(obj, list) and list_format == "pipe":
        return "|".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in obj)
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _extract_components(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    comps = []
    for comp in payload.get("target_components", []) or []:
        comps.append(
            {
                "component_id": comp.get("component_id"),
                "accession": comp.get("accession"),
                "component_type": comp.get("component_type"),
                "component_description": comp.get("component_description"),
            }
        )
    comps.sort(key=lambda c: (c.get("accession") or ""))
    return comps


def _extract_cross_refs(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    for ref in payload.get("cross_references", []) or []:
        refs.append({"xref_db": ref.get("xref_db"), "xref_id": ref.get("xref_id")})
    for comp in payload.get("target_components", []) or []:
        for ref in comp.get("target_component_xrefs", []) or []:
            refs.append(
                {"xref_db": ref.get("xref_src_db"), "xref_id": ref.get("xref_id")}
            )
    refs.sort(key=lambda r: (r.get("xref_db") or "", r.get("xref_id") or ""))
    return refs


def _extract_gene_symbols(payload: Dict[str, Any]) -> List[str]:
    genes: set[str] = set()
    for comp in payload.get("target_components", []) or []:
        for syn in comp.get("target_component_synonyms", []) or []:
            stype = (syn.get("syn_type") or "").upper()
            if "GENE_SYMBOL" in stype:
                genes.add(syn.get("component_synonym", ""))
    return sorted(g for g in genes if g)


def _extract_protein_synonyms(payload: Dict[str, Any]) -> List[str]:
    """Collect protein name synonyms from the payload.

    The ChEMBL API nests protein synonyms under each entry in the
    ``target_components`` list.  Synonyms are tagged with a ``syn_type``; we
    retain those where the type indicates a protein or target name.

    Parameters
    ----------
    payload:
        The JSON dictionary returned by the ChEMBL API.

    Returns
    -------
    list[str]
        Sorted list of unique synonym strings.
    """

    names: set[str] = set()
    for comp in payload.get("target_components", []) or []:
        for syn in comp.get("target_component_synonyms", []) or []:
            stype = (syn.get("syn_type") or "").upper()
            if "PROTEIN" in stype or "TARGET" in stype:
                names.add(syn.get("component_synonym", ""))
    return sorted(n for n in names if n)


def _extract_protein_classifications(payload: Dict[str, Any]) -> List[str]:
    classifications: List[str] = []
    pc = payload.get("protein_classification")
    while pc:
        name = pc.get("pref_name") or pc.get("protein_classification")
        if name:
            classifications.insert(0, name)
        pc = pc.get("parent") if isinstance(pc.get("parent"), dict) else None
    return classifications


# ---------------------------------------------------------------------------
# Public API


def fetch_targets(ids: Sequence[str], cfg: TargetConfig) -> pd.DataFrame:
    """Fetch ChEMBL targets and return a normalised DataFrame.

    Parameters
    ----------
    ids:
        Sequence of target ChEMBL identifiers.
    cfg:
        Configuration governing network behaviour and output formatting.
    """

    norm_ids = normalise_ids(ids)
    client = HttpClient(
        timeout=cfg.timeout_sec, max_retries=cfg.max_retries, rps=cfg.rps
    )
    records: List[Dict[str, Any]] = []
    for chembl_id in norm_ids:
        url = f"{cfg.base_url.rstrip('/')}/target/{chembl_id}?format=json"
        try:
            resp = client.request("get", url)
            if resp.status_code == 200:
                payload = resp.json()
            else:
                LOGGER.warning(
                    "Non-200 response for %s: %s", chembl_id, resp.status_code
                )
                payload = {}
        except Exception as exc:  # noqa: BLE001 - we log and continue
            LOGGER.warning("Failed to fetch %s: %s", chembl_id, exc)
            payload = {}
        record = {
            "target_chembl_id": chembl_id,
            "pref_name": payload.get("pref_name"),
            "protein_name_canonical": payload.get("pref_name"),
            "target_type": payload.get("target_type"),
            "organism": payload.get("organism"),
            "tax_id": payload.get("tax_id"),
            "species_group_flag": payload.get("species_group_flag"),
            "target_components": _serialize(
                _extract_components(payload), list_format=cfg.list_format
            ),
            "protein_classifications": _serialize(
                _extract_protein_classifications(payload), list_format=cfg.list_format
            ),
            "cross_references": _serialize(
                _extract_cross_refs(payload), list_format=cfg.list_format
            ),
            "gene_symbol_list": _serialize(
                _extract_gene_symbols(payload), list_format=cfg.list_format
            ),
            "protein_synonym_list": _serialize(
                _extract_protein_synonyms(payload), list_format=cfg.list_format
            ),
        }
        records.append(record)
    df = pd.DataFrame(records)
    return df


__all__ = [
    "TargetConfig",
    "fetch_targets",
    "normalise_ids",
]
