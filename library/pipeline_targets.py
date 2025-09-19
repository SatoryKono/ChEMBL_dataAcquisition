from __future__ import annotations

import json
import logging
from math import isnan
from datetime import UTC, datetime
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Sequence

import pandas as pd
from typing import TYPE_CHECKING
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

if TYPE_CHECKING:  # pragma: no cover - for static type checking only
    from .chembl_targets import TargetConfig, fetch_targets
    from .gtop_client import GtoPClient, resolve_target
    from .gtop_normalize import normalise_interactions, normalise_synonyms
    from .hgnc_client import HGNCClient, HGNCRecord
    from .orthologs import EnsemblHomologyClient, OmaClient
    from .uniprot_client import UniProtClient
    from .uniprot_normalize import (
        extract_ensembl_gene_ids,
        extract_isoforms,
        normalize_entry,
    )
else:  # pragma: no cover - support both package and direct imports
    try:
        from .chembl_targets import TargetConfig, fetch_targets
        from .gtop_client import GtoPClient, resolve_target
        from .gtop_normalize import normalise_interactions, normalise_synonyms
        from .hgnc_client import HGNCClient, HGNCRecord
        from .orthologs import EnsemblHomologyClient, OmaClient
        from .uniprot_client import UniProtClient
        from .uniprot_normalize import (
            extract_ensembl_gene_ids,
            extract_isoforms,
            normalize_entry,
        )
    except ImportError:
        from chembl_targets import TargetConfig, fetch_targets
        from gtop_client import GtoPClient, resolve_target
        from gtop_normalize import normalise_interactions, normalise_synonyms
        from hgnc_client import HGNCClient, HGNCRecord
        from orthologs import EnsemblHomologyClient, OmaClient
        from uniprot_client import UniProtClient
        from uniprot_normalize import (
            extract_ensembl_gene_ids,
            extract_isoforms,
            normalize_entry,
        )

try:  # pragma: no cover - import fallback mirrors the logic above
    from .chembl2uniprot.config import _apply_env_overrides as apply_env_overrides
except ImportError:  # pragma: no cover - script execution without package context
    from chembl2uniprot.config import (  # type: ignore[import-not-found]
        _apply_env_overrides as apply_env_overrides,
    )

LOGGER = logging.getLogger(__name__)

PIPELINE_VERSION = "0.1.0"

# Default columns for final output files
DEFAULT_COLUMNS = [
    "target_chembl_id",
    "uniprot_id_primary",
    "uniprot_ids_all",
    "isoform_ids",
    "isoform_names",
    "isoform_synonyms",
    "hgnc_id",
    "gene_symbol",
    "protein_name_canonical",
    "protein_name_alt",
    "names_all",
    "synonyms_all",
    "organism",
    "taxon_id",
    "lineage_superkingdom",
    "lineage_phylum",
    "lineage_class",
    "lineage_order",
    "lineage_family",
    "target_type",
    "protein_class_L1",
    "protein_class_L2",
    "protein_class_L3",
    "protein_class_L4",
    "protein_class_L5",
    "sequence_length",
    "features_signal_peptide",
    "features_transmembrane",
    "features_topology",
    "ptm_glycosylation",
    "ptm_lipidation",
    "ptm_disulfide_bond",
    "ptm_modified_residue",
    "pfam",
    "interpro",
    "xref_chembl",
    "xref_uniprot",
    "xref_ensembl",
    "xref_pdb",
    "xref_alphafold",
    "xref_iuphar",
    "gtop_target_id",
    "gtop_synonyms",
    "ortholog_uniprot_ids",
    "orthologs_json",
    "orthologs_count",
    "gtop_natural_ligands_n",
    "gtop_interactions_n",
    "gtop_function_text_short",
    "uniprot_last_update",
    "uniprot_version",
    "pipeline_version",
    "timestamp_utc",
]

_INVALID_IDENTIFIER_LITERALS = frozenset({"", "nan", "none", "null", "na", "n/a"})


_DEFAULT_SPECIES_PRIORITY = ("Human", "Homo sapiens")


def _default_timestamp() -> str:
    """Return an ISO 8601 timestamp in UTC for configuration defaults."""

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _coerce_str_sequence(
    value: Any, *, default: Sequence[str], context: str
) -> List[str]:
    """Return ``value`` as a list of strings honouring configuration defaults."""

    if value is None:
        return list(default)
    candidate = value
    if isinstance(candidate, str):
        text = candidate.strip()
        if not text:
            return []
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        candidate = decoded
    if isinstance(candidate, SequenceABC) and not isinstance(
        candidate, (bytes, bytearray, str)
    ):
        result: List[str] = []
        for item in candidate:
            if not isinstance(item, str):
                msg = (
                    f"Expected entries of '{context}' to be strings, "
                    f"found {type(item).__name__!s}"
                )
                raise TypeError(msg)
            result.append(item)
        return result
    if isinstance(candidate, str):  # pragma: no cover - handled above defensively
        return [candidate]
    msg = f"Expected '{context}' to be a sequence of strings"
    raise TypeError(msg)


class IupharConfig(BaseModel):
    """Settings for querying the IUPHAR/GtoP service."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    affinity_parameter: str = "pKi"
    approved_only: bool | None = None
    primary_target_only: bool = True

    @field_validator("approved_only", mode="before")
    @classmethod
    def _normalise_optional_bool(cls, value: Any) -> bool | None:
        """Normalise textual representations of optional boolean fields."""

        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"null", "none", ""}:
                return None
        return value


class PipelineConfig(BaseModel):
    """High level pipeline configuration.

    Attributes
    ----------
    rate_limit_rps:
        Requests per second for external services.
    retries:
        Maximum number of network retries.
    timeout_sec:
        Timeout in seconds for network operations.
    species_priority:
        Ordered list of species names used to select the primary UniProt
        record.
    list_format:
        Serialisation format for list-like fields (``"json"`` or ``"pipe"``).
    include_sequence:
        Whether to include protein sequences in the output.
    columns:
        Ordered list of columns written to the final output file.
    iuphar:
        Configuration for the IUPHAR/GtoP client.
    timestamp_utc:
        ISO 8601 timestamp recorded for the pipeline run. Defaults to the
        creation time of the configuration instance so repeated executions
        with the same configuration remain reproducible.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    rate_limit_rps: float = Field(default=2.0, gt=0)
    retries: int = Field(default=3, ge=0)
    timeout_sec: float = Field(default=30.0, gt=0)
    species_priority: List[str] = Field(
        default_factory=lambda: list(_DEFAULT_SPECIES_PRIORITY)
    )
    list_format: str = "json"
    include_sequence: bool = False
    include_isoforms: bool = False
    columns: List[str] = Field(default_factory=lambda: list(DEFAULT_COLUMNS))
    iuphar: IupharConfig = Field(default_factory=IupharConfig)
    timestamp_utc: str = Field(default_factory=_default_timestamp)

    @field_validator("iuphar", mode="before")
    @classmethod
    def _default_iuphar(cls, value: Any) -> dict[str, Any] | IupharConfig:
        """Allow ``null`` IUPHAR configuration entries in YAML files."""

        if value is None:
            return {}
        return value

    @field_validator("species_priority", mode="before")
    @classmethod
    def _validate_species_priority(cls, value: Any) -> List[str]:
        """Coerce the species priority field into a list of strings."""

        return _coerce_str_sequence(
            value,
            default=_DEFAULT_SPECIES_PRIORITY,
            context="pipeline.species_priority",
        )

    @field_validator("columns", mode="before")
    @classmethod
    def _validate_columns(cls, value: Any) -> List[str]:
        """Coerce the column list into a list of strings."""

        return _coerce_str_sequence(
            value, default=DEFAULT_COLUMNS, context="pipeline.columns"
        )

    @field_validator("list_format")
    @classmethod
    def _validate_list_format(cls, value: str) -> str:
        """Ensure the list serialisation format is supported."""

        cleaned = value.strip().lower()
        if cleaned not in {"json", "pipe"}:
            msg = "list_format must be either 'json' or 'pipe'"
            raise ValueError(msg)
        return cleaned

    @field_validator("timestamp_utc", mode="before")
    @classmethod
    def _normalise_timestamp(cls, value: Any) -> str:
        """Normalise timestamps to RFC 3339 compliant strings."""

        if value is None:
            return _default_timestamp()
        if isinstance(value, datetime):
            instant = value if value.tzinfo else value.replace(tzinfo=UTC)
            return instant.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        if isinstance(value, str):
            return value
        msg = "timestamp_utc must be an ISO 8601 string"
        raise TypeError(msg)


# ---------------------------------------------------------------------------
# Configuration helpers


def _ensure_mapping(
    value: Any, *, context: str, allow_none: bool = True
) -> dict[str, Any]:
    """Return ``value`` as a dictionary ensuring it is a mapping."""

    if value is None:
        if allow_none:
            return {}
        msg = f"Expected '{context}' to be a mapping, not null"
        raise TypeError(msg)
    if isinstance(value, MappingABC):
        return dict(value)
    msg = f"Expected '{context}' to be a mapping, not {type(value).__name__!s}"
    raise TypeError(msg)


def load_pipeline_config(path: str) -> PipelineConfig:
    """Load :class:`PipelineConfig` from ``path`` applying environment overrides."""

    import yaml  # type: ignore[import]

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    config_root = _ensure_mapping(
        data, context="pipeline configuration", allow_none=False
    )
    pipeline_section = _ensure_mapping(config_root.get("pipeline"), context="pipeline")
    apply_env_overrides(pipeline_section, section="pipeline")
    try:
        return PipelineConfig.model_validate(pipeline_section)
    except ValidationError as exc:  # pragma: no cover - exercised in tests
        raise ValueError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Utility functions


def _serialise_list(items: Sequence[Any], list_format: str) -> str:
    """Serialize a sequence of items into a string.

    The items are first cleaned (None and empty strings removed), deduplicated,
    and sorted. Then, they are serialized according to the specified format.

    Parameters
    ----------
    items:
        The sequence of items to serialize.
    list_format:
        The format to use for serialization ("pipe" or "json").

    Returns
    -------
    str
        The serialized string.
    """
    cleaned = [x for x in items if x not in (None, "")]
    cleaned = sorted(dict.fromkeys(cleaned))
    if list_format == "pipe":
        return "|".join(str(x) for x in cleaned)
    return json.dumps(cleaned, ensure_ascii=False, sort_keys=True)


def _split_pipe(serialised: str) -> List[str]:
    """Split a pipe-delimited string handling escaped separators."""

    parts: List[str] = []
    current: List[str] = []
    escape = False
    for char in serialised:
        if escape:
            if char not in ("|", "\\"):
                current.append("\\")
            current.append(char)
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == "|":
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    if escape:
        current.append("\\")
    parts.append("".join(current))
    return parts


def _load_serialised_list(serialised: Any | None, list_format: str) -> List[Any]:
    """Deserialize list-like data encoded as JSON or pipe-separated text.

    Parameters
    ----------
    serialised:
        Input value produced by :func:`_serialise_list` or the ChEMBL
        target serializer. ``None`` and empty strings are treated as empty
        lists. ``NaN`` values from pandas inputs are also interpreted as
        missing data.
    list_format:
        Encoding of the data. Supported values are ``"json"`` and
        ``"pipe"``.

    Returns
    -------
    List[Any]
        Decoded list of items. Items encoded as JSON primitives retain their
        original types. When pipe encoding is used and items are not valid
        JSON fragments they are returned as plain strings.

    Raises
    ------
    ValueError
        If ``list_format`` is unknown or the JSON payload cannot be
        decoded into a list.
    """

    if serialised is None:
        return []
    if isinstance(serialised, float) and isnan(serialised):
        return []
    if isinstance(serialised, list):
        return list(serialised)
    if isinstance(serialised, tuple):
        return list(serialised)
    if isinstance(serialised, str):
        text = serialised.strip()
    else:
        text = str(serialised).strip()
    if not text:
        return []

    if list_format == "json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON list serialisation") from exc
        if data is None:
            return []
        if isinstance(data, list):
            return data
        raise ValueError("JSON payload does not represent a list")
    if list_format == "pipe":
        parts = _split_pipe(text)
        items: List[Any] = []
        for part in parts:
            chunk = part.strip()
            if not chunk:
                continue
            try:
                items.append(json.loads(chunk))
            except json.JSONDecodeError:
                items.append(chunk)
        return items
    raise ValueError(f"Unsupported list_format: {list_format}")


def _normalise_identifier_value(value: Any) -> str:
    """Return a cleaned identifier string or ``""`` when invalid."""

    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in _INVALID_IDENTIFIER_LITERALS:
        return ""
    return text


def _clean_identifier_sequence(values: Sequence[Any]) -> list[str]:
    """Return unique, order-preserving identifiers from ``values``."""

    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in values:
        identifier = _normalise_identifier_value(raw)
        if not identifier or identifier in seen:
            continue
        seen.add(identifier)
        cleaned.append(identifier)
    return cleaned


def _has_isoform_annotation(entry: Mapping[str, Any]) -> bool:
    """Return ``True`` if ``entry`` describes alternative isoform products."""

    comments = entry.get("comments")
    if not isinstance(comments, SequenceABC):
        return False
    for comment in comments:
        if not isinstance(comment, MappingABC):
            continue
        if comment.get("commentType") != "ALTERNATIVE_PRODUCTS":
            continue
        isoforms = comment.get("isoforms")
        if isinstance(isoforms, SequenceABC) and isoforms:
            return True
    return False


def _select_primary(
    entries: List[Dict[str, Any]], priority: Sequence[str]
) -> Dict[str, Any] | None:
    """Select the primary UniProt entry from a list of entries.

    The selection is based on a priority list of species. If no entry matches
    the priority list, the first entry in the sorted list is returned.

    Parameters
    ----------
    entries:
        A list of normalized UniProt entry dictionaries.
    priority:
        A sequence of species names in order of priority.

    Returns
    -------
    Dict[str, Any] | None
        The selected primary entry, or None if the input list is empty.
    """
    if not entries:
        return None
    for sp in priority:
        for entry in entries:
            if entry.get("organism_name") == sp:
                return entry
    return sorted(entries, key=lambda e: e.get("uniprot_id") or "")[0]


# ---------------------------------------------------------------------------
# Core orchestration


def run_pipeline(
    ids: Sequence[str],
    cfg: PipelineConfig,
    *,
    chembl_fetcher=fetch_targets,
    chembl_config: TargetConfig | None = None,
    uniprot_client: UniProtClient,
    hgnc_client: HGNCClient | None = None,
    gtop_client: GtoPClient | None = None,
    ensembl_client: EnsemblHomologyClient | None = None,
    oma_client: OmaClient | None = None,
    target_species: Sequence[str] | None = None,
    progress_callback: Callable[[int], None] | None = None,
    entry_cache: MutableMapping[str, Any] | None = None,
    batch_size: int = 25,
) -> pd.DataFrame:
    """Orchestrates data acquisition for a sequence of ChEMBL target identifiers.

    Args:
        ids: A sequence of target ChEMBL identifiers.
        cfg: The pipeline configuration.
        chembl_fetcher: The function used to download ChEMBL target information.
        chembl_config: The configuration passed to the `chembl_fetcher`.
        uniprot_client: The client instance for the UniProt service.
        hgnc_client: An optional client for HGNC lookups.
        gtop_client: An optional client for IUPHAR/GtoP data.
        ensembl_client: An optional client for Ensembl ortholog retrieval.
        oma_client: An optional client for OMA ortholog lookups.
        target_species: A list of species to consider when retrieving orthologs.
        progress_callback: An optional callback that receives the incremental count
            of processed records.
        entry_cache: Optional mapping storing previously fetched UniProt entries.
            When supplied, the cache is updated with records retrieved during the
            pipeline execution so that downstream enrichment steps can reuse the
            data without additional network requests.
        batch_size: Maximum number of UniProt accessions retrieved per batch
            request. The value must be greater than zero.

    Returns:
        A pandas DataFrame containing the orchestrated data.
    """

    if batch_size <= 0:
        msg = "batch_size must be a positive integer"
        raise ValueError(msg)

    chembl_cfg = chembl_config or TargetConfig(list_format=cfg.list_format)
    cleaned_ids = _clean_identifier_sequence(ids)
    if not cleaned_ids:
        chembl_df = pd.DataFrame(columns=chembl_cfg.columns)
    else:
        chembl_df = chembl_fetcher(cleaned_ids, chembl_cfg)
    records: List[Dict[str, Any]] = []
    fetch_full_entry = bool(cfg.include_isoforms or ensembl_client or oma_client)
    for row in chembl_df.to_dict(orient="records"):
        chembl_id = row.get("target_chembl_id", "")
        comps = _load_serialised_list(row.get("target_components"), cfg.list_format)
        accessions = [c.get("accession") for c in comps if c.get("accession")]
        uniprot_entries: List[Dict[str, Any]] = []
        raw_entries: Dict[str, Dict[str, Any]] = {}
        unique_accessions = [
            accession for accession in sorted(dict.fromkeys(accessions)) if accession
        ]
        if fetch_full_entry and unique_accessions:
            pending_fetch: List[str] = []
            for acc in unique_accessions:
                cached_entry: Any = None
                if entry_cache is not None and acc in entry_cache:
                    cached_entry = entry_cache[acc]
                if cached_entry:
                    raw_entries[acc] = cached_entry
                else:
                    pending_fetch.append(acc)
            if pending_fetch:
                batch_records = uniprot_client.fetch_entries_json(
                    pending_fetch, batch_size=batch_size
                )
                for acc, raw in batch_records.items():
                    if raw:
                        raw_entries[acc] = raw
                        if entry_cache is not None:
                            entry_cache[acc] = raw
                remaining = [acc for acc in pending_fetch if acc not in raw_entries]
                for acc in remaining:
                    raw = uniprot_client.fetch_entry_json(acc)
                    if raw:
                        raw_entries[acc] = raw
                        if entry_cache is not None:
                            entry_cache[acc] = raw

        for acc in unique_accessions:
            if fetch_full_entry:
                raw = raw_entries.get(acc)
                if not raw:
                    continue
                isoforms: List[Dict[str, Any]] = []
                if cfg.include_isoforms and _has_isoform_annotation(raw):
                    fasta_headers = uniprot_client.fetch_isoforms_fasta(acc)
                    isoforms = extract_isoforms(raw, fasta_headers)
                elif cfg.include_isoforms:
                    isoforms = []
                uniprot_entries.append(
                    normalize_entry(
                        raw,
                        include_sequence=cfg.include_sequence,
                        isoforms=isoforms,
                    )
                )
            else:
                cached_basic_entry: Any = None
                cache_has_key = False
                if entry_cache is not None and acc in entry_cache:
                    cache_has_key = True
                    cached_basic_entry = entry_cache[acc]
                raw = cached_basic_entry
                if raw is None and not cache_has_key:
                    raw = uniprot_client.fetch(acc)
                if raw:
                    uniprot_entries.append(
                        normalize_entry(raw, include_sequence=cfg.include_sequence)
                    )
        primary = _select_primary(uniprot_entries, cfg.species_priority)
        primary_id = primary.get("uniprot_id") if primary else ""
        all_ids = [e.get("uniprot_id") for e in uniprot_entries]

        hgnc_id = ""
        gene_symbol = primary.get("gene_primary", "") if primary else ""
        hgnc_rec: HGNCRecord | None = None
        if (
            primary
            and hgnc_client
            and primary.get("organism_name") in cfg.species_priority
        ):
            hgnc_rec = hgnc_client.fetch(primary_id)
            if hgnc_rec:
                hgnc_id = hgnc_rec.hgnc_id
                if hgnc_rec.gene_symbol:
                    gene_symbol = hgnc_rec.gene_symbol

        gtop_id = ""
        gtop_synonyms: List[str] = []
        gtop_nat = 0
        gtop_int = 0
        gtop_func = ""
        gtop_name = ""
        if gtop_client:
            target = None
            for id_col, value in [
                ("uniprot_id", primary_id),
                ("hgnc_id", hgnc_id),
                ("gene_symbol", gene_symbol),
                ("target_name", row.get("pref_name", "")),
            ]:
                if value:
                    target = resolve_target(gtop_client, value, id_col)
                    if target:
                        break
            if target:
                gtop_id = str(target.get("targetId", ""))
                gtop_name = target.get("name", "") or ""
                syn_df = normalise_synonyms(
                    int(target.get("targetId", 0)),
                    gtop_client.fetch_target_endpoint(
                        target.get("targetId"), "synonyms"
                    ),
                )
                gtop_synonyms = syn_df["synonym"].tolist() if not syn_df.empty else []
                nat = gtop_client.fetch_target_endpoint(
                    target.get("targetId"), "naturalLigands"
                )
                gtop_nat = len(nat or [])
                params: Dict[str, Any] = {}
                if cfg.iuphar.approved_only:
                    params["approved"] = "true"
                if cfg.iuphar.primary_target_only:
                    params["primaryTarget"] = "true"
                inter = gtop_client.fetch_target_endpoint(
                    target.get("targetId"), "interactions", params
                )
                inter_df = normalise_interactions(int(target.get("targetId", 0)), inter)
                gtop_int = len(inter_df)
                func = gtop_client.fetch_target_endpoint(
                    target.get("targetId"), "function"
                )
                if func:
                    first = func[0]
                    gtop_func = (first.get("functionText") or "")[:200]

        isoform_ids = primary.get("isoform_ids_all", []) if primary else []
        isoform_names = primary.get("isoform_names", []) if primary else []
        isoform_synonyms: List[str] = []
        if cfg.include_isoforms and primary:
            try:
                iso_data = _load_serialised_list(primary.get("isoforms_json"), "json")
            except ValueError:
                iso_data = []
            for iso in iso_data:
                isoform_synonyms.extend(iso.get("isoform_synonyms", []))

        orthologs_json = "[]"
        orthologs_count = 0
        ortholog_ids: List[str] = []
        if (ensembl_client or oma_client) and primary_id:
            primary_raw = raw_entries.get(primary_id, {})
            gene_ids = extract_ensembl_gene_ids(primary_raw)
            gene_id = gene_ids[0] if gene_ids else ""
            orthologs = []
            if ensembl_client and gene_id:
                orthologs = ensembl_client.get_orthologs(
                    gene_id, list(target_species or [])
                )
            if not orthologs and oma_client:
                orthologs = oma_client.get_orthologs_by_uniprot(primary_id)
            if orthologs:
                ortholog_ids = [
                    o.target_uniprot_id or "" for o in orthologs if o.target_uniprot_id
                ]
                orthologs_json = json.dumps(
                    [o.to_ordered_dict() for o in orthologs],
                    separators=(",", ":"),
                    sort_keys=True,
                )
                orthologs_count = len(orthologs)

        pc = _load_serialised_list(row.get("protein_classifications"), cfg.list_format)
        pc += [""] * (5 - len(pc))
        cross_refs = _load_serialised_list(row.get("cross_references"), cfg.list_format)
        chembl_refs = [
            r.get("xref_id") for r in cross_refs if r.get("xref_db") == "ChEMBL"
        ]

        names: set[str] = set()
        synonyms: set[str] = set()
        if row.get("pref_name"):
            names.add(row.get("pref_name", ""))
        syn_list = _load_serialised_list(
            row.get("protein_synonym_list"), cfg.list_format
        )
        synonyms.update(syn_list)
        gene_list = _load_serialised_list(row.get("gene_symbol_list"), cfg.list_format)
        synonyms.update(gene_list)
        if primary:
            if primary.get("protein_recommended_name"):
                names.add(primary.get("protein_recommended_name", ""))
            synonyms.update(primary.get("protein_alternative_names", []))
            if primary.get("gene_primary"):
                names.add(primary.get("gene_primary", ""))
            synonyms.update(primary.get("gene_synonyms", []))
            synonyms.update(primary.get("isoform_names", []))
        if hgnc_rec:
            if hgnc_rec.gene_symbol:
                names.add(hgnc_rec.gene_symbol)
            if hgnc_rec.gene_name:
                names.add(hgnc_rec.gene_name)
            if hgnc_rec.protein_name:
                names.add(hgnc_rec.protein_name)
        if gtop_name:
            names.add(gtop_name)
        synonyms.update(gtop_synonyms)
        synonyms.update(isoform_synonyms)

        rec: Dict[str, Any] = {
            "target_chembl_id": chembl_id,
            "uniprot_id_primary": primary_id,
            "uniprot_ids_all": _serialise_list(all_ids, cfg.list_format),
            "isoform_ids": _serialise_list(isoform_ids, "pipe"),
            "isoform_names": _serialise_list(isoform_names, "pipe"),
            "isoform_synonyms": _serialise_list(isoform_synonyms, "pipe"),
            "hgnc_id": hgnc_id,
            "gene_symbol": gene_symbol,
            "protein_name_canonical": (
                primary.get("protein_recommended_name")
                if primary
                else row.get("protein_name_canonical", row.get("pref_name", ""))
            ),
            "protein_name_alt": _serialise_list(
                (
                    primary.get("protein_alternative_names", [])
                    if primary
                    else _load_serialised_list(
                        row.get("protein_synonym_list"), cfg.list_format
                    )
                ),
                cfg.list_format,
            ),
            "names_all": _serialise_list(list(names), "pipe"),
            "synonyms_all": _serialise_list(list(synonyms), "pipe"),
            "organism": (
                primary.get("organism_name", row.get("organism", ""))
                if primary
                else row.get("organism", "")
            ),
            "taxon_id": (
                primary.get("taxon_id", row.get("tax_id", ""))
                if primary
                else row.get("tax_id", "")
            ),
            "lineage_superkingdom": (
                primary.get("lineage_superkingdom", "") if primary else ""
            ),
            "lineage_phylum": primary.get("lineage_phylum", "") if primary else "",
            "lineage_class": primary.get("lineage_class", "") if primary else "",
            "lineage_order": primary.get("lineage_order", "") if primary else "",
            "lineage_family": primary.get("lineage_family", "") if primary else "",
            "target_type": row.get("target_type", ""),
            "protein_class_L1": pc[0],
            "protein_class_L2": pc[1],
            "protein_class_L3": pc[2],
            "protein_class_L4": pc[3],
            "protein_class_L5": pc[4],
            "sequence_length": primary.get("sequence_length", "") if primary else "",
            "features_signal_peptide": _serialise_list(
                primary.get("features_signal_peptide", []) if primary else [],
                cfg.list_format,
            ),
            "features_transmembrane": _serialise_list(
                primary.get("features_transmembrane", []) if primary else [],
                cfg.list_format,
            ),
            "features_topology": _serialise_list(
                primary.get("features_topology", []) if primary else [],
                cfg.list_format,
            ),
            "ptm_glycosylation": _serialise_list(
                primary.get("ptm_glycosylation", []) if primary else [],
                cfg.list_format,
            ),
            "ptm_lipidation": _serialise_list(
                primary.get("ptm_lipidation", []) if primary else [],
                cfg.list_format,
            ),
            "ptm_disulfide_bond": _serialise_list(
                primary.get("ptm_disulfide_bond", []) if primary else [],
                cfg.list_format,
            ),
            "ptm_modified_residue": _serialise_list(
                primary.get("ptm_modified_residue", []) if primary else [],
                cfg.list_format,
            ),
            "pfam": _serialise_list(
                [p[0] for p in primary.get("domains_pfam", [])] if primary else [],
                cfg.list_format,
            ),
            "interpro": _serialise_list(
                [p[0] for p in primary.get("domains_interpro", [])] if primary else [],
                cfg.list_format,
            ),
            "xref_chembl": _serialise_list(chembl_refs, cfg.list_format),
            "xref_uniprot": primary_id,
            "xref_ensembl": _serialise_list(
                primary.get("xref_ensembl", []) if primary else [], cfg.list_format
            ),
            "xref_pdb": _serialise_list(
                primary.get("3d_pdb_ids", []) if primary else [], cfg.list_format
            ),
            "xref_alphafold": primary.get("alphafold_id", "") if primary else "",
            "xref_iuphar": gtop_id,
            "gtop_target_id": gtop_id,
            "gtop_synonyms": _serialise_list(gtop_synonyms, cfg.list_format),
            "ortholog_uniprot_ids": _serialise_list(ortholog_ids, "pipe"),
            "orthologs_json": orthologs_json,
            "orthologs_count": orthologs_count,
            "gtop_natural_ligands_n": gtop_nat,
            "gtop_interactions_n": gtop_int,
            "gtop_function_text_short": gtop_func,
            "uniprot_last_update": (
                primary.get("last_annotation_update", "") if primary else ""
            ),
            "uniprot_version": primary.get("entry_version", "") if primary else "",
            "pipeline_version": PIPELINE_VERSION,
            "timestamp_utc": cfg.timestamp_utc,
        }
        records.append(rec)
        if progress_callback:
            progress_callback(1)

    df = pd.DataFrame(records, columns=cfg.columns)
    return df


__all__ = ["PipelineConfig", "IupharConfig", "load_pipeline_config", "run_pipeline"]
