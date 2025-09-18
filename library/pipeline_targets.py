from __future__ import annotations

import json
import logging
from math import isnan
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, Dict, List, Sequence

import pandas as pd
from typing import TYPE_CHECKING

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


_INVALID_ID_TOKENS = {"", "nan", "none", "null"}


@dataclass
class IupharConfig:
    """Settings for querying the IUPHAR/GtoP service."""

    affinity_parameter: str = "pKi"
    approved_only: bool | None = None
    primary_target_only: bool = True


@dataclass
class PipelineConfig:
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

    rate_limit_rps: float = 2.0
    retries: int = 3
    timeout_sec: float = 30.0
    species_priority: List[str] = field(
        default_factory=lambda: ["Human", "Homo sapiens"]
    )
    list_format: str = "json"
    include_sequence: bool = False
    include_isoforms: bool = False
    columns: List[str] = field(default_factory=lambda: list(DEFAULT_COLUMNS))
    iuphar: IupharConfig = field(default_factory=IupharConfig)
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )


# ---------------------------------------------------------------------------
# Configuration helpers


def load_pipeline_config(path: str) -> PipelineConfig:
    """Load ``PipelineConfig`` from a YAML file."""

    import yaml  # type: ignore[import]

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    cfg = data.get("pipeline", {})
    iuphar = cfg.get("iuphar", {})
    cfg_kwargs: Dict[str, Any] = {
        "rate_limit_rps": cfg.get("rate_limit_rps", 2.0),
        "retries": cfg.get("retries", 3),
        "timeout_sec": cfg.get("timeout_sec", 30.0),
        "species_priority": list(
            cfg.get("species_priority", ["Human", "Homo sapiens"])
        ),
        "list_format": cfg.get("list_format", "json"),
        "include_sequence": cfg.get("include_sequence", False),
        "include_isoforms": cfg.get("include_isoforms", False),
        "columns": list(cfg.get("columns", DEFAULT_COLUMNS)),
        "iuphar": IupharConfig(
            affinity_parameter=iuphar.get("affinity_parameter", "pKi"),
            approved_only=iuphar.get("approved_only"),
            primary_target_only=iuphar.get("primary_target_only", True),
        ),
    }
    timestamp_cfg = cfg.get("timestamp_utc")
    if isinstance(timestamp_cfg, str):
        cfg_kwargs["timestamp_utc"] = timestamp_cfg
    return PipelineConfig(**cfg_kwargs)


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


def _normalise_target_ids(ids: Sequence[str]) -> List[str]:
    """Return target identifiers stripped of placeholders and whitespace.

    Parameters
    ----------
    ids:
        Sequence of raw identifiers potentially containing empty strings or
        placeholder tokens such as ``"nan"``.

    Returns
    -------
    list[str]
        Cleaned identifiers preserving the original order with placeholder
        values removed.
    """

    cleaned: List[str] = []
    for raw in ids:
        if raw is None:
            continue
        if isinstance(raw, float) and isnan(raw):
            continue
        candidate = str(raw).strip()
        if not candidate:
            continue
        if candidate.lower() in _INVALID_ID_TOKENS:
            continue
        cleaned.append(candidate)
    return cleaned


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
) -> pd.DataFrame:
    """Orchestrate data acquisition for ``ids``.

    Parameters
    ----------
    ids:
        Sequence of target ChEMBL identifiers.
    cfg:
        Pipeline configuration controlling behaviour.
    chembl_fetcher:
        Function used to download ChEMBL target information.
    chembl_config:
        Configuration passed to ``chembl_fetcher``. When ``None`` a
        :class:`TargetConfig` using the pipeline's list format is created.
    uniprot_client:
        Client instance for the UniProt service.
    hgnc_client:
        Optional client for HGNC lookups.
    gtop_client:
        Optional client for IUPHAR/GtoP data.
    ensembl_client:
        Optional client for Ensembl ortholog retrieval.
    oma_client:
        Optional client for OMA ortholog lookups.
    target_species:
        List of species considered when retrieving orthologs.
    progress_callback:
        Optional callback receiving the incremental count of processed records.
        Can be used to update external progress indicators.

    Raises
    ------
    ValueError
        If ``ids`` does not contain any valid identifiers after normalisation.
    """

    ids = _normalise_target_ids(ids)
    if not ids:
        msg = "No valid target identifiers provided."
        raise ValueError(msg)
    chembl_cfg = chembl_config or TargetConfig(list_format=cfg.list_format)
    chembl_df = chembl_fetcher(ids, chembl_cfg)
    records: List[Dict[str, Any]] = []
    for row in chembl_df.to_dict(orient="records"):
        chembl_id = row.get("target_chembl_id", "")
        comps = _load_serialised_list(row.get("target_components"), cfg.list_format)
        accessions = [c.get("accession") for c in comps if c.get("accession")]
        uniprot_entries: List[Dict[str, Any]] = []
        raw_entries: Dict[str, Dict[str, Any]] = {}
        for acc in sorted(dict.fromkeys(accessions)):
            if cfg.include_isoforms or ensembl_client or oma_client:
                raw = uniprot_client.fetch_entry_json(acc)
                if raw:
                    raw_entries[acc] = raw
                    fasta_headers: List[str] = []
                    isoforms: List[Dict[str, Any]] = []
                    if cfg.include_isoforms:
                        fasta_headers = uniprot_client.fetch_isoforms_fasta(acc)
                        isoforms = extract_isoforms(raw, fasta_headers)
                    uniprot_entries.append(
                        normalize_entry(
                            raw,
                            include_sequence=cfg.include_sequence,
                            isoforms=isoforms,
                        )
                    )
            else:
                raw = uniprot_client.fetch(acc)
                if raw:
                    raw_entries[acc] = raw
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
