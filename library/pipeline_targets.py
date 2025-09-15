from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Sequence, cast

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
        Isoform,
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
            Isoform,
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


# ---------------------------------------------------------------------------
# Configuration helpers


def load_pipeline_config(path: str) -> PipelineConfig:
    """Load ``PipelineConfig`` from a YAML file."""

    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    cfg = data.get("pipeline", {})
    iuphar = cfg.get("iuphar", {})
    return PipelineConfig(
        rate_limit_rps=cfg.get("rate_limit_rps", 2.0),
        retries=cfg.get("retries", 3),
        timeout_sec=cfg.get("timeout_sec", 30.0),
        species_priority=list(cfg.get("species_priority", ["Human", "Homo sapiens"])),
        list_format=cfg.get("list_format", "json"),
        include_sequence=cfg.get("include_sequence", False),
        include_isoforms=cfg.get("include_isoforms", False),
        columns=list(cfg.get("columns", DEFAULT_COLUMNS)),
        iuphar=IupharConfig(
            affinity_parameter=iuphar.get("affinity_parameter", "pKi"),
            approved_only=iuphar.get("approved_only"),
            primary_target_only=iuphar.get("primary_target_only", True),
        ),
    )


# ---------------------------------------------------------------------------
# Utility functions


def _serialise_list(items: Sequence[Any], list_format: str) -> str:
    cleaned = [x for x in items if x not in (None, "")]
    cleaned = sorted(dict.fromkeys(cleaned))
    if list_format == "pipe":
        return "|".join(str(x) for x in cleaned)
    return json.dumps(cleaned, ensure_ascii=False, sort_keys=True)


def _select_primary(
    entries: List[Dict[str, Any]], priority: Sequence[str]
) -> Dict[str, Any] | None:
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
    chembl_fetcher: Callable[
        [Sequence[str], TargetConfig], pd.DataFrame
    ] = fetch_targets,
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
    """
    chembl_cfg = chembl_config or TargetConfig(list_format=cfg.list_format)
    chembl_df = chembl_fetcher(ids, chembl_cfg)
    records: List[Dict[str, Any]] = []
    for row in chembl_df.to_dict(orient="records"):
        chembl_id = str(row.get("target_chembl_id", ""))
        comps = json.loads(str(row.get("target_components") or "[]"))
        accessions = [
            str(c["accession"])
            for c in comps
            if isinstance(c, dict) and isinstance(c.get("accession"), str)
        ]
        uniprot_entries: List[Dict[str, Any]] = []
        raw_entries: Dict[str, Dict[str, Any]] = {}
        for acc in sorted(dict.fromkeys(accessions)):
            if cfg.include_isoforms or ensembl_client or oma_client:
                raw = uniprot_client.fetch_entry_json(acc)
                if raw:
                    raw_entries[acc] = raw
                    fasta_headers: List[str] = []
                    isoforms: List[Isoform] = []
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
        primary_id = cast(str, primary.get("uniprot_id", "")) if primary else ""
        all_ids = [str(e.get("uniprot_id", "")) for e in uniprot_entries]

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
            target: Dict[str, Any] | None = None
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
                target_id = int(target.get("targetId", 0))
                gtop_id = str(target_id)
                gtop_name = str(target.get("name", ""))
                syn_df = normalise_synonyms(
                    target_id,
                    gtop_client.fetch_target_endpoint(target_id, "synonyms"),
                )
                gtop_synonyms = syn_df["synonym"].tolist() if not syn_df.empty else []
                nat = gtop_client.fetch_target_endpoint(target_id, "naturalLigands")
                gtop_nat = len(nat or [])
                params: Dict[str, Any] = {}
                if cfg.iuphar.approved_only:
                    params["approved"] = "true"
                if cfg.iuphar.primary_target_only:
                    params["primaryTarget"] = "true"
                inter = gtop_client.fetch_target_endpoint(
                    target_id, "interactions", params
                )
                inter_df = normalise_interactions(target_id, inter)
                gtop_int = len(inter_df)
                func = gtop_client.fetch_target_endpoint(target_id, "function")
                if func:
                    first = func[0]
                    gtop_func = str(first.get("functionText", ""))[:200]

        isoform_ids = primary.get("isoform_ids_all", []) if primary else []
        isoform_names = primary.get("isoform_names", []) if primary else []
        isoform_synonyms: List[str] = []
        if cfg.include_isoforms and primary:
            try:
                iso_data = json.loads(primary.get("isoforms_json", "[]"))
            except json.JSONDecodeError:
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

        pc = json.loads(row.get("protein_classifications") or "[]")
        pc += [""] * (5 - len(pc))
        cross_refs = json.loads(row.get("cross_references") or "[]")
        chembl_refs = [
            r.get("xref_id") for r in cross_refs if r.get("xref_db") == "ChEMBL"
        ]

        names: set[str] = set()
        synonyms: set[str] = set()
        if row.get("pref_name"):
            names.add(row.get("pref_name", ""))
        syn_list = json.loads(row.get("protein_synonym_list") or "[]")
        synonyms.update(syn_list)
        gene_list = json.loads(row.get("gene_symbol_list") or "[]")
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
                    else json.loads(row.get("protein_synonym_list") or "[]")
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
            "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        records.append(rec)
        if progress_callback:
            progress_callback(1)

    df = pd.DataFrame(records, columns=cfg.columns)
    return df


__all__ = ["PipelineConfig", "IupharConfig", "load_pipeline_config", "run_pipeline"]
