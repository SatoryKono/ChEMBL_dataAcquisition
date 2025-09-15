from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Sequence

import pandas as pd

from chembl_targets import TargetConfig, fetch_targets
from gtop_client import GtoPClient, resolve_target
from gtop_normalize import normalise_interactions, normalise_synonyms
from hgnc_client import HGNCClient, HGNCRecord
from orthologs import EnsemblHomologyClient, OmaClient, Ortholog
from uniprot_client import UniProtClient
from uniprot_normalize import extract_ensembl_gene_ids, normalize_entry

LOGGER = logging.getLogger(__name__)

PIPELINE_VERSION = "0.1.0"

# Default columns for final output files
DEFAULT_COLUMNS = [
    "target_chembl_id",
    "uniprot_id_primary",
    "uniprot_ids_all",
    "hgnc_id",
    "gene_symbol",
    "protein_name_canonical",
    "protein_name_alt",
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
    "gtop_natural_ligands_n",
    "gtop_interactions_n",
    "gtop_function_text_short",
    "uniprot_isoforms",
    "orthologs_json",
    "orthologs_count",
    "names_synonyms_all",
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
    ortholog_target_species:
        Species names for which orthologs should be retrieved.
    """

    rate_limit_rps: float = 2.0
    retries: int = 3
    timeout_sec: float = 30.0
    species_priority: List[str] = field(
        default_factory=lambda: ["Human", "Homo sapiens"]
    )
    list_format: str = "json"
    include_sequence: bool = False
    columns: List[str] = field(default_factory=lambda: list(DEFAULT_COLUMNS))
    iuphar: IupharConfig = field(default_factory=IupharConfig)
    ortholog_target_species: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Configuration helpers


def load_pipeline_config(path: str) -> PipelineConfig:
    """Load ``PipelineConfig`` from a YAML file."""

    import yaml  # type: ignore[import]

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    cfg = data.get("pipeline", {})
    iuphar = cfg.get("iuphar", {})
    orth = data.get("orthologs", {})
    return PipelineConfig(
        rate_limit_rps=cfg.get("rate_limit_rps", 2.0),
        retries=cfg.get("retries", 3),
        timeout_sec=cfg.get("timeout_sec", 30.0),
        species_priority=list(cfg.get("species_priority", ["Human", "Homo sapiens"])),
        list_format=cfg.get("list_format", "json"),
        include_sequence=cfg.get("include_sequence", False),
        columns=list(cfg.get("columns", DEFAULT_COLUMNS)),
        iuphar=IupharConfig(
            affinity_parameter=iuphar.get("affinity_parameter", "pKi"),
            approved_only=iuphar.get("approved_only"),
            primary_target_only=iuphar.get("primary_target_only", True),
        ),
        ortholog_target_species=list(orth.get("target_species", [])),
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
    chembl_fetcher=fetch_targets,
    uniprot_client: UniProtClient,
    hgnc_client: HGNCClient | None = None,
    gtop_client: GtoPClient | None = None,
    ensembl_client: EnsemblHomologyClient | None = None,
    oma_client: OmaClient | None = None,
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
    uniprot_client:
        Client instance for the UniProt service.
    hgnc_client:
        Optional client for HGNC lookups.
    gtop_client:
        Optional client for IUPHAR/GtoP data.
    ensembl_client:
        Optional client for Ensembl ortholog lookups.
    oma_client:
        Optional fallback client for OMA ortholog lookups.
    """

    chembl_df = chembl_fetcher(ids, TargetConfig(list_format="json"))
    records: List[Dict[str, Any]] = []
    for row in chembl_df.to_dict(orient="records"):
        chembl_id = row.get("target_chembl_id", "")
        comps = json.loads(row.get("target_components") or "[]")
        accessions = [c.get("accession") for c in comps if c.get("accession")]
        uniprot_entries: List[tuple[Dict[str, Any], List[str]]] = []
        for acc in sorted(dict.fromkeys(accessions)):
            raw = uniprot_client.fetch(acc)
            if raw:
                entry = normalize_entry(raw, include_sequence=cfg.include_sequence)
                gene_ids = extract_ensembl_gene_ids(raw)
                uniprot_entries.append((entry, gene_ids))
        primary_entry = _select_primary(
            [e[0] for e in uniprot_entries], cfg.species_priority
        )
        primary_gene_ids: List[str] = []
        if primary_entry:
            for entry, genes in uniprot_entries:
                if entry is primary_entry:
                    primary_gene_ids = genes
                    break
        primary = primary_entry
        primary_id = primary.get("uniprot_id") if primary else ""
        all_ids = [e[0].get("uniprot_id") for e in uniprot_entries]

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

        orthologs: List[Ortholog] = []
        if (
            cfg.ortholog_target_species
            and primary_gene_ids
            and ensembl_client is not None
        ):
            for gene_id in primary_gene_ids:
                orthologs.extend(
                    ensembl_client.get_orthologs(gene_id, cfg.ortholog_target_species)
                )
            if not orthologs and oma_client is not None and primary_id:
                orthologs = oma_client.get_orthologs_by_uniprot(primary_id)
        orthologs_json = json.dumps(
            [o.to_ordered_dict() for o in orthologs],
            ensure_ascii=False,
            sort_keys=True,
        )

        pc = json.loads(row.get("protein_classifications") or "[]")
        pc += [""] * (5 - len(pc))
        cross_refs = json.loads(row.get("cross_references") or "[]")
        chembl_refs = [
            r.get("xref_id") for r in cross_refs if r.get("xref_db") == "ChEMBL"
        ]
        chembl_gene_symbols = json.loads(row.get("gene_symbol_list") or "[]")
        names_synonyms: List[str] = []
        if row.get("pref_name"):
            names_synonyms.append(row.get("pref_name", ""))
        names_synonyms.extend(chembl_gene_symbols)
        if primary:
            names_synonyms.append(primary.get("protein_recommended_name", ""))
            names_synonyms.extend(primary.get("protein_alternative_names", []))
            names_synonyms.append(primary.get("gene_primary", ""))
            names_synonyms.extend(primary.get("gene_synonyms", []))
        if hgnc_rec:
            if hgnc_rec.gene_symbol:
                names_synonyms.append(hgnc_rec.gene_symbol)
            if hgnc_rec.gene_name:
                names_synonyms.append(hgnc_rec.gene_name)
        names_synonyms.extend(gtop_synonyms)
        names_synonyms_str = _serialise_list(names_synonyms, "pipe")

        rec: Dict[str, Any] = {
            "target_chembl_id": chembl_id,
            "uniprot_id_primary": primary_id,
            "uniprot_ids_all": _serialise_list(all_ids, cfg.list_format),
            "hgnc_id": hgnc_id,
            "gene_symbol": gene_symbol,
            "protein_name_canonical": (
                primary.get("protein_recommended_name")
                if primary
                else row.get("pref_name", "")
            ),
            "protein_name_alt": _serialise_list(
                primary.get("protein_alternative_names", []) if primary else [],
                cfg.list_format,
            ),
            "organism": (
                primary.get("organism_name", row.get("organism", ""))
                if primary
                else row.get("organism", "")
            ),
            "taxon_id": primary.get("taxon_id", "") if primary else "",
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
            "gtop_natural_ligands_n": gtop_nat,
            "gtop_interactions_n": gtop_int,
            "gtop_function_text_short": gtop_func,
            "uniprot_isoforms": _serialise_list(
                primary.get("isoform_ids", []) if primary else [], "pipe"
            ),
            "orthologs_json": orthologs_json,
            "orthologs_count": len(orthologs),
            "names_synonyms_all": names_synonyms_str,
            "uniprot_last_update": (
                primary.get("last_annotation_update", "") if primary else ""
            ),
            "uniprot_version": primary.get("entry_version", "") if primary else "",
            "pipeline_version": PIPELINE_VERSION,
            "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        records.append(rec)

    df = pd.DataFrame(records, columns=cfg.columns)
    return df


__all__ = ["PipelineConfig", "IupharConfig", "load_pipeline_config", "run_pipeline"]
