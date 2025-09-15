"""Utilities to normalise UniProtKB JSON records into tabular form."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Tuple

ORDERED_COLUMNS_BASE = [
    "uniprot_id",
    "entry_type",
    "protein_recommended_name",
    "protein_alternative_names",
    "gene_primary",
    "gene_synonyms",
    "organism_name",
    "taxon_id",
    "lineage_superkingdom",
    "lineage_phylum",
    "lineage_class",
    "lineage_order",
    "lineage_family",
    "protein_existence",
    "sequence_length",
    "sequence_mass",
    "sequence_md5",
    # "sequence" optionally inserted here
    "subcellular_location",
    "function",
    "catalytic_activity",
    "cofactor",
    "pathway",
    "tissue_specificity",
    "expression",
    "features_signal_peptide",
    "features_transmembrane",
    "features_topology",
    "ptm_glycosylation",
    "ptm_lipidation",
    "ptm_disulfide_bond",
    "ptm_modified_residue",
    "isoform_ids",
    "isoform_names",
    "domains_pfam",
    "domains_interpro",
    "3d_pdb_ids",
    "alphafold_id",
    "xref_chembl_target",
    "xref_hgnc",
    "xref_ensembl",
    "last_annotation_update",
    "entry_version",
]


def output_columns(include_sequence: bool) -> List[str]:
    cols = ORDERED_COLUMNS_BASE.copy()
    if include_sequence:
        cols.insert(17, "sequence")
    return cols


# ---------------------------------------------------------------------------
# Helper functions


def _get(entry: Dict[str, Any], *path: str) -> Any:
    cur: Any = entry
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur


def _loc_to_range(feature: Dict[str, Any]) -> str:
    loc = feature.get("location", {})
    start = _get(loc, "start", "value")
    end = _get(loc, "end", "value")
    if start is None and end is None:
        return ""
    if start == end:
        return str(start)
    return f"{start}-{end}"


def _collect_comment(entry: Dict[str, Any], ctype: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for c in entry.get("comments", []):
        if c.get("commentType") == ctype:
            items.append(c)
    return items


def _collect_cross_refs(entry: Dict[str, Any], db: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for xref in entry.get("uniProtKBCrossReferences", []):
        if xref.get("database") == db:
            out.append(xref)
    return out


# ---------------------------------------------------------------------------
# Normalisation


def normalize_entry(
    entry: Dict[str, Any], include_sequence: bool = False
) -> Dict[str, Any]:
    """Normalise a raw UniProt record into a flat dictionary.

    Parameters
    ----------
    entry:
        Parsed JSON document as returned by the UniProt API.
    include_sequence:
        Whether to include the full amino acid sequence.
    """

    result: Dict[str, Any] = {c: "" for c in output_columns(include_sequence)}

    result["uniprot_id"] = entry.get("primaryAccession", "")
    result["entry_type"] = entry.get("entryType", "").lower()

    # Protein names -----------------------------------------------------
    prot = entry.get("proteinDescription", {})
    rec = prot.get("recommendedName", {})
    result["protein_recommended_name"] = _get(rec, "fullName", "value") or ""
    alt_names: List[str] = []
    for sn in rec.get("shortNames", []):
        val = sn.get("value")
        if val:
            alt_names.append(val)
    for alt in prot.get("alternativeNames", []):
        val = _get(alt, "fullName", "value")
        if val:
            alt_names.append(val)
        for sn in alt.get("shortNames", []):
            val = sn.get("value")
            if val:
                alt_names.append(val)
    result["protein_alternative_names"] = sorted(set(alt_names))

    # Gene names --------------------------------------------------------
    genes = entry.get("genes", [])
    if genes:
        g0 = genes[0]
        result["gene_primary"] = _get(g0, "geneName", "value") or ""
        syns = [s.get("value") for s in g0.get("synonyms", []) if s.get("value")]
        result["gene_synonyms"] = sorted(set(syns))

    # Organism ----------------------------------------------------------
    org = entry.get("organism", {})
    result["organism_name"] = org.get("scientificName", "")
    result["taxon_id"] = str(org.get("taxonId", ""))
    lineage = org.get("lineage", [])
    levels = [
        "lineage_superkingdom",
        "lineage_phylum",
        "lineage_class",
        "lineage_order",
        "lineage_family",
    ]
    for lvl, name in zip(levels, lineage):
        result[lvl] = name

    # Protein existence -------------------------------------------------
    result["protein_existence"] = entry.get("proteinExistence", "")

    seq = entry.get("sequence", {})
    result["sequence_length"] = seq.get("length", "")
    result["sequence_mass"] = seq.get("molWeight", "")
    md5 = seq.get("sequenceChecksum")
    if not md5 and seq.get("value"):
        md5 = hashlib.md5(seq["value"].encode()).hexdigest()
    result["sequence_md5"] = md5 or ""
    if include_sequence:
        result["sequence"] = seq.get("value", "")

    # Comments ----------------------------------------------------------
    sub_loc = []
    for c in _collect_comment(entry, "SUBCELLULAR_LOCATION"):
        for loc in c.get("subcellularLocations", []):
            val = _get(loc, "location", "value")
            if val:
                sub_loc.append(val)
    result["subcellular_location"] = sorted(set(sub_loc))

    funcs = []
    for c in _collect_comment(entry, "FUNCTION"):
        for txt in c.get("texts", []):
            val = txt.get("value")
            if val:
                funcs.append(val)
    result["function"] = " ".join(funcs)

    catal = []
    for c in _collect_comment(entry, "CATALYTIC_ACTIVITY"):
        name = _get(c, "reaction", "name")
        if name:
            catal.append(name)
    result["catalytic_activity"] = sorted(set(catal))

    cof = []
    for c in _collect_comment(entry, "COFACTOR"):
        for cf in c.get("cofactors", []):
            val = _get(cf, "name", "value")
            if val:
                cof.append(val)
    result["cofactor"] = sorted(set(cof))

    path = []
    for c in _collect_comment(entry, "PATHWAY"):
        for p in c.get("pathways", []):
            val = p.get("value")
            if val:
                path.append(val)
    result["pathway"] = sorted(set(path))

    ts = []
    for c in _collect_comment(entry, "TISSUE_SPECIFICITY"):
        for txt in c.get("texts", []):
            val = txt.get("value")
            if val:
                ts.append(val)
    result["tissue_specificity"] = " ".join(ts)

    expr = []
    for c in _collect_comment(entry, "EXPRESSION"):
        for txt in c.get("texts", []):
            val = txt.get("value")
            if val:
                expr.append(val)
    result["expression"] = " ".join(expr)

    # Features ----------------------------------------------------------
    feat_sig: List[Tuple[int, str]] = []
    feat_tm: List[Tuple[int, str]] = []
    feat_topo: List[Tuple[int, str]] = []
    glyco: List[Tuple[int, str]] = []
    lipid: List[Tuple[int, str]] = []
    disul: List[Tuple[int, str]] = []
    modres: List[Tuple[int, str]] = []
    for feat in entry.get("features", []):
        ftype = feat.get("type", "")
        rng = _loc_to_range(feat)
        start = _get(feat, "location", "start", "value") or 0
        desc = feat.get("description", "")
        item = (int(start), f"{rng}:{desc}" if desc else rng)
        if ftype == "Signal peptide":
            feat_sig.append(item)
        elif ftype == "Transmembrane":
            feat_tm.append(item)
        elif ftype == "Topological domain":
            feat_topo.append(item)
        elif ftype == "Glycosylation":
            glyco.append(item)
        elif ftype == "Lipidation":
            lipid.append(item)
        elif ftype == "Disulfide bond":
            disul.append(item)
        elif ftype == "Modified residue":
            modres.append(item)
    result["features_signal_peptide"] = [v for _, v in sorted(feat_sig)]
    result["features_transmembrane"] = [v for _, v in sorted(feat_tm)]
    result["features_topology"] = [v for _, v in sorted(feat_topo)]
    result["ptm_glycosylation"] = [v for _, v in sorted(glyco)]
    result["ptm_lipidation"] = [v for _, v in sorted(lipid)]
    result["ptm_disulfide_bond"] = [v for _, v in sorted(disul)]
    result["ptm_modified_residue"] = [v for _, v in sorted(modres)]

    # Isoforms ---------------------------------------------------------
    iso_ids: List[str] = []
    iso_names: List[str] = []
    for c in _collect_comment(entry, "ALTERNATIVE_PRODUCTS"):
        for iso in c.get("isoforms", []):
            iso_id = iso.get("id")
            iso_name = _get(iso, "name", "value")
            if iso_id:
                iso_ids.append(iso_id)
            if iso_name:
                iso_names.append(iso_name)
    result["isoform_ids"] = sorted(set(iso_ids))
    result["isoform_names"] = sorted(set(iso_names))

    # Cross references -------------------------------------------------
    def _domains(db: str) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for ref in _collect_cross_refs(entry, db):
            acc = ref.get("id", "")
            name = ""
            for prop in ref.get("properties", []):
                if prop.get("key") in {"Entry name", "entry name"}:
                    name = prop.get("value", "")
            pairs.append((acc, name))
        return sorted(pairs, key=lambda x: x[0])

    result["domains_pfam"] = _domains("Pfam")
    result["domains_interpro"] = _domains("InterPro")
    pdb_ids = [ref.get("id", "") for ref in _collect_cross_refs(entry, "PDB")]
    result["3d_pdb_ids"] = sorted(set(pdb_ids))
    af = _collect_cross_refs(entry, "AlphaFoldDB")
    result["alphafold_id"] = af[0].get("id", "") if af else ""
    chembl = _collect_cross_refs(entry, "ChEMBL")
    result["xref_chembl_target"] = chembl[0].get("id", "") if chembl else ""
    hgnc = _collect_cross_refs(entry, "HGNC")
    result["xref_hgnc"] = hgnc[0].get("id", "") if hgnc else ""
    ensembl_ids = [ref.get("id", "") for ref in _collect_cross_refs(entry, "Ensembl")]
    result["xref_ensembl"] = sorted(set(ensembl_ids))

    # Entry audit ------------------------------------------------------
    audit = entry.get("entryAudit", {})
    result["last_annotation_update"] = audit.get("lastAnnotationUpdateDate", "")
    result["entry_version"] = audit.get("entryVersion", "")

    return result


def extract_ensembl_gene_ids(entry: Dict[str, Any]) -> List[str]:
    """Return Ensembl gene identifiers associated with a UniProt entry.

    Parameters
    ----------
    entry:
        Parsed UniProt JSON document.
    """

    ids: set[str] = set()
    for xref in _collect_cross_refs(entry, "Ensembl"):
        for prop in xref.get("properties", []):
            if prop.get("key") == "GeneId":
                val = prop.get("value")
                if val:
                    ids.add(str(val).split(".")[0])
    return sorted(ids)
