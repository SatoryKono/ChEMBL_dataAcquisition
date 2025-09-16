import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional
from pathlib import Path

import pandas as pd
import requests  # type: ignore[import-untyped]

try:
    from data_profiling import analyze_table_quality
except ModuleNotFoundError:  # pragma: no cover
    from ..data_profiling import analyze_table_quality

OUTPUT_COLUMNS = [
    "uniprotkb_Id",
    "recommended_name",
    "synonyms",
    "type",
    "secondary_uniprot_id",
    "secondary_accession_names",
    "gene_name",
    "genus",
    "superkingdom",
    "phylum",
    "taxon_id",
    "ec_number",
    "hgnc_name",
    "hgnc_id",
    "molecular_function",
    "cellular_component",
    "subcellular_location",
    "topology",
    "transmembrane",
    "intramembrane",
    "glycosylation",
    "lipidation",
    "disulfide_bond",
    "modified_residue",
    "phosphorylation",
    "acetylation",
    "ubiquitination",
    "signal_peptide",
    "propeptide",
    "isoform_names",
    "isoform_ids",
    "isoform_synonyms",
    "reactions",
    "reaction_ec_numbers",
    "GuidetoPHARMACOLOGY",
    "family",
    "SUPFAM",
    "PROSITE",
    "InterPro",
    "Pfam",
    "PRINTS",
    "TCDB",
]

LOGGER = logging.getLogger(__name__)


def _serialize_list(values: Iterable[str], sep: str) -> str:
    """Serialize an iterable of strings into a single delimited string.

    Duplicate values and empty strings are removed.

    Parameters
    ----------
    values:
        An iterable of strings to serialize.
    sep:
        The separator to use between values.

    Returns
    -------
    str
        A single string with the values joined by the separator.
    """
    seen = set()
    items = []
    for v in values:
        v = v.strip()
        if v and v not in seen:
            items.append(v)
            seen.add(v)
    return sep.join(items)


class UniProtClient:
    """A lightweight client for the UniProt REST API.

    This client provides methods to fetch and parse protein data from UniProt,
    with support for batching, caching, and retries.

    Attributes
    ----------
    session:
        A `requests.Session` object for making HTTP requests.
    max_workers:
        The maximum number of threads to use for concurrent requests.
    base_url:
        The base URL for the UniProt API.
    cache:
        A dictionary to cache fetched UniProt entries.
    list_sep:
        The separator to use for joining list values in the output.
    """

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        max_workers: int = 4,
        base_url: str = "https://rest.uniprot.org/uniprotkb",
        list_sep: str = "|",
    ) -> None:
        self.session = session or requests.Session()
        self.max_workers = max_workers
        self.base_url = base_url.rstrip("/")
        self.cache: Dict[str, Dict[str, str]] = {}
        self.list_sep = list_sep

    # Retry logic with exponential backoff
    def _request(self, method: str, url: str, **kwargs) -> Optional[requests.Response]:
        """Perform an HTTP request with exponential backoff.

        This method attempts a request up to 5 times, with an increasing delay
        between attempts. It retries on network errors and specific HTTP status
        codes (429, 500, 502, 503, 504).

        Parameters
        ----------
        method:
            The HTTP method to use (e.g., "GET", "POST").
        url:
            The URL to request.
        **kwargs:
            Additional keyword arguments to pass to `requests.request`.

        Returns
        -------
        Optional[requests.Response]
            The HTTP response object if the request is successful, otherwise None.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(5):
            try:
                resp = self.session.request(method, url, timeout=30, **kwargs)
            except requests.RequestException as exc:  # network errors
                last_exc = exc
                wait = (2**attempt) + random.random()
                time.sleep(wait)
                continue
            if resp.status_code == 200:
                return resp
            if resp.status_code in {429, 500, 502, 503, 504}:
                wait = (2**attempt) + random.random()
                time.sleep(wait)
                continue
            return resp
        if last_exc:
            raise RuntimeError(f"Failed to fetch {url}: {last_exc}")
        return None

    def _fetch_single(self, accession: str) -> Optional[dict]:
        """Fetch a single UniProt entry by its accession number.

        Parameters
        ----------
        accession:
            The UniProt accession number.

        Returns
        -------
        Optional[dict]
            A dictionary representing the UniProt entry, or None if not found.
        """
        url = f"{self.base_url}/{accession}?format=json"
        resp = self._request("GET", url)
        if resp and resp.status_code == 200:
            return resp.json()
        return None

    def _fetch_batch(self, accessions: List[str]) -> List[dict]:
        """Fetch a batch of UniProt entries using a single search query.

        Parameters
        ----------
        accessions:
            A list of UniProt accession numbers.

        Returns
        -------
        List[dict]
            A list of dictionaries, each representing a UniProt entry.
        """
        query = " OR ".join(f"accession:{a}" for a in accessions)
        params = {"query": query, "format": "json", "size": len(accessions)}
        url = f"{self.base_url}/search"
        resp = self._request("GET", url, params=params)
        if resp and resp.status_code == 200:
            return resp.json().get("results", [])
        return []

    def fetch_all(self, accessions: Iterable[str]) -> Dict[str, Dict[str, str]]:
        """Fetch and parse all specified UniProt entries.

        This method orchestrates fetching all requested accessions, using either
        batch or individual requests based on the number of unique IDs. It uses a
        `ThreadPoolExecutor` for concurrent fetching of smaller batches.

        Parameters
        ----------
        accessions:
            An iterable of UniProt accession numbers.

        Returns
        -------
        Dict[str, Dict[str, str]]
            A dictionary mapping each accession to its parsed UniProt data.
        """
        unique = list(dict.fromkeys(accessions))
        results: Dict[str, Dict[str, str]] = {}
        if len(unique) > 100:
            for i in range(0, len(unique), 100):
                chunk = unique[i : i + 100]
                for entry in self._fetch_batch(chunk):
                    parsed = _parse_entry(entry, self.list_sep)
                    acc = entry.get("primaryAccession", "")
                    results[acc] = parsed
                    self.cache[acc] = parsed
            for acc in unique:
                if acc not in results:
                    results[acc] = {c: "" for c in OUTPUT_COLUMNS}
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                future_map = {pool.submit(self.fetch_entry, acc): acc for acc in unique}
                for fut in as_completed(future_map):
                    acc = future_map[fut]
                    results[acc] = fut.result()
        return results

    def fetch_entry(self, accession: str) -> Dict[str, str]:
        """Fetch, parse, and cache a single UniProt entry.

        If the entry is already in the cache, it is returned directly. Otherwise,
        it is fetched, parsed, and stored in the cache before being returned.
        This method also resolves names for secondary accession IDs.

        Parameters
        ----------
        accession:
            The UniProt accession number.

        Returns
        -------
        Dict[str, str]
            A dictionary of parsed data for the UniProt entry.
        """
        if accession in self.cache:
            return self.cache[accession]
        data = self._fetch_single(accession)
        if not data:
            parsed = {c: "" for c in OUTPUT_COLUMNS}
        else:
            parsed = _parse_entry(data, self.list_sep)
            sec_ids = parsed.get("secondary_uniprot_id", "")
            if sec_ids:
                sec_names: List[str] = []
                for sid in sec_ids.split(self.list_sep):
                    entry_sec = self._fetch_single(sid)
                    if entry_sec:
                        sec_names.extend(_protein_names_from_entry(entry_sec))
                parsed["secondary_accession_names"] = _serialize_list(
                    sec_names, self.list_sep
                )
            else:
                parsed["secondary_accession_names"] = ""
        self.cache[accession] = parsed
        return parsed


# Helper functions ---------------------------------------------------------


def _feature_to_string(feature: dict) -> str:
    """Format a UniProt feature object into a descriptive string.

    The string includes the feature's description and its position, if
    available.

    Parameters
    ----------
    feature:
        A dictionary representing a feature from a UniProt entry.

    Returns
    -------
    str
        A formatted string describing the feature.
    """
    desc = feature.get("description", "").strip()
    loc = feature.get("location", {})
    start = loc.get("start", {}).get("value")
    end = loc.get("end", {}).get("value")
    pos = ""
    if start is not None and end is not None:
        pos = str(start) if start == end else f"{start}-{end}"
    if desc and pos:
        return f"{desc}[{pos}]"
    if desc:
        return desc
    if pos:
        return f"[{pos}]"
    return ""


def _collect_ec_numbers(name_obj: dict) -> List[str]:
    """Extract EC numbers from a UniProt name object.

    Parameters
    ----------
    name_obj:
        Name or reaction object potentially containing ``ecNumbers`` or
        ``ecNumber`` entries.

    Returns
    -------
    List[str]
        All EC numbers found within ``name_obj``.
    """

    numbers: List[str] = []
    ec_data = name_obj.get("ecNumbers") or name_obj.get("ecNumber")
    if isinstance(ec_data, list):
        for item in ec_data:
            if isinstance(item, dict):
                value = item.get("value")
                if value:
                    numbers.append(value)
            elif isinstance(item, str):
                numbers.append(item)
    elif isinstance(ec_data, dict):
        value = ec_data.get("value")
        if value:
            numbers.append(value)
    elif isinstance(ec_data, str):
        numbers.append(ec_data)
    return numbers


def _protein_names_from_entry(entry: dict) -> List[str]:
    """Return protein names from ``entry``.

    The function collects the recommended full name and all alternative full
    names. It is used to resolve names for secondary accession records.
    """

    names: List[str] = []
    desc = entry.get("proteinDescription", {})
    if not isinstance(desc, dict):
        return names
    rec = desc.get("recommendedName", {})
    if isinstance(rec, dict):
        value = rec.get("fullName", {}).get("value")
        if value:
            names.append(value)
    for alt in desc.get("alternativeNames", []) or []:
        if isinstance(alt, dict):
            value = alt.get("fullName", {}).get("value")
            if value:
                names.append(value)
    return names


def _parse_entry(entry: dict, sep: str) -> Dict[str, str]:
    """Parse a raw UniProt entry dictionary into a flattened dictionary.

    This function extracts relevant fields from the nested UniProt JSON
    structure and formats them into a flat dictionary with predefined keys.

    Parameters
    ----------
    entry:
        The raw UniProt entry as a dictionary.
    sep:
        The separator to use for joining list values.

    Returns
    -------
    Dict[str, str]
        A flat dictionary containing the parsed UniProt data.
    """
    data = {c: "" for c in OUTPUT_COLUMNS}
    data["uniprotkb_Id"] = entry.get("primaryAccession", "")
    data["type"] = entry.get("entryType", "")
    # Names
    protein = entry.get("proteinDescription", {})
    rec = protein.get("recommendedName", {})
    data["recommended_name"] = rec.get("fullName", {}).get("value", "")
    synonyms: List[str] = []
    for sn in rec.get("shortNames", []):
        v = sn.get("value")
        if v:
            synonyms.append(v)
    for alt in protein.get("alternativeNames", []):
        v = alt.get("fullName", {}).get("value")
        if v:
            synonyms.append(v)
        for sn in alt.get("shortNames", []):
            v = sn.get("value")
            if v:
                synonyms.append(v)
    data["synonyms"] = _serialize_list(synonyms, sep)
    ec_numbers: List[str] = []
    for ec in rec.get("ecNumbers", []):
        v = ec.get("value")
        if v:
            ec_numbers.append(v)
    for alt in protein.get("alternativeNames", []):
        for ec in alt.get("ecNumbers", []):
            v = ec.get("value")
            if v:
                ec_numbers.append(v)
    data["ec_number"] = _serialize_list(ec_numbers, sep)
    data["secondary_uniprot_id"] = _serialize_list(
        entry.get("secondaryAccessions", []), sep
    )
    # Genes
    genes = entry.get("genes", [])
    if genes:
        data["gene_name"] = genes[0].get("geneName", {}).get("value", "")
    # Taxonomy
    organism = entry.get("organism", {})
    lineage = organism.get("lineage", [])
    if lineage:
        data["superkingdom"] = lineage[0] if len(lineage) > 0 else ""
        data["phylum"] = lineage[2] if len(lineage) > 2 else ""
        data["genus"] = lineage[-1]
    data["taxon_id"] = str(organism.get("taxonId", ""))
    # Cross references
    hgnc_names: List[str] = []
    hgnc_ids: List[str] = []
    mf_terms: List[str] = []
    cc_terms: List[str] = []
    guide_ids: List[str] = []
    family_ids: List[str] = []
    supfam_ids: List[str] = []
    prosite_ids: List[str] = []
    interpro_ids: List[str] = []
    pfam_ids: List[str] = []
    prints_ids: List[str] = []
    tcdb_ids: List[str] = []
    for ref in entry.get("uniProtKBCrossReferences", []):
        db = ref.get("database")
        if db == "HGNC":
            hgnc_ids.append(ref.get("id", ""))
            for prop in ref.get("properties", []):
                if prop.get("key") == "Name":
                    hgnc_names.append(prop.get("value", ""))
        elif db == "GO":
            term = ""
            for prop in ref.get("properties", []):
                if prop.get("key") == "GoTerm":
                    term = prop.get("value", "")
                    break
            if term.startswith("F:"):
                mf_terms.append(f"{term[2:]} ({ref.get('id')})")
            elif term.startswith("C:"):
                cc_terms.append(f"{term[2:]} ({ref.get('id')})")
        elif db == "GuidetoPHARMACOLOGY":
            guide_ids.append(ref.get("id", ""))
        elif db == "family":
            family_ids.append(ref.get("id", ""))
        elif db == "SUPFAM":
            supfam_ids.append(ref.get("id", ""))
        elif db == "PROSITE":
            prosite_ids.append(ref.get("id", ""))
        elif db == "InterPro":
            interpro_ids.append(ref.get("id", ""))
        elif db == "Pfam":
            pfam_ids.append(ref.get("id", ""))
        elif db == "PRINTS":
            prints_ids.append(ref.get("id", ""))
        elif db == "TCDB":
            tcdb_ids.append(ref.get("id", ""))
    data["hgnc_name"] = _serialize_list(hgnc_names, sep)
    data["hgnc_id"] = _serialize_list(hgnc_ids, sep)
    data["molecular_function"] = _serialize_list(mf_terms, sep)
    data["cellular_component"] = _serialize_list(cc_terms, sep)
    data["GuidetoPHARMACOLOGY"] = _serialize_list(guide_ids, sep)
    data["family"] = _serialize_list(family_ids, sep)
    data["SUPFAM"] = _serialize_list(supfam_ids, sep)
    data["PROSITE"] = _serialize_list(prosite_ids, sep)
    data["InterPro"] = _serialize_list(interpro_ids, sep)
    data["Pfam"] = _serialize_list(pfam_ids, sep)
    data["PRINTS"] = _serialize_list(prints_ids, sep)
    data["TCDB"] = _serialize_list(tcdb_ids, sep)
    # Comments
    sublocs: List[str] = []
    topologies: List[str] = []
    isoform_names: List[str] = []
    isoform_ids: List[str] = []
    isoform_synonyms: List[str] = []
    reactions: List[str] = []
    reaction_ecs: List[str] = []
    for comment in entry.get("comments", []):
        ctype = comment.get("commentType")
        if ctype == "SUBCELLULAR_LOCATION":
            for loc in comment.get("subcellularLocations", []):
                if not isinstance(loc, dict):
                    continue
                sub = loc.get("location")
                if isinstance(sub, dict):
                    value = sub.get("value")
                    if isinstance(value, str):
                        sublocs.append(value)
                topo = loc.get("topology")
                if isinstance(topo, dict):
                    value = topo.get("value")
                    if isinstance(value, str):
                        topologies.append(value)
        elif ctype == "ALTERNATIVE_PRODUCTS":
            for iso in comment.get("isoforms", []):
                name = iso.get("name", {}).get("value")
                if name:
                    isoform_names.append(name)
                iso_id = iso.get("id")
                if iso_id:
                    isoform_ids.append(iso_id)
                for syn in iso.get("synonyms", []):
                    v = syn.get("value")
                    if v:
                        isoform_synonyms.append(v)
        elif ctype == "CATALYTIC_ACTIVITY":
            reaction = comment.get("reaction", {})
            if isinstance(reaction, dict):
                name = reaction.get("name")
                if isinstance(name, dict):
                    name = name.get("value")
                if name:
                    reactions.append(name)
                reaction_ecs.extend(_collect_ec_numbers(reaction))
    data["subcellular_location"] = _serialize_list(sublocs, sep)
    data["isoform_names"] = _serialize_list(isoform_names, sep)
    data["isoform_ids"] = _serialize_list(isoform_ids, sep)
    data["isoform_synonyms"] = _serialize_list(isoform_synonyms, sep)
    data["reactions"] = _serialize_list(reactions, sep)
    data["reaction_ec_numbers"] = _serialize_list(reaction_ecs, sep)

    # Features --------------------------------------------------------
    topo_feats: List[str] = []
    trans_feats: List[str] = []
    intra_feats: List[str] = []
    glyco_feats: List[str] = []
    lipid_feats: List[str] = []
    disulfide_feats: List[str] = []
    modres_feats: List[str] = []
    phospho_feats: List[str] = []
    acetyl_feats: List[str] = []
    ubiquit_feats: List[str] = []
    signal_feats: List[str] = []
    propep_feats: List[str] = []

    features = entry.get("features", [])
    if isinstance(features, list):
        for feat in features:
            if not isinstance(feat, dict):
                continue
            ftype = feat.get("type")
            desc = feat.get("description", "").lower()
            value = _feature_to_string(feat)
            if ftype in {"Topological domain"}:
                topo_feats.append(value)
            elif ftype in {"Transmembrane region", "TRANSMEMBRANE"}:
                trans_feats.append(value)
            elif ftype in {"Intramembrane region", "INTRAMEMBRANE"}:
                intra_feats.append(value)
            elif ftype == "Glycosylation":
                glyco_feats.append(value)
            elif ftype == "Lipidation":
                lipid_feats.append(value)
            elif ftype == "Disulfide bond":
                disulfide_feats.append(value)
            elif ftype == "Signal peptide":
                signal_feats.append(value)
            elif ftype == "Propeptide":
                propep_feats.append(value)
            elif ftype == "Modified residue":
                if "phospho" in desc:
                    phospho_feats.append(value)
                elif "acetyl" in desc:
                    acetyl_feats.append(value)
                elif "ubiquitin" in desc:
                    ubiquit_feats.append(value)
                else:
                    modres_feats.append(value)

    data["topology"] = _serialize_list(topologies + topo_feats, sep)
    data["transmembrane"] = "1" if trans_feats else ""
    data["intramembrane"] = "1" if intra_feats else ""
    data["glycosylation"] = _serialize_list(glyco_feats, sep)
    data["lipidation"] = _serialize_list(lipid_feats, sep)
    data["disulfide_bond"] = _serialize_list(disulfide_feats, sep)
    data["modified_residue"] = _serialize_list(modres_feats, sep)
    data["phosphorylation"] = _serialize_list(phospho_feats, sep)
    data["acetylation"] = _serialize_list(acetyl_feats, sep)
    data["ubiquitination"] = _serialize_list(ubiquit_feats, sep)
    data["signal_peptide"] = _serialize_list(signal_feats, sep)
    data["propeptide"] = _serialize_list(propep_feats, sep)
    return data


# Public API ---------------------------------------------------------------


def enrich_uniprot(input_csv_path: str, list_sep: str = "|") -> None:
    """Read a CSV file and enrich with UniProt annotations.

    Parameters
    ----------
    input_csv_path : str
        Path to input CSV containing a ``uniprot_id`` column.
    list_sep : str, optional
        Separator used for serializing lists, by default "|".
    """
    df = pd.read_csv(input_csv_path, dtype={"uniprot_id": str})
    if "uniprot_id" not in df.columns:
        raise ValueError("input CSV must contain 'uniprot_id' column")
    ids = df["uniprot_id"].astype(str).tolist()
    unique_ids = list(dict.fromkeys(ids))
    LOGGER.info("Unique UniProt IDs: %d", len(unique_ids))
    client = UniProtClient(list_sep=list_sep)
    id_map = client.fetch_all(unique_ids)
    success = sum(1 for v in id_map.values() if any(v.values()))
    LOGGER.info("Fetched %d records, %d failures", success, len(id_map) - success)
    for col in OUTPUT_COLUMNS:
        df[col] = [id_map.get(i, {}).get(col, "") for i in ids]
    # backup
    backup_path = f"{input_csv_path}.bak"
    df_original = pd.read_csv(input_csv_path)
    df_original.to_csv(backup_path, index=False, encoding="utf-8", lineterminator="\n")
    df.to_csv(input_csv_path, index=False, encoding="utf-8", lineterminator="\n")
    analyze_table_quality(df, table_name=str(Path(input_csv_path).with_suffix("")))
