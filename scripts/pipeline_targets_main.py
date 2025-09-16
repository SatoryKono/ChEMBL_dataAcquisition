# ruff: noqa: E402

"""CLI entry point for the unified target data pipeline."""

from __future__ import annotations

import argparse
import logging

import sys
from pathlib import Path

from typing import Any, Callable, Dict, Iterable, List, Sequence

import pandas as pd
import yaml  # type: ignore[import]
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


from library.protein_classifier import classify_protein
from library.data_profiling import analyze_table_quality


from library.pipeline_targets import (
    PipelineConfig,
    load_pipeline_config,
    run_pipeline,
)

from library.iuphar import ClassificationRecord, IUPHARClassifier, IUPHARData


# Columns produced by :func:`add_iuphar_classification`.
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
    try:
        entry_map = fetch_entries([i for i in ids if i])
    except Exception as exc:  # pragma: no cover - logging side effect
        logging.getLogger(__name__).warning("Failed to fetch UniProt entries: %s", exc)
        entry_map = {}

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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pipeline."""

    parser = argparse.ArgumentParser(description="Unified target data pipeline")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--id-column", default="target_chembl_id")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--sep", default=",")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--list-format", default="json")
    parser.add_argument("--species", default="Human")
    parser.add_argument("--affinity-parameter", default="pKi")
    parser.add_argument("--approved-only", default="false")
    parser.add_argument("--primary-target-only", default="true")
    parser.add_argument("--with-isoforms", action="store_true")
    parser.add_argument("--with-orthologs", action="store_true")
    parser.add_argument("--iuphar-target", help="Path to _IUPHAR_target.csv")
    parser.add_argument("--iuphar-family", help="Path to _IUPHAR_family.csv")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Maximum number of IDs per network request",
    )
    return parser.parse_args()


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

    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    global_cache = default_cache or CacheConfig.from_dict(data.get("http_cache"))
    uni_cfg = data["uniprot"]
    fields = ",".join(uni_cfg.get("fields", []))
    uni_cache = CacheConfig.from_dict(uni_cfg.get("cache")) or global_cache
    uni = UniProtClient(
        base_url=uni_cfg["base_url"],
        fields=fields,
        network=UniNetworkConfig(
            timeout_sec=pipeline_cfg.timeout_sec,
            max_retries=pipeline_cfg.retries,
        ),
        rate_limit=UniRateConfig(rps=pipeline_cfg.rate_limit_rps),
        cache=uni_cache,
    )

    # The HGNC configuration is nested under the top-level "hgnc" section
    # in the YAML file. Explicitly select this section to avoid passing the
    # entire configuration dictionary to ``HGNCServiceConfig``, which would
    # otherwise raise ``TypeError`` due to unexpected keys.
    hcfg = load_hgnc_config(cfg_path, section="hgnc")
    hcfg.network.timeout_sec = pipeline_cfg.timeout_sec
    hcfg.network.max_retries = pipeline_cfg.retries
    hcfg.rate_limit.rps = pipeline_cfg.rate_limit_rps
    hcfg.cache = (
        hcfg.cache
        or CacheConfig.from_dict(data.get("hgnc", {}).get("cache"))
        or global_cache
    )
    hgnc = HGNCClient(hcfg)

    gcfg = GtoPConfig(
        base_url=data["gtop"]["base_url"],
        timeout_sec=pipeline_cfg.timeout_sec,
        max_retries=pipeline_cfg.retries,
        rps=pipeline_cfg.rate_limit_rps,
        cache=CacheConfig.from_dict(data.get("gtop", {}).get("cache")) or global_cache,
    )
    gtop = GtoPClient(gcfg)

    ens_client: EnsemblHomologyClient | None = None
    oma_client: OmaClient | None = None
    target_species: list[str] = []
    if with_orthologs:
        orth_cfg = data.get("orthologs", {})
        orth_cache = CacheConfig.from_dict(orth_cfg.get("cache")) or global_cache
        ens_client = EnsemblHomologyClient(
            base_url="https://rest.ensembl.org",
            network=UniNetworkConfig(
                timeout_sec=pipeline_cfg.timeout_sec,
                max_retries=pipeline_cfg.retries,
            ),
            rate_limit=UniRateConfig(rps=pipeline_cfg.rate_limit_rps),
            cache=orth_cache,
        )
        oma_client = OmaClient(
            base_url="https://omabrowser.org/api",
            network=UniNetworkConfig(
                timeout_sec=pipeline_cfg.timeout_sec,
                max_retries=pipeline_cfg.retries,
            ),
            rate_limit=UniRateConfig(rps=pipeline_cfg.rate_limit_rps),
            cache=orth_cache,
        )
        target_species = list(orth_cfg.get("target_species", []))
    return uni, hgnc, gtop, ens_client, oma_client, target_species


def save_output(
    df: pd.DataFrame,
    output: str | Path,
    *,
    sep: str = ",",
    encoding: str = "utf-8",
) -> Path:
    """Persist ``df`` to ``output`` ensuring the path is valid.

    The user-provided path may include a tilde (``~``) to reference the home
    directory or point to a location in a non-existent folder.  This helper
    expands user references and creates any missing parent directories before
    writing the CSV file.

    Parameters
    ----------
    df:
        Data frame to serialise.
    output:
        Destination file path.  ``"~"`` and ``".."`` segments are resolved.
    sep:
        Column delimiter for ``pandas.DataFrame.to_csv``.
    encoding:
        Text encoding for the resulting file.

    Returns
    -------
    pathlib.Path
        Absolute path to the written file.
    """

    out_path = Path(output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, sep=sep, encoding=encoding)
    analyze_table_quality(df, table_name=str(out_path.with_suffix("")))
    return out_path


def main() -> None:
    """Main entry point for the unified target data pipeline."""

    args = parse_args()
    logging.basicConfig(level=args.log_level.upper())
    pipeline_cfg = load_pipeline_config(args.config)
    pipeline_cfg.list_format = args.list_format
    pipeline_cfg.species_priority = [args.species]
    pipeline_cfg.iuphar.affinity_parameter = args.affinity_parameter
    pipeline_cfg.iuphar.approved_only = (
        None
        if args.approved_only.lower() == "null"
        else args.approved_only.lower() == "true"
    )
    pipeline_cfg.iuphar.primary_target_only = args.primary_target_only.lower() == "true"
    use_isoforms = args.with_isoforms
    pipeline_cfg.include_isoforms = False

    # Load optional ChEMBL column configuration and ensure required fields
    with open(args.config, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    global_cache = CacheConfig.from_dict(data.get("http_cache"))
    chembl_cols = list(data.get("chembl", {}).get("columns", []))
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
    chembl_cache = CacheConfig.from_dict(data.get("chembl", {}).get("cache"))
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
        for col in data.get(section, {}).get("columns", []):
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
        with_orthologs=args.with_orthologs,
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
    enrich_client = UniProtEnrichClient()
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

    out_df.to_csv(args.output, index=False, sep=args.sep, encoding=args.encoding)


if __name__ == "__main__":
    main()
