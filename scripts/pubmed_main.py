"""Command line entry point for collecting bibliographic metadata."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from itertools import chain
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Sequence, cast

import pandas as pd
import requests
import yaml
from tqdm.auto import tqdm

# Ensure imports resolve when the script is executed directly.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from library.chembl_client import ApiCfg, ChemblClient, get_documents
from library.cli_common import analyze_table_quality, resolve_cli_sidecar_paths
from library.document_pipeline import (
    CH_EMBL_COLUMNS,
    DOCUMENT_SCHEMA_COLUMNS,
    SEMANTIC_SCHOLAR_COLUMNS,
    DocumentsSchema,
    build_dataframe,
    compute_file_hash,
    dataframe_to_strings,
    merge_metadata,
    merge_with_chembl,
    quality_report,
    save_quality_report,
)
from library.http_client import HttpClient
from library.pubmed_client import PubMedRecord, fetch_pubmed_records
from library.semantic_scholar_client import (
    SemanticScholarRecord,
    fetch_semantic_scholar_records,
)
from library.openalex_client import OpenAlexRecord, fetch_openalex_records
from library.crossref_client import CrossrefRecord, fetch_crossref_records
from library.logging_utils import configure_logging

LOGGER = logging.getLogger("pubmed_main")
DEFAULT_LOG_FORMAT = "human"
DEFAULT_COMMAND = "all"

DEFAULT_CONFIG: Dict[str, Any] = {
    "io": {"sep": ",", "encoding": "utf-8"},
    "pubmed": {
        "batch_size": 100,
        "sleep": 0.34,
        "timeout_connect": 5.0,
        "timeout_read": 30.0,
        "max_retries": 5,
    },
    "semantic_scholar": {
        "chunk_size": 100,
        "timeout": 30.0,
        "max_retries": 6,
        "rps": 0.3,
        "high_throughput": False,
        "api_key_env": "SEMANTIC_SCHOLAR_API_KEY",
        "backoff_multiplier": 5.0,
        "retry_penalty_seconds": 30.0,
    },
    "openalex": {
        "rps": 1.0,
        "timeout": 30.0,
        "max_retries": 3,
    },
    "crossref": {
        "rps": 1.0,
        "timeout": 30.0,
        "max_retries": 3,
    },
    "chembl": {
        "chunk_size": 10,
        "timeout": 30.0,
        "max_retries": 3,
        "rps": 1.0,
    },
    "pipeline": {
        "workers": 1,
        "column_pubmed": "PMID",
        "column_chembl": "document_chembl_id",
        "column_crossref": "DOI",
        "status_forcelist": [408, 409, 429, 500, 502, 503, 504],
    },
}


SEMANTIC_SCHOLAR_PUBLIC_RPS = float(DEFAULT_CONFIG["semantic_scholar"]["rps"])


OPENALEX_ONLY_PLACEHOLDER_ERROR = "PubMed metadata not requested (OpenAlex-only run)"


CROSSREF_COLUMNS = [
    "crossref.DOI",
    "crossref.Type",
    "crossref.Subtype",
    "crossref.Title",
    "crossref.Subtitle",
    "crossref.Subject",
    "crossref.Error",
]


def _build_semantic_scholar_dataframe(
    records: Sequence[SemanticScholarRecord],
) -> pd.DataFrame:
    """Return a DataFrame containing Semantic Scholar metadata.

    Parameters
    ----------
    records:
        Sequence of parsed Semantic Scholar responses.

    Returns
    -------
    pandas.DataFrame
        Normalised table with deterministic column ordering.
    """

    if records:
        df = pd.DataFrame([record.to_dict() for record in records])
    else:
        df = pd.DataFrame(columns=SEMANTIC_SCHOLAR_COLUMNS)
    for column in SEMANTIC_SCHOLAR_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    df = df.reindex(columns=SEMANTIC_SCHOLAR_COLUMNS)
    return df


def _build_crossref_dataframe(records: Sequence[CrossrefRecord]) -> pd.DataFrame:
    """Return a DataFrame containing Crossref metadata only.

    Parameters
    ----------
    records:
        Sequence of parsed Crossref responses.

    Returns
    -------
    pandas.DataFrame
        Normalised table containing Crossref-only metadata columns.
    """

    if records:
        df = pd.DataFrame([record.to_dict() for record in records])
    else:
        df = pd.DataFrame(columns=CROSSREF_COLUMNS)
    for column in CROSSREF_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    df = df.reindex(columns=CROSSREF_COLUMNS)
    return df


def _deep_update(
    base: MutableMapping[str, Any], updates: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    for key, value in updates.items():
        if (
            key in base
            and isinstance(base[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _normalise_crossref_doi(doi: str | None) -> str | None:
    """Return a canonical DOI representation suitable for Crossref lookups."""

    if not doi:
        return None
    value = doi.strip()
    if not value:
        return None
    lowered = value.lower()
    prefixes = (
        "urn:doi:",
        "doi:",
        "https://doi.org/",
        "http://doi.org/",
    )
    for prefix in prefixes:
        if lowered.startswith(prefix):
            value = value[len(prefix) :]
            lowered = value.lower()
    cleaned = value.strip()
    return cleaned.lower() or None


def load_config(path: str | None) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path:
        config_path = Path(path)
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                file_cfg = yaml.safe_load(handle) or {}
            if not isinstance(file_cfg, Mapping):
                msg = f"Configuration in {path} is not a mapping"
                raise ValueError(msg)
            file_mapping = cast(Mapping[str, Any], file_cfg)
            _deep_update(config, file_mapping)
        else:
            LOGGER.warning("Config file %s not found; using defaults", path)
    return config


def _create_http_client(
    cfg: Mapping[str, Any], *, override_rps: float | None = None
) -> HttpClient:
    timeout = cfg.get("timeout")
    timeout_connect = float(
        cfg.get("timeout_connect", timeout if timeout is not None else 5.0)
    )
    timeout_read = float(
        cfg.get("timeout_read", timeout if timeout is not None else 30.0)
    )
    max_retries = int(cfg.get("max_retries", 3))
    rps = float(override_rps if override_rps is not None else cfg.get("rps", 0.0))
    status_forcelist = (
        cfg.get("status_forcelist") or DEFAULT_CONFIG["pipeline"]["status_forcelist"]
    )
    backoff_multiplier = float(cfg.get("backoff_multiplier", 1.0))
    retry_penalty_seconds = float(cfg.get("retry_penalty_seconds", 0.0))
    session: requests.Session | None = None
    api_key = cfg.get("api_key")
    if isinstance(api_key, str) and api_key.strip():
        session = requests.Session()
        session.headers.update({"x-api-key": api_key.strip()})
    return HttpClient(
        timeout=(timeout_connect, timeout_read),
        max_retries=max_retries,
        rps=rps,
        status_forcelist=status_forcelist,
        backoff_multiplier=backoff_multiplier,
        retry_penalty_seconds=retry_penalty_seconds,
        session=session,
    )


def _determine_output_path(input_path: Path, output: str | None, command: str) -> Path:
    if output:
        return Path(output)
    stem = input_path.stem
    date = datetime.now(UTC).strftime("%Y%m%d")
    return input_path.parent / f"output_{command}_{stem}_{date}.csv"


def _build_global_parser(
    default_config: Path, *, include_defaults: bool = True
) -> argparse.ArgumentParser:
    """Create a parser with arguments shared across commands.

    Parameters
    ----------
    default_config:
        Path to the default configuration file bundled with the project.
    include_defaults:
        Whether to assign default values to the shared arguments. Defaults are
        only applied when constructing the top-level parser to avoid subparsers
        overriding arguments provided before the command name.

    Returns
    -------
    argparse.ArgumentParser
        Parser instance containing global CLI arguments.
    """

    parser = argparse.ArgumentParser(add_help=False)
    default_config_value: Any = argparse.SUPPRESS
    if include_defaults:
        default_config_value = str(default_config)
    parser.add_argument(
        "--config",
        help="Path to YAML configuration file",
        default=default_config_value,
    )
    parser.add_argument(
        "--input",
        help="Input CSV path",
        default=(Path("input.csv") if include_defaults else argparse.SUPPRESS),
        type=Path,
    )
    parser.add_argument(
        "--output",
        help="Output CSV path",
        default=(None if include_defaults else argparse.SUPPRESS),
    )
    parser.add_argument(
        "--column",
        help="Name of the identifier column",
        default=(None if include_defaults else argparse.SUPPRESS),
    )
    parser.add_argument(
        "--sep",
        help="CSV separator",
        default=(None if include_defaults else argparse.SUPPRESS),
    )
    parser.add_argument(
        "--encoding",
        help="CSV encoding",
        default=(None if include_defaults else argparse.SUPPRESS),
    )
    parser.add_argument(
        "--log-level",
        default=("INFO" if include_defaults else argparse.SUPPRESS),
    )
    parser.add_argument(
        "--log-format",
        choices=("human", "json"),
        default=(DEFAULT_LOG_FORMAT if include_defaults else argparse.SUPPRESS),
        help="Logging output format (human or json)",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print effective configuration and exit",
        default=(False if include_defaults else argparse.SUPPRESS),
    )
    return parser


class ColumnNotFoundError(Exception):
    """Raised when the expected column is not present in the CSV file."""

    def __init__(self, column: str, available: Sequence[str]):
        super().__init__(f"Column '{column}' not found in input")
        self.column = column
        self.available = list(available)


def _detect_csv_separator(
    path: Path, *, encoding: str, sample_size: int = 65536
) -> str | None:
    """Detect the delimiter used in a CSV file.

    Parameters
    ----------
    path:
        CSV file path.
    encoding:
        Text encoding used to decode the file.
    sample_size:
        Maximum number of characters inspected when inferring the delimiter.

    Returns
    -------
    str | None
        The detected delimiter if inference succeeds, otherwise :data:`None`.
    """

    with path.open("r", encoding=encoding, newline="") as handle:
        sample = handle.read(sample_size)
    if not sample:
        return None
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample, delimiters=",\t;|")
    except csv.Error:
        return None
    return dialect.delimiter


def _read_csv_column(path: Path, column: str, *, sep: str, encoding: str) -> pd.Series:
    """Return a single column from a CSV file as a string series."""

    try:
        frame = pd.read_csv(
            path,
            sep=sep,
            encoding=encoding,
            dtype=str,
            usecols=[column],
        )
    except ValueError as error:
        if "Usecols do not match columns" in str(error):
            header = pd.read_csv(
                path,
                sep=sep,
                encoding=encoding,
                dtype=str,
                nrows=0,
            )
            raise ColumnNotFoundError(column, header.columns.tolist()) from error
        raise
    return frame[column].fillna("")


def _read_identifier_column(
    path: Path, column: str, *, sep: str, encoding: str
) -> List[str]:
    try:
        series = _read_csv_column(path, column, sep=sep, encoding=encoding)
    except (pd.errors.ParserError, ColumnNotFoundError) as error:
        detected_sep = _detect_csv_separator(path, encoding=encoding)
        if detected_sep and detected_sep != sep:
            LOGGER.warning(
                (
                    "Failed to parse %s with separator %r; retrying with detected "
                    "separator %r"
                ),
                path,
                sep,
                detected_sep,
            )
            try:
                series = _read_csv_column(
                    path, column, sep=detected_sep, encoding=encoding
                )
            except ColumnNotFoundError as retry_error:
                available = ", ".join(retry_error.available) or "<no columns>"
                msg = f"Column '{column}' not found in input. Available columns: {available}"
                raise SystemExit(msg) from retry_error
            except pd.errors.ParserError as retry_error:
                msg = (
                    f"Failed to parse '{path}' using detected separator {detected_sep!r}: "
                    f"{retry_error}"
                )
                raise SystemExit(msg) from retry_error
        else:
            if isinstance(error, ColumnNotFoundError):
                available = ", ".join(error.available) or "<no columns>"
                msg = f"Column '{column}' not found in input. Available columns: {available}"
                raise SystemExit(msg) from error
            msg = (
                f"Failed to parse '{path}' using separator {sep!r}: {error}. "
                "Provide --sep/--encoding options or update the configuration to match "
                "the input format."
            )
            raise SystemExit(msg) from error
    values = [str(value).strip() for value in series]
    return [value for value in values if value]


def _build_openalex_pubmed_placeholders(
    records: Sequence[OpenAlexRecord],
    *,
    error_message: str = OPENALEX_ONLY_PLACEHOLDER_ERROR,
) -> List[PubMedRecord]:
    """Create minimal PubMed records acting as OpenAlex merge anchors."""

    placeholders: List[PubMedRecord] = []
    for record in records:
        placeholders.append(
            PubMedRecord(
                pmid=record.pmid,
                doi=record.doi,
                title=None,
                abstract=None,
                journal=None,
                journal_abbrev=None,
                volume=None,
                issue=None,
                start_page=None,
                end_page=None,
                issn=None,
                publication_types=[],
                mesh_descriptors=[],
                mesh_qualifiers=[],
                chemical_list=[],
                year_completed=None,
                month_completed=None,
                day_completed=None,
                year_revised=None,
                month_revised=None,
                day_revised=None,
                error=error_message,
            )
        )
    return placeholders


def _gather_pubmed_sources(
    pmids: Sequence[str],
    *,
    cfg: Dict[str, Any],
) -> tuple[List[Any], List[Any], List[Any], List[Any]]:
    if not pmids:
        return [], [], [], []

    pubmed_cfg = cfg["pubmed"]
    sleep = float(pubmed_cfg.get("sleep", 0.0))
    rps = 0.0 if sleep <= 0 else 1.0 / sleep
    pubmed_client = _create_http_client(pubmed_cfg, override_rps=rps)
    pubmed_records = fetch_pubmed_records(
        pmids,
        client=pubmed_client,
        batch_size=int(pubmed_cfg.get("batch_size", 100)),
    )

    scholar_cfg = cfg["semantic_scholar"]
    scholar_client = _create_http_client(scholar_cfg)
    scholar_records = fetch_semantic_scholar_records(
        pmids,
        client=scholar_client,
        chunk_size=int(scholar_cfg.get("chunk_size", 100)),
    )

    openalex_cfg = cfg["openalex"]
    openalex_client = _create_http_client(openalex_cfg)
    openalex_records = fetch_openalex_records(pmids, client=openalex_client)

    doi_set: set[str] = set()

    def _collect_doi(candidate: str | None) -> None:
        normalised = _normalise_crossref_doi(candidate)
        if normalised:
            doi_set.add(normalised)

    for pubmed_record in pubmed_records:
        _collect_doi(pubmed_record.doi)
    for scholar_record in scholar_records:
        _collect_doi(scholar_record.doi)
    for openalex_record in openalex_records:
        _collect_doi(openalex_record.doi)
    unique_dois = sorted(doi_set)

    crossref_cfg = cfg["crossref"]
    crossref_client = _create_http_client(crossref_cfg)
    crossref_records = fetch_crossref_records(unique_dois, client=crossref_client)

    return pubmed_records, scholar_records, openalex_records, crossref_records


def _gather_openalex_sources(
    pmids: Sequence[str],
    *,
    cfg: Dict[str, Any],
) -> tuple[List[PubMedRecord], List[OpenAlexRecord], List[Any]]:
    """Collect OpenAlex metadata and Crossref enrichments for ``pmids``."""

    if not pmids:
        return [], [], []

    openalex_cfg = cfg["openalex"]
    openalex_client = _create_http_client(openalex_cfg)
    openalex_records = fetch_openalex_records(pmids, client=openalex_client)

    doi_set: set[str] = set()
    for record in openalex_records:
        normalised = _normalise_crossref_doi(record.doi)
        if normalised:
            doi_set.add(normalised)
    dois = sorted(doi_set)
    crossref_records: List[Any]
    if dois:
        crossref_cfg = cfg["crossref"]
        crossref_client = _create_http_client(crossref_cfg)
        crossref_records = fetch_crossref_records(dois, client=crossref_client)
    else:
        crossref_records = []

    placeholders = _build_openalex_pubmed_placeholders(openalex_records)
    return placeholders, openalex_records, crossref_records


class _QualityAccumulator:
    """Accumulate CSV quality metrics incrementally during streaming writes."""

    def __init__(self) -> None:
        self.row_count: int = 0
        self.doi_missing: int = 0
        self.publication_class_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}
        self._error_columns_seen: set[str] = set()

    def update(self, df: pd.DataFrame) -> None:
        """Update the aggregated metrics with ``df``.

        Args:
            df: A DataFrame chunk already converted to string columns.
        """

        chunk_size = len(df)
        self.row_count += chunk_size
        if chunk_size == 0:
            return

        doi_columns = [
            col
            for col in (
                "PubMed.DOI",
                "scholar.DOI",
                "OpenAlex.DOI",
                "crossref.DOI",
                "ChEMBL.doi",
            )
            if col in df.columns
        ]
        if doi_columns:
            doi_present = pd.Series(False, index=df.index, dtype=bool)
            for column in doi_columns:
                values = df[column].fillna("").astype(str).str.strip()
                doi_present = doi_present | values.ne("")
            self.doi_missing += int((~doi_present).sum())
        else:
            self.doi_missing += chunk_size

        if "publication_class" in df.columns:
            values = df["publication_class"].astype("string")
            counts = values.value_counts(dropna=False)
            for key, count in counts.items():
                text_key = str(key)
                self.publication_class_counts[text_key] = (
                    self.publication_class_counts.get(text_key, 0) + int(count)
                )

        error_columns = [col for col in df.columns if col.endswith(".Error")]
        for column in error_columns:
            self._error_columns_seen.add(column)
            lengths = df[column].fillna("").astype(str).str.len()
            count = int((lengths > 0).sum())
            self.error_counts[column] = self.error_counts.get(column, 0) + count

    def as_dict(self) -> Dict[str, Any]:
        """Return the accumulated metrics as a serialisable dictionary."""

        error_summary = {
            column: self.error_counts.get(column, 0)
            for column in sorted(self._error_columns_seen)
        }
        return {
            "row_count": self.row_count,
            "missing_doi": self.doi_missing,
            "publication_class_distribution": {
                key: int(value) for key, value in self.publication_class_counts.items()
            },
            "error_counts": error_summary,
        }


def _write_progress(progress_path: Path, processed_rows: int) -> None:
    """Persist the current number of processed rows to ``progress_path``."""

    progress_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"processed_rows": int(processed_rows)}
    progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _prepare_chembl_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Normalise a ChEMBL chunk to the exported column schema."""

    if chunk.empty:
        return pd.DataFrame(columns=CH_EMBL_COLUMNS)

    renamed = chunk.rename(
        columns={
            "document_chembl_id": "ChEMBL.document_chembl_id",
            "title": "ChEMBL.title",
            "abstract": "ChEMBL.abstract",
            "doi": "ChEMBL.doi",
            "year": "ChEMBL.year",
            "journal": "ChEMBL.journal",
            "journal_abbrev": "ChEMBL.journal_abbrev",
            "volume": "ChEMBL.volume",
            "issue": "ChEMBL.issue",
            "first_page": "ChEMBL.first_page",
            "last_page": "ChEMBL.last_page",
            "pubmed_id": "ChEMBL.pubmed_id",
            "authors": "ChEMBL.authors",
            "source": "ChEMBL.source",
        }
    )
    for column in CH_EMBL_COLUMNS:
        if column not in renamed.columns:
            renamed[column] = pd.NA
    return renamed.reindex(columns=CH_EMBL_COLUMNS)


def _write_chunked_chembl_output(
    chunks: Iterator[pd.DataFrame],
    *,
    output_path: Path,
    sep: str,
    encoding: str,
) -> Dict[str, Any]:
    """Stream ChEMBL chunks to ``output_path`` and return the metadata report."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.with_name(f"{output_path.name}.progress.json")
    accumulator = _QualityAccumulator()
    header_written = False
    total_rows = 0

    for raw_chunk in chunks:
        prepared = dataframe_to_strings(_prepare_chembl_chunk(raw_chunk))
        accumulator.update(prepared)
        if prepared.empty:
            continue
        mode = "w" if not header_written else "a"
        prepared.to_csv(
            output_path,
            sep=sep,
            index=False,
            mode=mode,
            header=not header_written,
            encoding=encoding,
        )
        header_written = True
        total_rows += len(prepared)
        _write_progress(progress_path, total_rows)

    if not header_written:
        empty = dataframe_to_strings(pd.DataFrame(columns=CH_EMBL_COLUMNS))
        empty.to_csv(output_path, sep=sep, index=False, encoding=encoding)
        _write_progress(progress_path, total_rows)

    checksum = compute_file_hash(output_path)
    report = accumulator.as_dict()
    report["file_sha256"] = checksum
    report["output"] = str(output_path)
    meta_path = output_path.with_name(f"{output_path.name}.meta.json")
    save_quality_report(meta_path, report)
    LOGGER.info("Wrote %d rows to %s", total_rows, output_path)
    LOGGER.info("Metadata report saved to %s", meta_path)
    LOGGER.info("Progress stored in %s", progress_path)
    return report


def _collect_document_chunks(chunks: Iterator[pd.DataFrame]) -> pd.DataFrame:
    """Materialise ``chunks`` into a single DataFrame while preserving order."""

    iterator = iter(chunks)
    try:
        first = next(iterator)
    except StopIteration:
        return pd.DataFrame()

    columns = list(first.columns)
    combined = pd.concat(chain((first,), iterator), ignore_index=True)
    return combined.reindex(columns=columns)


def _write_output(
    df: pd.DataFrame,
    *,
    output_path: Path,
    sep: str,
    encoding: str,
) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep=sep, index=False, encoding=encoding)
    checksum = compute_file_hash(output_path)
    report = cast(Dict[str, Any], quality_report(df))
    report["file_sha256"] = checksum
    report["output"] = str(output_path)
    meta_path = output_path.with_name(f"{output_path.name}.meta.json")
    save_quality_report(meta_path, report)
    LOGGER.info("Wrote %d rows to %s", len(df), output_path)
    LOGGER.info("Metadata report saved to %s", meta_path)
    return report


def run_crossref_command(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Writes Crossref metadata for the provided DOI values to disk.

    Args:
        args: Parsed command-line options.
        config: The effective configuration mapping with CLI overrides applied.
    """

    io_cfg = config["io"]
    column = args.column or config["pipeline"].get("column_crossref", "DOI")
    doi_values = _read_identifier_column(
        args.input, column, sep=io_cfg["sep"], encoding=io_cfg["encoding"]
    )
    LOGGER.info("Loaded %d DOI values", len(doi_values))

    # Normalise DOIs to the canonical lower-case format used for Crossref lookups
    # while preserving the first occurrence ordering for deterministic fetches.
    seen: set[str] = set()
    dois: List[str] = []
    for value in doi_values:
        normalised = _normalise_crossref_doi(value)
        if not normalised or normalised in seen:
            continue
        seen.add(normalised)
        dois.append(normalised)

    if not dois:
        LOGGER.warning("No valid DOI values found; writing empty Crossref output")

    crossref_cfg = config["crossref"]
    crossref_client = _create_http_client(crossref_cfg)
    records = fetch_crossref_records(dois, client=crossref_client)

    df = _build_crossref_dataframe(records)
    df = dataframe_to_strings(df)
    df = df.sort_values("crossref.DOI", na_position="last").reset_index(drop=True)
    _write_output(
        df, output_path=args.output, sep=io_cfg["sep"], encoding=io_cfg["encoding"]
    )


def run_openalex_command(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Writes OpenAlex and Crossref metadata for the provided PMIDs to disk.

    Args:
        args: Parsed command-line options.
        config: The effective configuration mapping with CLI overrides applied.
    """

    io_cfg = config["io"]
    column = args.column or config["pipeline"].get("column_pubmed", "PMID")
    pmids = _read_identifier_column(
        args.input, column, sep=io_cfg["sep"], encoding=io_cfg["encoding"]
    )
    LOGGER.info("Loaded %d unique PMIDs", len(pmids))
    (
        placeholder_pubmed,
        openalex_records,
        crossref_records,
    ) = _gather_openalex_sources(pmids, cfg=config)
    workers = int(args.workers or config["pipeline"].get("workers", 1))
    rows = merge_metadata(
        placeholder_pubmed,
        [],
        openalex_records,
        crossref_records,
        max_workers=workers,
    )
    df = build_dataframe(rows)
    schema = DocumentsSchema(DOCUMENT_SCHEMA_COLUMNS)
    errors = schema.validate(df)
    if errors:
        error_path = Path(str(args.output) + ".schema_errors.json")
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(
            json.dumps({"errors": errors}, indent=2), encoding="utf-8"
        )
        raise SystemExit("Schema validation failed; see error report")
    df = dataframe_to_strings(df)
    df = df.sort_values("PubMed.PMID", na_position="last").reset_index(drop=True)
    _write_output(
        df, output_path=args.output, sep=io_cfg["sep"], encoding=io_cfg["encoding"]
    )


def run_pubmed_command(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Runs the PubMed command, which fetches and merges data from PubMed and
    partner sources.

    Args:
        args: Parsed command-line options.
        config: The effective configuration mapping with CLI overrides applied.
    """
    io_cfg = config["io"]
    column = args.column or config["pipeline"].get("column_pubmed", "PMID")
    ids = _read_identifier_column(
        args.input, column, sep=io_cfg["sep"], encoding=io_cfg["encoding"]
    )
    LOGGER.info("Loaded %d unique PMIDs", len(ids))
    workers = int(args.workers or config["pipeline"].get("workers", 1))
    (
        pubmed_records,
        scholar_records,
        openalex_records,
        crossref_records,
    ) = _gather_pubmed_sources(ids, cfg=config)
    if pubmed_records:
        with tqdm(total=len(pubmed_records), desc="merge metadata") as pbar:
            rows = merge_metadata(
                pubmed_records,
                scholar_records,
                openalex_records,
                crossref_records,
                max_workers=workers,
                progress_callback=pbar.update,
            )
    else:
        rows = merge_metadata(
            pubmed_records,
            scholar_records,
            openalex_records,
            crossref_records,
            max_workers=workers,
        )
    df = build_dataframe(rows)
    schema = DocumentsSchema(DOCUMENT_SCHEMA_COLUMNS)
    errors = schema.validate(df)
    if errors:
        error_path = Path(str(args.output) + ".schema_errors.json")
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(
            json.dumps({"errors": errors}, indent=2), encoding="utf-8"
        )
        raise SystemExit("Schema validation failed; see error report")
    df = dataframe_to_strings(df)
    df = df.sort_values("PubMed.PMID", na_position="last").reset_index(drop=True)
    _write_output(
        df, output_path=args.output, sep=io_cfg["sep"], encoding=io_cfg["encoding"]
    )


def run_chembl_command(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Download ChEMBL document metadata and profile the resulting table.

    The command streams records from the ChEMBL API directly to ``args.output``
    to keep memory usage predictable. Once the export completes the function
    generates column completeness and numeric correlation reports using
    :func:`library.data_profiling.analyze_table_quality` to mirror the
    behaviour of other data extraction scripts in the project.

    Args:
        args: Parsed command-line options.
        config: The effective configuration mapping with CLI overrides applied.
    """
    io_cfg = config["io"]
    column = args.column or config["pipeline"].get(
        "column_chembl", "document_chembl_id"
    )
    ids = _read_identifier_column(
        args.input, column, sep=io_cfg["sep"], encoding=io_cfg["encoding"]
    )
    LOGGER.info("Loaded %d ChEMBL document IDs", len(ids))
    chem_cfg = config["chembl"]
    chem_client = ChemblClient(
        http_client=_create_http_client(
            chem_cfg,
            override_rps=float(chem_cfg.get("rps", 1.0)),
        )
    )
    cfg_obj = ApiCfg(timeout_read=float(chem_cfg.get("timeout", 30.0)))
    chunks = get_documents(
        ids,
        cfg=cfg_obj,
        client=chem_client,
        chunk_size=int(chem_cfg.get("chunk_size", 10)),
        timeout=float(chem_cfg.get("timeout", 30.0)),
    )
    _write_chunked_chembl_output(
        chunks,
        output_path=args.output,
        sep=io_cfg["sep"],
        encoding=io_cfg["encoding"],
    )
    _, _, quality_base = resolve_cli_sidecar_paths(args.output)
    analyze_table_quality(
        args.output,
        table_name=str(quality_base),
        separator=io_cfg["sep"],
        encoding=io_cfg["encoding"],
    )
    quality_report_path = quality_base.with_name(
        f"{quality_base.name}_quality_report_table.csv"
    )
    correlation_path = quality_base.with_name(
        f"{quality_base.name}_data_correlation_report_table.csv"
    )
    LOGGER.info("Quality report saved to %s", quality_report_path)
    LOGGER.info("Correlation report saved to %s", correlation_path)


def run_all_command(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Runs the 'all' command, which fetches ChEMBL documents and enriches them
    with data from the PubMed ecosystem.

    Args:
        args: Parsed command-line options.
        config: The effective configuration mapping with CLI overrides applied.
    """
    io_cfg = config["io"]
    column = args.column or config["pipeline"].get(
        "column_chembl", "document_chembl_id"
    )
    ids = _read_identifier_column(
        args.input, column, sep=io_cfg["sep"], encoding=io_cfg["encoding"]
    )
    LOGGER.info("Loaded %d ChEMBL document IDs", len(ids))
    chem_cfg = config["chembl"]
    chem_client = ChemblClient(
        http_client=_create_http_client(
            chem_cfg, override_rps=float(chem_cfg.get("rps", 1.0))
        )
    )
    cfg_obj = ApiCfg(timeout_read=float(chem_cfg.get("timeout", 30.0)))
    chem_chunks = get_documents(
        ids,
        cfg=cfg_obj,
        client=chem_client,
        chunk_size=int(chem_cfg.get("chunk_size", 10)),
        timeout=float(chem_cfg.get("timeout", 30.0)),
    )
    chem_df = _collect_document_chunks(chem_chunks)
    pmids = [
        str(value).strip()
        for value in chem_df.get("pubmed_id", pd.Series(dtype="string")).fillna("")
        if str(value).strip()
    ]
    pmids = sorted({pmid for pmid in pmids if pmid})
    LOGGER.info("Collected %d unique PubMed IDs from ChEMBL", len(pmids))
    workers = int(args.workers or config["pipeline"].get("workers", 1))
    (
        pubmed_records,
        scholar_records,
        openalex_records,
        crossref_records,
    ) = _gather_pubmed_sources(pmids, cfg=config)
    if pubmed_records:
        with tqdm(total=len(pubmed_records), desc="merge metadata") as pbar:
            rows = merge_metadata(
                pubmed_records,
                scholar_records,
                openalex_records,
                crossref_records,
                max_workers=workers,
                progress_callback=pbar.update,
            )
    else:
        rows = merge_metadata(
            pubmed_records,
            scholar_records,
            openalex_records,
            crossref_records,
            max_workers=workers,
        )
    df_metadata = build_dataframe(rows)
    df_metadata = dataframe_to_strings(df_metadata)
    df_metadata = df_metadata.sort_values(
        "PubMed.PMID", na_position="last"
    ).reset_index(drop=True)
    df = merge_with_chembl(df_metadata, chem_df)
    df = dataframe_to_strings(df)
    df = df.sort_values(
        ["ChEMBL.document_chembl_id", "PubMed.PMID"],
        na_position="last",
    ).reset_index(drop=True)
    _write_output(
        df, output_path=args.output, sep=io_cfg["sep"], encoding=io_cfg["encoding"]
    )


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser for the script.

    Returns:
        An `argparse.ArgumentParser` object.
    """
    default_config = (
        Path(__file__).resolve().parent.parent / "config" / "documents.yaml"
    )
    parser = argparse.ArgumentParser(
        description="Collect bibliographic metadata",
        parents=[_build_global_parser(default_config)],
    )
    common_parser = _build_global_parser(default_config, include_defaults=False)
    parser.add_argument(
        "--batch-size",
        dest="global_batch_size",
        type=int,
        default=None,
        help="Override the batch size for PubMed-related commands",
    )
    parser.add_argument(
        "--sleep",
        dest="global_sleep",
        type=float,
        default=None,
        help="Override sleep interval between PubMed requests",
    )
    parser.add_argument(
        "--workers",
        dest="global_workers",
        type=int,
        default=None,
        help="Override the worker count for metadata merging",
    )
    parser.add_argument(
        "--openalex-rps",
        dest="global_openalex_rps",
        type=float,
        default=None,
        help="Override the OpenAlex requests-per-second limit",
    )
    parser.add_argument(
        "--crossref-rps",
        dest="global_crossref_rps",
        type=float,
        default=None,
        help="Override the Crossref requests-per-second limit",
    )
    parser.add_argument(
        "--semantic-scholar-rps",
        dest="global_semantic_scholar_rps",
        type=float,
        default=None,
        help="Override the Semantic Scholar requests-per-second limit",
    )
    parser.add_argument(
        "--semantic-scholar-high-throughput",
        dest="global_semantic_scholar_high_throughput",
        action="store_true",
        help="Allow Semantic Scholar rates above the public throttle (requires API key)",
    )
    parser.add_argument(
        "--semantic-scholar-api-key",
        dest="global_semantic_scholar_api_key",
        default=None,
        help="Semantic Scholar API key used for authenticated requests",
    )
    parser.add_argument(
        "--chunk-size",
        dest="global_chunk_size",
        type=int,
        default=None,
        help="Override the chunk size for ChEMBL document downloads (must be positive)",
    )
    parser.add_argument(
        "--timeout",
        dest="global_timeout",
        type=float,
        default=None,
        help="Override the timeout for ChEMBL document downloads",
    )
    parser.add_argument(
        "--semantic-scholar-chunk-size",
        dest="global_semantic_scholar_chunk_size",
        type=int,
        default=None,
        help="Override the Semantic Scholar chunk size (must be positive)",
    )
    parser.add_argument(
        "--semantic-scholar-timeout",
        dest="global_semantic_scholar_timeout",
        type=float,
        default=None,
        help="Override the Semantic Scholar timeout",
    )
    subparsers = parser.add_subparsers(dest="command")
    parser.set_defaults(
        command=DEFAULT_COMMAND,
        batch_size=None,
        sleep=None,
        workers=None,
        openalex_rps=None,
        crossref_rps=None,
        chunk_size=None,
        timeout=None,
        semantic_scholar_rps=None,
        semantic_scholar_chunk_size=None,
        semantic_scholar_timeout=None,
        semantic_scholar_high_throughput=False,
        global_semantic_scholar_high_throughput=False,
        semantic_scholar_api_key=None,
        global_semantic_scholar_api_key=None,
    )

    pubmed_parser = subparsers.add_parser(
        "pubmed",
        help="Fetch PubMed and partner metadata",
        parents=[common_parser],
    )
    pubmed_parser.add_argument("--batch-size", type=int, default=None)
    pubmed_parser.add_argument("--sleep", type=float, default=None)
    pubmed_parser.add_argument("--workers", type=int, default=None)
    pubmed_parser.add_argument("--openalex-rps", type=float, default=None)
    pubmed_parser.add_argument("--crossref-rps", type=float, default=None)
    pubmed_parser.add_argument("--semantic-scholar-rps", type=float, default=None)
    pubmed_parser.add_argument(
        "--semantic-scholar-chunk-size",
        type=int,
        default=None,
        help="Override the Semantic Scholar chunk size (must be positive)",
    )
    pubmed_parser.add_argument("--semantic-scholar-timeout", type=float, default=None)
    pubmed_parser.add_argument(
        "--semantic-scholar-high-throughput",
        action="store_true",
        help="Allow Semantic Scholar rates above the public throttle (requires API key)",
    )
    pubmed_parser.add_argument(
        "--semantic-scholar-api-key",
        default=None,
        help="Semantic Scholar API key used for authenticated requests",
    )

    openalex_parser = subparsers.add_parser(
        "openalex",
        help="Fetch OpenAlex metadata with Crossref enrichment",
        parents=[common_parser],
    )
    openalex_parser.add_argument("--workers", type=int, default=None)
    openalex_parser.add_argument("--openalex-rps", type=float, default=None)
    openalex_parser.add_argument("--crossref-rps", type=float, default=None)

    crossref_parser = subparsers.add_parser(
        "crossref",
        help="Fetch Crossref metadata only",
        parents=[common_parser],
    )
    crossref_parser.add_argument("--crossref-rps", type=float, default=None)

    chembl_parser = subparsers.add_parser(
        "chembl",
        help="Download ChEMBL document metadata",
        parents=[common_parser],
    )
    chembl_parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override the chunk size for ChEMBL document downloads (must be positive)",
    )
    chembl_parser.add_argument("--timeout", type=float, default=None)

    scholar_parser = subparsers.add_parser(
        "scholar",
        help="Fetch Semantic Scholar metadata",
        parents=[common_parser],
    )
    scholar_parser.add_argument(
        "--semantic-scholar-chunk-size",
        type=int,
        default=None,
        help="Override the Semantic Scholar chunk size (must be positive)",
    )
    scholar_parser.add_argument("--semantic-scholar-rps", type=float, default=None)
    scholar_parser.add_argument("--semantic-scholar-timeout", type=float, default=None)
    scholar_parser.add_argument(
        "--semantic-scholar-high-throughput",
        action="store_true",
        help="Allow Semantic Scholar rates above the public throttle (requires API key)",
    )
    scholar_parser.add_argument(
        "--semantic-scholar-api-key",
        default=None,
        help="Semantic Scholar API key used for authenticated requests",
    )

    all_parser = subparsers.add_parser(
        "all",
        help="Fetch ChEMBL documents and enrich with PubMed ecosystems",
        parents=[common_parser],
    )
    all_parser.add_argument("--batch-size", type=int, default=None)
    all_parser.add_argument("--sleep", type=float, default=None)
    all_parser.add_argument("--workers", type=int, default=None)
    all_parser.add_argument("--openalex-rps", type=float, default=None)
    all_parser.add_argument("--crossref-rps", type=float, default=None)
    all_parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override the chunk size for ChEMBL document downloads (must be positive)",
    )
    all_parser.add_argument("--timeout", type=float, default=None)
    all_parser.add_argument("--semantic-scholar-rps", type=float, default=None)
    all_parser.add_argument(
        "--semantic-scholar-chunk-size",
        type=int,
        default=None,
        help="Override the Semantic Scholar chunk size (must be positive)",
    )
    all_parser.add_argument("--semantic-scholar-timeout", type=float, default=None)
    all_parser.add_argument(
        "--semantic-scholar-high-throughput",
        action="store_true",
        help="Allow Semantic Scholar rates above the public throttle (requires API key)",
    )
    all_parser.add_argument(
        "--semantic-scholar-api-key",
        default=None,
        help="Semantic Scholar API key used for authenticated requests",
    )

    return parser


def _ensure_positive_chunk_size(
    parser: argparse.ArgumentParser,
    namespace: argparse.Namespace,
    attribute: str,
    option_string: str,
) -> None:
    """Validate that ``attribute`` on ``namespace`` is a positive integer."""

    value = getattr(namespace, attribute, None)
    if value is not None and value <= 0:
        parser.error(f"{option_string} must be a positive integer")


def _validate_chunk_size_options(
    parser: argparse.ArgumentParser, namespace: argparse.Namespace
) -> None:
    """Validate all chunk-size related CLI options."""

    _ensure_positive_chunk_size(parser, namespace, "global_chunk_size", "--chunk-size")
    _ensure_positive_chunk_size(parser, namespace, "chunk_size", "--chunk-size")
    _ensure_positive_chunk_size(
        parser,
        namespace,
        "global_semantic_scholar_chunk_size",
        "--semantic-scholar-chunk-size",
    )
    _ensure_positive_chunk_size(
        parser,
        namespace,
        "semantic_scholar_chunk_size",
        "--semantic-scholar-chunk-size",
    )


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments and validate chunk-size options."""

    parser = build_parser()
    parsed_args = parser.parse_args(args)
    _validate_chunk_size_options(parser, parsed_args)
    return parsed_args


def _cli_option(args: argparse.Namespace, *names: str) -> Any | None:
    """Return the first configured CLI option from the provided attribute names."""

    for name in names:
        if hasattr(args, name):
            value = getattr(args, name)
            if value is not None:
                return value
    return None


def _cli_flag(args: argparse.Namespace, name: str, global_name: str) -> bool:
    """Return ``True`` when either the command-specific or global flag is set."""

    return bool(getattr(args, name, False) or getattr(args, global_name, False))


def _resolve_semantic_scholar_api_key(config: Mapping[str, Any]) -> str | None:
    """Resolve the Semantic Scholar API key from the configuration or environment."""

    api_key = config.get("api_key")
    if isinstance(api_key, str) and api_key.strip():
        return api_key.strip()
    env_name = config.get("api_key_env")
    if isinstance(env_name, str) and env_name.strip():
        env_value = os.environ.get(env_name.strip())
        if isinstance(env_value, str) and env_value.strip():
            return env_value.strip()
    return None


def _enforce_semantic_scholar_rate_limit(config: MutableMapping[str, Any]) -> None:
    """Clamp Semantic Scholar throughput unless high-throughput mode is enabled."""

    requested_rps = float(config.get("rps", SEMANTIC_SCHOLAR_PUBLIC_RPS))
    high_throughput = bool(config.get("high_throughput"))
    api_key = _resolve_semantic_scholar_api_key(config)

    if high_throughput:
        if requested_rps > SEMANTIC_SCHOLAR_PUBLIC_RPS and not api_key:
            msg = (
                "Semantic Scholar high-throughput mode requires an API key. "
                "Provide --semantic-scholar-api-key or set the"
                f" {config.get('api_key_env', 'SEMANTIC_SCHOLAR_API_KEY')} environment variable."
            )
            raise ValueError(msg)
        if api_key:
            config["api_key"] = api_key
        return

    if requested_rps > SEMANTIC_SCHOLAR_PUBLIC_RPS:
        LOGGER.warning(
            "Semantic Scholar RPS %.2f exceeds the public limit; falling back to %.2f. "
            "Use --semantic-scholar-high-throughput with an API key to override.",
            requested_rps,
            SEMANTIC_SCHOLAR_PUBLIC_RPS,
        )
        config["rps"] = SEMANTIC_SCHOLAR_PUBLIC_RPS
    if api_key:
        config["api_key"] = api_key


def _redact_sensitive_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of ``config`` with sensitive values redacted."""

    redacted = deepcopy(config)
    semantic = redacted.get("semantic_scholar")
    if isinstance(semantic, MutableMapping) and semantic.get("api_key"):
        semantic["api_key"] = "***"
    return redacted


def apply_cli_overrides(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Applies command-line argument overrides to the configuration.

    Args:
        args: The parsed command-line arguments.
        config: The configuration dictionary to update.
    """
    if args.sep:
        config["io"]["sep"] = args.sep
    if args.encoding:
        config["io"]["encoding"] = args.encoding

    scholar_cfg_raw = config.get("semantic_scholar")
    scholar_cfg = (
        cast(MutableMapping[str, Any], scholar_cfg_raw)
        if isinstance(scholar_cfg_raw, MutableMapping)
        else None
    )
    if args.command in {"pubmed", "all"}:
        batch_size = _cli_option(args, "batch_size", "global_batch_size")
        if batch_size is not None:
            config["pubmed"]["batch_size"] = int(batch_size)
        sleep = _cli_option(args, "sleep", "global_sleep")
        if sleep is not None:
            config["pubmed"]["sleep"] = float(sleep)
    if args.command in {"pubmed", "all", "openalex"}:
        openalex_rps = _cli_option(args, "openalex_rps", "global_openalex_rps")
        if openalex_rps is not None:
            config["openalex"]["rps"] = float(openalex_rps)
        workers = _cli_option(args, "workers", "global_workers")
        if workers is not None:
            config["pipeline"]["workers"] = int(workers)
    if args.command in {"pubmed", "all", "openalex", "crossref"}:
        crossref_rps = _cli_option(args, "crossref_rps", "global_crossref_rps")
        if crossref_rps is not None:
            config["crossref"]["rps"] = float(crossref_rps)
    if scholar_cfg is not None and args.command in {"pubmed", "all", "scholar"}:
        scholar_rps = _cli_option(
            args,
            "semantic_scholar_rps",
            "global_semantic_scholar_rps",
        )
        if scholar_rps is not None:
            scholar_cfg["rps"] = float(scholar_rps)
        scholar_chunk_size = _cli_option(
            args,
            "semantic_scholar_chunk_size",
            "global_semantic_scholar_chunk_size",
        )
        if scholar_chunk_size is not None:
            scholar_cfg["chunk_size"] = int(scholar_chunk_size)
        scholar_timeout = _cli_option(
            args,
            "semantic_scholar_timeout",
            "global_semantic_scholar_timeout",
        )
        if scholar_timeout is not None:
            scholar_cfg["timeout"] = float(scholar_timeout)
        if _cli_flag(
            args,
            "semantic_scholar_high_throughput",
            "global_semantic_scholar_high_throughput",
        ):
            scholar_cfg["high_throughput"] = True
        scholar_api_key = _cli_option(
            args,
            "semantic_scholar_api_key",
            "global_semantic_scholar_api_key",
        )
        if scholar_api_key is not None:
            scholar_cfg["api_key"] = str(scholar_api_key)
    if scholar_cfg is not None:
        _enforce_semantic_scholar_rate_limit(scholar_cfg)
    if args.command in {"chembl", "all"}:
        chunk_size = _cli_option(args, "chunk_size", "global_chunk_size")
        if chunk_size is not None:
            config["chembl"]["chunk_size"] = int(chunk_size)
        timeout = _cli_option(args, "timeout", "global_timeout")
        if timeout is not None:
            config["chembl"]["timeout"] = float(timeout)


def run_semantic_scholar_command(
    args: argparse.Namespace, config: Dict[str, Any]
) -> None:
    """Writes Semantic Scholar metadata for the provided PMIDs to disk.

    Args:
        args: Parsed command-line options.
        config: The effective configuration mapping with CLI overrides applied.
    """

    io_cfg = config["io"]
    column = args.column or config["pipeline"].get("column_pubmed", "PMID")
    ids = _read_identifier_column(
        args.input, column, sep=io_cfg["sep"], encoding=io_cfg["encoding"]
    )
    LOGGER.info("Loaded %d unique PMIDs", len(ids))
    scholar_cfg = config["semantic_scholar"]
    scholar_client = _create_http_client(scholar_cfg)
    records = fetch_semantic_scholar_records(
        ids,
        client=scholar_client,
        chunk_size=int(scholar_cfg.get("chunk_size", 100)),
    )
    df = _build_semantic_scholar_dataframe(records)
    df = dataframe_to_strings(df)
    df = df.sort_values("scholar.PMID").reset_index(drop=True)
    _write_output(
        df, output_path=args.output, sep=io_cfg["sep"], encoding=io_cfg["encoding"]
    )


def main() -> None:
    """The main entry point for the script."""
    args = parse_args()
    configure_logging(args.log_level, log_format=args.log_format)
    config = load_config(args.config)
    apply_cli_overrides(args, config)

    if args.print_config:
        print(json.dumps(_redact_sensitive_config(config), indent=2, sort_keys=True))
        return

    input_path: Path = args.input
    output_path = _determine_output_path(input_path, args.output, args.command)
    args.output = output_path

    if args.command == "pubmed":
        run_pubmed_command(args, config)
    elif args.command == "openalex":
        run_openalex_command(args, config)
    elif args.command == "crossref":
        run_crossref_command(args, config)
    elif args.command == "chembl":
        run_chembl_command(args, config)
    elif args.command == "scholar":
        run_semantic_scholar_command(args, config)
    elif args.command == "all":
        run_all_command(args, config)
    else:  # pragma: no cover - safeguarded by argparse
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
