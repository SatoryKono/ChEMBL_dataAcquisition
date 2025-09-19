"""Retrieve and normalise UniProtKB target information.

Examples
--------
>>> python scripts/get_uniprot_target_data.py --input data.csv --output out.csv
"""

from __future__ import annotations

import argparse
import logging
import sys

 
from itertools import chain
 
from collections.abc import Iterator, Mapping, Sequence
 

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import yaml

if __package__ in {None, ""}:
    from _path_utils import ensure_project_root as _ensure_project_root

    _ensure_project_root()

from pipeline_targets_main import _resolve_uniprot_fields

from library.config.uniprot import ConfigError, load_uniprot_target_config

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from library.orthologs import EnsemblHomologyClient, OmaClient
    from library.uniprot_normalize import Isoform

DEFAULT_INPUT = "input.csv"
DEFAULT_OUTPUT = "output_input_{date}.csv"
DEFAULT_COLUMN = "uniprot_id"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_SEP = ","
DEFAULT_ENCODING = "utf-8"

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
BATCH_SIZE = 100


def _ensure_mapping(value: Any, *, section: str) -> dict[str, Any]:
    """Return ``value`` as a mapping or raise ``ValueError``.

    This helper is retained for backward compatibility with existing tests
    that monkeypatch the function to inject synthetic configuration data.

    Args:
        value: Configuration value to validate.
        section: Human-readable section name for error messages.

    Returns:
        dict[str, Any]: Shallow copy of ``value`` when it is mapping-like.

    Raises:
        ValueError: If ``value`` is not a mapping.
    """

    if isinstance(value, Mapping):
        return dict(value)
    msg = f"Expected '{section}' section to be a mapping"
    raise ValueError(msg)


def _default_output(input_path: Path) -> Path:
    """Return the default output location for ``input_path``.

    Args:
        input_path: Source CSV path supplied by the user.

    Returns:
        Path: Location sharing the directory of ``input_path`` and the
        ``DEFAULT_OUTPUT`` name pattern containing today's date. This avoids
        overwriting previous exports when ``--output`` is omitted.
    """

    date = datetime.now().strftime("%Y%m%d")
    return input_path.with_name(DEFAULT_OUTPUT.format(date=date))


def _batched(values: Sequence[str], size: int) -> Iterator[list[str]]:
    """Yield ``values`` in contiguous lists of at most ``size`` items."""

    if size <= 0:
        msg = "Batch size must be a positive integer"
        raise ValueError(msg)
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def main(argv: Sequence[str] | None = None) -> None:
    """Runs the UniProt target data retrieval workflow.

    Args:
        argv: An optional sequence of command-line arguments. If None, the
            arguments are taken from `sys.argv`.
    """

    from library.cli_common import (
        analyze_table_quality,
        ensure_output_dir,
        resolve_cli_sidecar_paths,
        serialise_dataframe,
        write_cli_metadata,
    )
    from library.http_client import CacheConfig
    from library.io import read_ids
    from library.io_utils import CsvConfig, write_rows
    from library.logging_utils import configure_logging
    from library.orthologs import EnsemblHomologyClient, OmaClient
    from library.uniprot_client import (
        NetworkConfig,
        RateLimitConfig,
        UniProtClient,
    )
    from library.uniprot_normalize import (
        extract_ensembl_gene_ids,
        extract_isoforms,
        normalize_entry,
        output_columns,
    )

    parser = argparse.ArgumentParser(description="Fetch UniProt target data")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV path")
    parser.add_argument("--output", help="Output CSV path")
    parser.add_argument(
        "--column",
        default=DEFAULT_COLUMN,
        help=(
            "Column containing UniProt accessions (default: 'uniprot_id'; must be"
            " present in the input file)"
        ),
    )
    parser.add_argument("--sep", help="CSV separator, e.g. ','")
    parser.add_argument("--encoding", help="File encoding, e.g. 'utf-8'")
    parser.add_argument(
        "--include-sequence",
        action="store_true",
        help="Include full protein sequence",
    )
    parser.add_argument(
        "--with-isoforms",
        action="store_true",
        help="Fetch and export isoform information",
    )
    parser.add_argument(
        "--isoforms-output",
        help="Path to write normalised isoform table",
    )
    parser.add_argument(
        "--with-orthologs", action="store_true", help="Retrieve ortholog information"
    )
    parser.add_argument(
        "--orthologs-output", help="Path to write normalised ortholog table"
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        help="Logging level (INFO, DEBUG, ... )",
    )
    parser.add_argument(
        "--log-format",
        choices=("human", "json"),
        default="human",
        help="Logging output format (human or json)",
    )
    args = parser.parse_args(argv)

    configure_logging(args.log_level, log_format=args.log_format)

    command_parts = (
        tuple(sys.argv)
        if argv is None
        else ("get_uniprot_target_data.py", *tuple(argv))
    )

    config_path = ROOT / "config.yaml"
    raw_config = yaml.safe_load(config_path.read_text())
    _ensure_mapping(raw_config, section="root configuration")

    try:
        cfg = load_uniprot_target_config(config_path)
    except ConfigError as exc:  # pragma: no cover - defensive logging
        LOGGER.error("%s", exc)
        raise

    http_cache_mapping = cfg.http_cache.to_cache_dict() if cfg.http_cache else None
    global_cache = CacheConfig.from_dict(http_cache_mapping)

    list_format = cfg.output.list_format
    include_seq = bool(args.include_sequence or cfg.output.include_sequence)
    sep_value = args.sep or cfg.output.sep or DEFAULT_SEP
    encoding_value = args.encoding or cfg.output.encoding or DEFAULT_ENCODING
    include_iso = bool(args.with_isoforms or cfg.uniprot.include_isoforms)
    use_fasta_stream = bool(cfg.uniprot.use_fasta_stream_for_isoform_ids)

    # Use configuration defaults when CLI options are omitted
    csv_cfg = CsvConfig(sep=sep_value, encoding=encoding_value, list_format=list_format)

    uniprot_cache = (
        CacheConfig.from_dict(cfg.uniprot.cache.to_cache_dict())
        if cfg.uniprot.cache
        else None
    )
    client = UniProtClient(
        base_url=cfg.uniprot.base_url,
        fields=_resolve_uniprot_fields(cfg.uniprot.model_dump()),
        network=NetworkConfig(
            timeout_sec=cfg.uniprot.timeout_sec,
            max_retries=cfg.uniprot.retries,
            backoff_sec=1.0,
        ),
        rate_limit=RateLimitConfig(rps=cfg.uniprot.rps),
        cache=uniprot_cache or global_cache,
    )

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else _default_output(input_path)

    orthologs_path = (
        Path(args.orthologs_output)
        if args.orthologs_output
        else output_path.with_name(output_path.stem + "_orthologs.csv")
    )

    iso_out_path = (
        Path(args.isoforms_output)
        if args.isoforms_output
        else output_path.with_name(f"{output_path.stem}_isoforms.csv")
    )

    output_path = ensure_output_dir(output_path)
    meta_path, _, quality_base = resolve_cli_sidecar_paths(output_path)
    if args.with_orthologs and cfg.orthologs.enabled:
        orthologs_path = ensure_output_dir(orthologs_path)
    if include_iso:
        iso_out_path = ensure_output_dir(iso_out_path)

    command_parts = (
        tuple(sys.argv)
        if argv is None
        else ("get_uniprot_target_data.py", *tuple(argv))
    )

    def _report_missing_column() -> None:
        message = (
            f"Input file {input_path} does not contain the required "
            f"'{args.column}' column"
        )
        LOGGER.error(message)
        write_cli_metadata(
            output_path,
            row_count=0,
            column_count=0,
            namespace=args,
            command_parts=command_parts,
            meta_path=meta_path,
            status="error",
            error=message,
        )
        raise SystemExit(1) from None

    raw_accessions = read_ids(input_path, args.column, csv_cfg)
    try:
        first_accession = next(raw_accessions)
    except StopIteration:
        accessions = iter(())
    except KeyError:
        _report_missing_column()
    else:
        accessions = chain((first_accession,), raw_accessions)
    rows: list[dict[str, Any]] = []
    iso_rows: list[dict[str, Any]] = []

    cols = output_columns(include_seq)

    target_species: list[str] = []
    ensembl_client: EnsemblHomologyClient | None = None
    oma_client: OmaClient | None = None
    orth_cols: list[str] = []
    orth_rows: list[dict[str, Any]] = []

    if args.with_orthologs and cfg.orthologs.enabled:
        orth_cache_mapping = (
            cfg.orthologs.cache.to_cache_dict() if cfg.orthologs.cache else None
        )
        orth_cache = CacheConfig.from_dict(orth_cache_mapping) or global_cache
        ensembl_client = EnsemblHomologyClient(
            base_url="https://rest.ensembl.org",
            network=NetworkConfig(
                timeout_sec=cfg.orthologs.timeout_sec,
                max_retries=cfg.orthologs.retries,
                backoff_sec=cfg.orthologs.backoff_base_sec,
            ),
            rate_limit=RateLimitConfig(rps=cfg.orthologs.rate_limit_rps),
            cache=orth_cache,
        )
        oma_client = OmaClient(
            base_url="https://omabrowser.org/api",
            network=NetworkConfig(
                timeout_sec=cfg.orthologs.timeout_sec,
                max_retries=cfg.orthologs.retries,
                backoff_sec=cfg.orthologs.backoff_base_sec,
            ),
            rate_limit=RateLimitConfig(rps=cfg.orthologs.rate_limit_rps),
            cache=orth_cache,
        )
        target_species = list(cfg.orthologs.target_species)
        orth_cols = [
            "source_uniprot_id",
            "source_ensembl_gene_id",
            "source_species",
            "target_species",
            "target_gene_symbol",
            "target_ensembl_gene_id",
            "target_uniprot_id",
            "homology_type",
            "perc_id",
            "perc_pos",
            "dn",
            "ds",
            "is_high_confidence",
            "source_db",
        ]

    for batch in _batched(accessions, BATCH_SIZE):
        batch_entries = client.fetch_entries_json(batch, batch_size=BATCH_SIZE)
        for acc in batch:
            data = batch_entries.get(acc)
            if data is None and "-" in acc:
                canonical_acc = acc.split("-", 1)[0]
                data = batch_entries.get(canonical_acc)
            if data is None:
                LOGGER.warning(
                    "No UniProt entry available for %s",
                    acc,
                    extra={"uniprot_id": acc},
                )
                row = {c: "" for c in cols}
                row["uniprot_id"] = acc
                if ensembl_client:
                    row["orthologs_json"] = []
                    row["orthologs_count"] = 0
                rows.append(row)
                continue

            gene_ids = extract_ensembl_gene_ids(data)

            isoforms: list[Isoform] = []
            if include_iso:
                entry = data
                fasta_headers: list[str] = []
                if use_fasta_stream:
                    fasta_headers = client.fetch_isoforms_fasta(acc)
                isoforms = extract_isoforms(entry, fasta_headers)
                for iso in isoforms:
                    iso_rows.append(
                        {
                            "parent_uniprot_id": acc,
                            "isoform_uniprot_id": iso["isoform_uniprot_id"],
                            "isoform_name": iso["isoform_name"],
                            "isoform_synonyms": list(iso["isoform_synonyms"]),
                            "is_canonical": iso["is_canonical"],
                        }
                    )

            row = normalize_entry(data, include_seq, isoforms)

            orthologs_json: list[dict[str, Any]] = []
            orthologs_count = 0
            if ensembl_client and gene_ids:
                gene_id = gene_ids[0]
                orthologs = ensembl_client.get_orthologs(gene_id, target_species)
                if not orthologs and oma_client:
                    orthologs = oma_client.get_orthologs_by_uniprot(acc)
                orthologs_json = [o.to_ordered_dict() for o in orthologs]
                orthologs_count = len(orthologs)
                for o in orthologs:
                    orth_rows.append(
                        {
                            "source_uniprot_id": acc,
                            "source_ensembl_gene_id": gene_id,
                            "source_species": row.get("organism_name", ""),
                            "target_species": o.target_species,
                            "target_gene_symbol": o.target_gene_symbol,
                            "target_ensembl_gene_id": o.target_ensembl_gene_id,
                            "target_uniprot_id": o.target_uniprot_id or "",
                            "homology_type": o.homology_type or "",
                            "perc_id": o.perc_id,
                            "perc_pos": o.perc_pos,
                            "dn": o.dn,
                            "ds": o.ds,
                            "is_high_confidence": o.is_high_confidence,
                            "source_db": o.source_db,
                        }
                    )
            elif ensembl_client:
                LOGGER.warning(
                    "Missing Ensembl gene identifier for %s",
                    acc,
                    extra={"uniprot_id": acc},
                )
                if oma_client:
                    orthologs = oma_client.get_orthologs_by_uniprot(acc)
                    orthologs_json = [o.to_ordered_dict() for o in orthologs]
                    orthologs_count = len(orthologs)
                    for o in orthologs:
                        orth_rows.append(
                            {
                                "source_uniprot_id": acc,
                                "source_ensembl_gene_id": "",
                                "source_species": row.get("organism_name", ""),
                                "target_species": o.target_species,
                                "target_gene_symbol": o.target_gene_symbol,
                                "target_ensembl_gene_id": o.target_ensembl_gene_id,
                                "target_uniprot_id": o.target_uniprot_id or "",
                                "homology_type": o.homology_type or "",
                                "perc_id": o.perc_id,
                                "perc_pos": o.perc_pos,
                                "dn": o.dn,
                                "ds": o.ds,
                                "is_high_confidence": o.is_high_confidence,
                                "source_db": o.source_db,
                            }
                        )
            if ensembl_client:
                row["orthologs_json"] = orthologs_json
                row["orthologs_count"] = orthologs_count

            rows.append(row)

    if ensembl_client:
        cols.extend(["orthologs_json", "orthologs_count"])

    if ensembl_client and orth_rows:
        orth_rows.sort(
            key=lambda x: (
                x["source_uniprot_id"],
                x["target_species"],
                x["target_gene_symbol"],
            )
        )
        write_rows(orthologs_path, orth_rows, orth_cols, csv_cfg)
        orth_df = pd.DataFrame(orth_rows, columns=orth_cols)
        serialised_orth_df = serialise_dataframe(
            orth_df, list_format=csv_cfg.list_format
        )
        orth_meta_path, _, orth_quality_base = resolve_cli_sidecar_paths(
            orthologs_path
        )
        analyze_table_quality(serialised_orth_df, table_name=str(orth_quality_base))
        write_cli_metadata(
            orthologs_path,
            row_count=int(serialised_orth_df.shape[0]),
            column_count=int(serialised_orth_df.shape[1]),
            namespace=args,
            command_parts=command_parts,
            meta_path=orth_meta_path,
        )
        LOGGER.info(
            "Ortholog table written to %s",
            orthologs_path,
            extra={"path": str(orthologs_path)},
        )

    rows.sort(key=lambda r: r.get("uniprot_id", ""))
    output_df = pd.DataFrame(rows, columns=cols)
    serialised_df = serialise_dataframe(
        output_df, list_format=csv_cfg.list_format, inplace=True
    )
    write_rows(output_path, rows, cols, csv_cfg)
    if include_iso:
        iso_cols = [
            "parent_uniprot_id",
            "isoform_uniprot_id",
            "isoform_name",
            "isoform_synonyms",
            "is_canonical",
        ]
        iso_rows.sort(
            key=lambda r: (
                r["parent_uniprot_id"],
                (
                    int(r["isoform_uniprot_id"].split("-")[-1])
                    if r["isoform_uniprot_id"].count("-")
                    else 999999
                ),
            )
        )
        write_rows(iso_out_path, iso_rows, iso_cols, csv_cfg)
        iso_df = pd.DataFrame(iso_rows, columns=iso_cols)
        serialised_iso_df = serialise_dataframe(
            iso_df, list_format=csv_cfg.list_format
        )
        iso_meta_path, _, iso_quality_base = resolve_cli_sidecar_paths(iso_out_path)
        analyze_table_quality(serialised_iso_df, table_name=str(iso_quality_base))
        write_cli_metadata(
            iso_out_path,
            row_count=int(serialised_iso_df.shape[0]),
            column_count=int(serialised_iso_df.shape[1]),
            namespace=args,
            command_parts=command_parts,
            meta_path=iso_meta_path,
        )

    analyze_table_quality(serialised_df, table_name=str(quality_base))

    write_cli_metadata(
        output_path,
        row_count=int(serialised_df.shape[0]),
        column_count=int(serialised_df.shape[1]),
        namespace=args,
        command_parts=command_parts,
        meta_path=meta_path,
    )

    LOGGER.info(
        "Target table written to %s",
        output_path,
        extra={"path": str(output_path)},
    )


if __name__ == "__main__":  # pragma: no cover
    main()
