"""Retrieve and normalise UniProtKB target information.

Examples
--------
>>> python scripts/get_uniprot_target_data.py --input data.csv --output out.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import yaml

if __package__ in {None, ""}:
    from _path_utils import ensure_project_root as _ensure_project_root

    _ensure_project_root()

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


def _ensure_mapping(value: Any, *, section: str) -> dict[str, Any]:
    """Return ``value`` as a mapping or raise ``ValueError``.

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


def main(argv: Sequence[str] | None = None) -> None:
    """Run the UniProt target data retrieval workflow.

    Args:
        argv: Optional sequence of command-line arguments. When ``None``,
            the arguments provided via the command line are used implicitly.
    """

    from library.http_client import CacheConfig
    from library.io_utils import CsvConfig, read_ids, write_rows
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
        "--column", default=DEFAULT_COLUMN, help="Column with UniProt IDs"
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

    config_data = yaml.safe_load((ROOT / "config.yaml").read_text())
    config = _ensure_mapping(config_data, section="root configuration")
    uniprot_cfg = _ensure_mapping(config.get("uniprot", {}), section="uniprot")
    output_cfg = _ensure_mapping(config.get("output", {}), section="output")
    orth_cfg = _ensure_mapping(config.get("orthologs", {}), section="orthologs")

    http_cache_raw = config.get("http_cache")
    http_cache_cfg = (
        _ensure_mapping(http_cache_raw, section="http_cache")
        if http_cache_raw is not None
        else None
    )
    global_cache = CacheConfig.from_dict(http_cache_cfg)

    list_format = str(output_cfg.get("list_format", "json") or "json")
    include_seq = bool(
        args.include_sequence or output_cfg.get("include_sequence", False)
    )
    sep_value = args.sep or output_cfg.get("sep") or DEFAULT_SEP
    if not isinstance(sep_value, str):
        msg = "CSV separator must be a string"
        raise ValueError(msg)
    encoding_value = args.encoding or output_cfg.get("encoding") or DEFAULT_ENCODING
    if not isinstance(encoding_value, str):
        msg = "File encoding must be a string"
        raise ValueError(msg)
    include_iso = bool(args.with_isoforms or uniprot_cfg.get("include_isoforms", False))
    use_fasta_stream = bool(uniprot_cfg.get("use_fasta_stream_for_isoform_ids", True))

    # Use configuration defaults when CLI options are omitted
    csv_cfg = CsvConfig(sep=sep_value, encoding=encoding_value, list_format=list_format)

    client = UniProtClient(
        base_url=uniprot_cfg.get("base_url", "https://rest.uniprot.org/uniprotkb"),
        fields=(
            ",".join(uniprot_cfg.get("fields", [])) if uniprot_cfg.get("fields") else ""
        ),
        network=NetworkConfig(
            timeout_sec=uniprot_cfg.get("timeout_sec", 30),
            max_retries=uniprot_cfg.get("retries", 3),
            backoff_sec=1.0,
        ),
        rate_limit=RateLimitConfig(rps=uniprot_cfg.get("rps", 3)),
        cache=CacheConfig.from_dict(uniprot_cfg.get("cache")) or global_cache,
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

    accessions = read_ids(input_path, args.column, csv_cfg)
    rows: list[dict[str, Any]] = []
    iso_rows: list[dict[str, str]] = []

    cols = output_columns(include_seq)

    target_species: list[str] = []
    ensembl_client: EnsemblHomologyClient | None = None
    oma_client: OmaClient | None = None
    orth_cols: list[str] = []
    orth_rows: list[dict[str, Any]] = []

    if args.with_orthologs and orth_cfg.get("enabled", True):
        orth_cache_raw = orth_cfg.get("cache")
        orth_cache_cfg = (
            _ensure_mapping(orth_cache_raw, section="orthologs.cache")
            if orth_cache_raw is not None
            else None
        )
        orth_cache = CacheConfig.from_dict(orth_cache_cfg) or global_cache
        ensembl_client = EnsemblHomologyClient(
            base_url="https://rest.ensembl.org",
            network=NetworkConfig(
                timeout_sec=orth_cfg.get("timeout_sec", 30),
                max_retries=orth_cfg.get("retries", 3),
                backoff_sec=orth_cfg.get("backoff_base_sec", 1.0),
            ),
            rate_limit=RateLimitConfig(rps=orth_cfg.get("rate_limit_rps", 2)),
            cache=orth_cache,
        )
        oma_client = OmaClient(
            base_url="https://omabrowser.org/api",
            network=NetworkConfig(
                timeout_sec=orth_cfg.get("timeout_sec", 30),
                max_retries=orth_cfg.get("retries", 3),
                backoff_sec=orth_cfg.get("backoff_base_sec", 1.0),
            ),
            rate_limit=RateLimitConfig(rps=orth_cfg.get("rate_limit_rps", 2)),
            cache=orth_cache,
        )
        target_species_raw = orth_cfg.get("target_species", [])
        if not isinstance(target_species_raw, list):
            msg = "'orthologs.target_species' must be a list"
            raise ValueError(msg)
        target_species = [str(species) for species in target_species_raw]
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

    for acc in accessions:
        data = client.fetch_entry_json(acc)
        if data is None:
            LOGGER.warning(
                "No UniProt entry available for %s",
                acc,
                extra={"uniprot_id": acc},
            )
            row: dict[str, Any] = {c: "" for c in cols}
            row["uniprot_id"] = acc
            if ensembl_client:
                row["orthologs_json"] = "[]"
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
                        "isoform_synonyms": json.dumps(
                            iso["isoform_synonyms"],
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "is_canonical": str(iso["is_canonical"]).lower(),
                    }
                )

        row = normalize_entry(data, include_seq, isoforms)

        orthologs_json = "[]"
        orthologs_count = 0
        if ensembl_client and gene_ids:
            gene_id = gene_ids[0]
            orthologs = ensembl_client.get_orthologs(gene_id, target_species)
            if not orthologs and oma_client:
                orthologs = oma_client.get_orthologs_by_uniprot(acc)
            orthologs_json = json.dumps(
                [o.to_ordered_dict() for o in orthologs],
                separators=(",", ":"),
                sort_keys=True,
            )
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
                orthologs_json = json.dumps(
                    [o.to_ordered_dict() for o in orthologs],
                    separators=(",", ":"),
                    sort_keys=True,
                )
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
        LOGGER.info(
            "Ortholog table written to %s",
            orthologs_path,
            extra={"path": str(orthologs_path)},
        )

    rows.sort(key=lambda r: r.get("uniprot_id", ""))
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

    LOGGER.info(
        "Target table written to %s",
        output_path,
        extra={"path": str(output_path)},
    )


if __name__ == "__main__":  # pragma: no cover
    main()
