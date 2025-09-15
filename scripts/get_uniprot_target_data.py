"""Retrieve and normalise UniProtKB target information.

Example
-------
>>> python scripts/get_uniprot_target_data.py --input data.csv --output out.csv
"""

from __future__ import annotations

import argparse
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import json

import yaml

ROOT = Path(__file__).resolve().parents[1]
LIB_DIR = ROOT / "library"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from io_utils import CsvConfig, read_ids, write_rows  # noqa: E402
from uniprot_client import NetworkConfig, RateLimitConfig, UniProtClient  # noqa: E402
from uniprot_normalize import (  # noqa: E402

    extract_ensembl_gene_ids,
    normalize_entry,
    output_columns,
)
from orthologs import EnsemblHomologyClient, OmaClient  # noqa: E402

    Isoform,
    extract_isoforms,
    normalize_entry,
    output_columns,
)


DEFAULT_INPUT = "input.csv"
DEFAULT_OUTPUT = "output_input_{date}.csv"
DEFAULT_COLUMN = "uniprot_id"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_SEP = ","
DEFAULT_ENCODING = "utf-8"


def _default_output(input_path: Path) -> Path:
    date = datetime.now().strftime("%Y%m%d")
    return input_path.with_name(DEFAULT_OUTPUT.format(date=date))


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch UniProt target data")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV path")
    parser.add_argument("--output", help="Output CSV path")
    parser.add_argument(
        "--column", default=DEFAULT_COLUMN, help="Column with UniProt IDs"
    )
    parser.add_argument("--sep", default=DEFAULT_SEP, help="CSV separator")
    parser.add_argument("--encoding", default=DEFAULT_ENCODING, help="File encoding")
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
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    uniprot_cfg = config.get("uniprot", {})
    output_cfg = config.get("output", {})
    orth_cfg = config.get("orthologs", {})

    list_format = output_cfg.get("list_format", "json")
    include_seq = args.include_sequence or output_cfg.get("include_sequence", False)
    include_iso = args.with_isoforms or uniprot_cfg.get("include_isoforms", False)
    use_fasta_stream = uniprot_cfg.get("use_fasta_stream_for_isoform_ids", True)

    csv_cfg = CsvConfig(sep=args.sep, encoding=args.encoding, list_format=list_format)

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
    rows: List[Dict[str, str]] = []
    iso_rows: List[Dict[str, str]] = []

    cols = output_columns(include_seq)

    target_species: List[str] = []
    ensembl_client: EnsemblHomologyClient | None = None
    oma_client: OmaClient | None = None
    orth_cols: List[str] = []
    orth_rows: List[Dict[str, Any]] = []

    if args.with_orthologs and orth_cfg.get("enabled", True):
        ensembl_client = EnsemblHomologyClient(
            base_url="https://rest.ensembl.org",
            network=NetworkConfig(
                timeout_sec=orth_cfg.get("timeout_sec", 30),
                max_retries=orth_cfg.get("retries", 3),
                backoff_sec=orth_cfg.get("backoff_base_sec", 1.0),
            ),
            rate_limit=RateLimitConfig(rps=orth_cfg.get("rate_limit_rps", 2)),
        )
        oma_client = OmaClient(
            base_url="https://omabrowser.org/api",
            network=NetworkConfig(
                timeout_sec=orth_cfg.get("timeout_sec", 30),
                max_retries=orth_cfg.get("retries", 3),
                backoff_sec=orth_cfg.get("backoff_base_sec", 1.0),
            ),
            rate_limit=RateLimitConfig(rps=orth_cfg.get("rate_limit_rps", 2)),
        )
        target_species = orth_cfg.get("target_species", [])
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
<
        data = client.fetch(acc)
        gene_ids: List[str] = []
        if data is None:

        entry = client.fetch_entry_json(acc)
        fasta_headers: List[str] = []
        if include_iso and entry is not None and use_fasta_stream:
            fasta_headers = client.fetch_isoforms_fasta(acc)
        if entry is None:

            logging.warning("No entry for %s", acc)
            row: Dict[str, Any] = {c: "" for c in cols}
            row["uniprot_id"] = acc

        else:
            gene_ids = extract_ensembl_gene_ids(data)
            row = normalize_entry(data, include_seq)

        orthologs_json = "[]"
        orthologs_count = 0
        if ensembl_client and gene_ids:
            gene_id = gene_ids[0]
            orthologs = ensembl_client.get_orthologs(gene_id, target_species)
            if not orthologs:
                orthologs = (
                    oma_client.get_orthologs_by_uniprot(acc) if oma_client else []
                )
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
            logging.warning("No Ensembl gene ID for %s", acc)
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

    write_rows(output_path, rows, cols, csv_cfg)
    if ensembl_client and orth_rows:
        orth_rows.sort(
            key=lambda x: (
                x["source_uniprot_id"],
                x["target_species"],
                x["target_gene_symbol"],
            )
        )
        write_rows(orthologs_path, orth_rows, orth_cols, csv_cfg)
        print(orthologs_path)

            rows.append(row)
            continue
        isoforms: List[Isoform] = []
        if include_iso:
            isoforms = extract_isoforms(entry, fasta_headers)
            for iso in isoforms:
                iso_rows.append(
                    {
                        "parent_uniprot_id": acc,
                        "isoform_uniprot_id": iso["isoform_uniprot_id"],
                        "isoform_name": iso["isoform_name"],
                        "isoform_synonyms": iso["isoform_synonyms"],
                        "is_canonical": str(iso["is_canonical"]).lower(),
                    }
                )
        row = normalize_entry(entry, include_seq, isoforms)
        rows.append(row)

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

    print(output_path)


if __name__ == "__main__":  # pragma: no cover
    main()
