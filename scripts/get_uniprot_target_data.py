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
from typing import Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
LIB_DIR = ROOT / "library"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from io_utils import CsvConfig, read_ids, write_rows  # noqa: E402
from uniprot_client import NetworkConfig, RateLimitConfig, UniProtClient  # noqa: E402
from uniprot_normalize import normalize_entry, output_columns  # noqa: E402

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
        "--include-sequence", action="store_true", help="Include full protein sequence"
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
    network_cfg = config.get("network", {})
    rate_cfg = config.get("rate_limit", {})
    output_cfg = config.get("output", {})

    list_format = output_cfg.get("list_format", "json")
    include_seq = args.include_sequence or output_cfg.get("include_sequence", False)

    csv_cfg = CsvConfig(sep=args.sep, encoding=args.encoding, list_format=list_format)

    client = UniProtClient(
        base_url=uniprot_cfg.get("base_url", "https://rest.uniprot.org/uniprotkb"),
        fields=",".join(uniprot_cfg.get("fields", [])),
        network=NetworkConfig(
            timeout_sec=network_cfg.get("timeout_sec", 30),
            max_retries=network_cfg.get("max_retries", 3),
            backoff_sec=1.0,
        ),
        rate_limit=RateLimitConfig(rps=rate_cfg.get("rps", 3)),
    )

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else _default_output(input_path)

    accessions = read_ids(input_path, args.column, csv_cfg)
    rows: List[Dict[str, str]] = []
    cols = output_columns(include_seq)

    for acc in accessions:
        data = client.fetch(acc)
        if data is None:
            logging.warning("No entry for %s", acc)
            row = {c: "" for c in cols}
            row["uniprot_id"] = acc
        else:
            row = normalize_entry(data, include_seq)
        rows.append(row)

    write_rows(output_path, rows, cols, csv_cfg)
    print(output_path)


if __name__ == "__main__":  # pragma: no cover
    main()
