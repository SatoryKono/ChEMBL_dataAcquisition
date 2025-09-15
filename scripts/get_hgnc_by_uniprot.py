"""Command line interface for HGNC lookup by UniProt accession."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB_DIR = ROOT / "library"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from hgnc_client import map_uniprot_to_hgnc  # noqa: E402

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_SEP = ","
DEFAULT_ENCODING = "utf-8"
DEFAULT_COLUMN = "uniprot_id"


def main(argv: list[str] | None = None) -> None:
    """Entry point for the HGNC mapping CLI."""

    parser = argparse.ArgumentParser(description="Map UniProt accessions to HGNC IDs")
    parser.add_argument("--input", default="input.csv", help="Path to input CSV file")
    parser.add_argument("--output", help="Path to output CSV file", required=False)
    parser.add_argument(
        "--column", default=DEFAULT_COLUMN, help="Name of UniProt column"
    )
    parser.add_argument(
        "--config", help="Path to YAML configuration file", required=False
    )
    parser.add_argument("--log-level", default=DEFAULT_LOG_LEVEL, help="Logging level")
    parser.add_argument("--sep", default=DEFAULT_SEP, help="CSV field separator")
    parser.add_argument("--encoding", default=DEFAULT_ENCODING, help="File encoding")
    args = parser.parse_args(argv)

    if args.config:
        config_path = Path(args.config)
        section = None
    else:
        config_path = ROOT / "config.yaml"
        section = "hgnc"

    out_path = map_uniprot_to_hgnc(
        input_csv_path=Path(args.input),
        output_csv_path=Path(args.output) if args.output else None,
        config_path=config_path,
        config_section=section,
        column=args.column,
        sep=args.sep,
        encoding=args.encoding,
        log_level=args.log_level,
    )
    print(out_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
