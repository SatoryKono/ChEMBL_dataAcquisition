"""CLI tool to map UniProt accessions to IUPHAR classifications."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

from library.iuphar import IUPHARData


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="Path to _IUPHAR_target.csv")
    parser.add_argument("--family", required=True, help="Path to _IUPHAR_family.csv")
    parser.add_argument(
        "--input", default="input.csv", help="Input CSV with uniprot_id column"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: output_<input>_<YYYYMMDD>.csv)",
    )
    parser.add_argument("--sep", default=",", help="CSV delimiter")
    parser.add_argument("--encoding", default="utf-8", help="File encoding")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s %(message)s",
    )
    if args.output is None:
        date = dt.date.today().strftime("%Y%m%d")
        stem = Path(args.input).stem
        args.output = f"output_{stem}_{date}.csv"

    data = IUPHARData.from_files(args.target, args.family, encoding=args.encoding)
    data.map_uniprot_file(
        args.input,
        args.output,
        encoding=args.encoding,
        sep=args.sep,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
