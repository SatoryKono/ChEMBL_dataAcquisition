# ruff: noqa: E402
"""Command line interface for :mod:`uniprot_enrich`.

This script enriches a CSV file containing UniProt accessions with additional
annotations fetched from the UniProt REST API.  By default the input file is
modified in-place, but an explicit output path can be provided to write the
enriched data to a new file.

Examples
--------
Enrich ``data.csv`` in-place::

    python scripts/uniprot_enrich_main.py --input data.csv

Write the enriched result to ``out.csv`` and use a semicolon as list
separator::

    python scripts/uniprot_enrich_main.py \
        --input data.csv \
        --output out.csv \
        --sep ';'
"""

from __future__ import annotations

import argparse


from library.uniprot_enrich import enrich_uniprot
from library.chembl2uniprot.logging_utils import configure_logging

import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIB_DIR = ROOT / "library"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))


from uniprot_enrich import (  # noqa: E402
    enrich_uniprot,
)  # type: ignore[reportMissingImports]



DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_SEP = "|"
DEFAULT_ENCODING = "utf-8"


def main(argv: list[str] | None = None) -> None:
    """Entry point for the command line interface.

    Parameters
    ----------
    argv:
        Optional list of command line arguments. When ``None`` the arguments
        are taken from :data:`sys.argv`.
    """

    parser = argparse.ArgumentParser(description="Enrich UniProt data in a CSV file")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument(
        "--output",
        required=False,
        help="Optional path to write enriched CSV. Defaults to in-place update.",
    )
    parser.add_argument(
        "--log-level", default=DEFAULT_LOG_LEVEL, help="Logging level (e.g. INFO)"
    )
    parser.add_argument("--sep", default=DEFAULT_SEP, help="List separator")
    parser.add_argument(
        "--encoding",
        default=DEFAULT_ENCODING,
        help="File encoding for CSV input and output",
    )
    parser.add_argument(
        "--log-format",
        default="human",
        choices=["human", "json"],
        help="Logging output format",
    )
    args = parser.parse_args()
    configure_logging(args.log_level, json_logs=args.log_format == "json")
    enrich_uniprot(args.input, list_sep=args.sep)



if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
