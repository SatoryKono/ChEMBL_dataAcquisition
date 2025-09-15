"""CLI for enriching UniProt annotations in-place."""

import argparse
import logging

from .library.uniprot_enrich import enrich


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich UniProt data in CSV")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--sep", default="|", help="List separator")
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (e.g. INFO, DEBUG)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    enrich_uniprot(args.input, list_sep=args.sep)


if __name__ == "__main__":
    main()
