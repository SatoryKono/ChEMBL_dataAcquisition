"""CLI for enriching UniProt annotations in-place."""

import argparse

from library.uniprot_enrich import enrich_uniprot
from library.chembl2uniprot.logging_utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich UniProt data in CSV")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--sep", default="|", help="List separator")
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (e.g. INFO, DEBUG)"
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


if __name__ == "__main__":
    main()
