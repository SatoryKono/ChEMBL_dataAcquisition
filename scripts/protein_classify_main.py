# ruff: noqa: E402
"""Command line interface for protein classification.

This script reads a CSV file containing UniProt JSON entries and adds
classification columns (L1/L2/L3, rule ID, confidence and evidence).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    from _path_utils import ensure_project_root as _ensure_project_root

    _ensure_project_root()

from library.protein_classifier import classify_protein
from library.logging_utils import configure_logging
from library.data_profiling import analyze_table_quality

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_SEP = ";"
DEFAULT_ENCODING = "utf-8"
DEFAULT_COLUMN = "uniprot_json"
DEFAULT_LOG_FORMAT = "human"


def _default_output(path: str) -> str:
    """Generate a default output file path based on the input path."""
    stem = Path(path).stem
    date = datetime.utcnow().strftime("%Y%m%d")
    return f"output_{stem}_{date}.csv"


def main(argv: list[str] | None = None) -> None:
    """Entry point for the protein classification CLI."""
    parser = argparse.ArgumentParser(
        description="Annotate proteins with classification labels"
    )
    parser.add_argument("--input", default="input.csv", help="Path to input CSV file")
    parser.add_argument("--output", help="Optional path for output CSV")
    parser.add_argument("--log-level", default=DEFAULT_LOG_LEVEL, help="Logging level")
    parser.add_argument(
        "--log-format",
        default=DEFAULT_LOG_FORMAT,
        choices=("human", "json"),
        help="Logging output format (human or json)",
    )
    parser.add_argument(
        "--sep", default=DEFAULT_SEP, help="List separator for evidence column"
    )
    parser.add_argument("--encoding", default=DEFAULT_ENCODING, help="CSV encoding")
    parser.add_argument(
        "--column", default=DEFAULT_COLUMN, help="Column containing UniProt JSON"
    )
    args = parser.parse_args(argv)

    configure_logging(args.log_level, log_format=args.log_format)

    input_path = Path(args.input)
    output_path = (
        Path(args.output) if args.output else Path(_default_output(args.input))
    )

    df = pd.read_csv(input_path, dtype={args.column: str}, encoding=args.encoding)
    if args.column not in df.columns:
        raise ValueError(f"input CSV must contain '{args.column}' column")

    results = df[args.column].apply(
        lambda x: classify_protein(json.loads(x)) if isinstance(x, str) and x else {}
    )
    df["protein_class_L1"] = results.apply(lambda r: r.get("protein_class_L1", ""))
    df["protein_class_L2"] = results.apply(lambda r: r.get("protein_class_L2", ""))
    df["protein_class_L3"] = results.apply(lambda r: r.get("protein_class_L3", ""))
    df["rule_id"] = results.apply(lambda r: r.get("rule_id", ""))
    df["confidence"] = results.apply(lambda r: r.get("confidence", ""))
    df["evidence"] = results.apply(lambda r: ";".join(r.get("evidence", [])))

    df.to_csv(output_path, index=False, encoding=args.encoding, lineterminator="\n")
    analyze_table_quality(df, table_name=str(output_path.with_suffix("")))
    print(output_path)


if __name__ == "__main__":  # pragma: no cover
    main()
