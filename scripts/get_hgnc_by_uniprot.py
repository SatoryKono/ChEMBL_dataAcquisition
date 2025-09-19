"""Command line interface for HGNC lookup by UniProt accession."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    from _path_utils import ensure_project_root as _ensure_project_root

    _ensure_project_root()

import pandas as pd

from hgnc_client import map_uniprot_to_hgnc  # noqa: E402
from library.cli_common import resolve_cli_sidecar_paths, write_cli_metadata  # noqa: E402
from library.logging_utils import configure_logging  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_SEP = ","
DEFAULT_ENCODING = "utf-8"
DEFAULT_COLUMN = "uniprot_id"
DEFAULT_LOG_FORMAT = "human"


def main(argv: Sequence[str] | None = None) -> None:
    """Parses command-line arguments and runs the HGNC mapping process.

    Args:
        argv: An optional list of command-line arguments. If None, the
            arguments are taken from `sys.argv`.
    """

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
    parser.add_argument(
        "--log-format",
        default=DEFAULT_LOG_FORMAT,
        choices=("human", "json"),
        help="Logging output format (human or json)",
    )
    parser.add_argument("--sep", default=DEFAULT_SEP, help="CSV field separator")
    parser.add_argument("--encoding", default=DEFAULT_ENCODING, help="File encoding")
    parser.add_argument(
        "--errors-output",
        default=None,
        help="Optional path for the JSON error report",
    )
    parser.add_argument(
        "--meta-output",
        default=None,
        help="Optional path for the generated .meta.yaml file",
    )
    args = parser.parse_args(argv)

    configure_logging(args.log_level, log_format=args.log_format)

    if args.config:
        config_path = Path(args.config)
        section = None
    else:
        config_path = ROOT / "config.yaml"
        section = "hgnc"

    output_override = Path(args.output) if args.output else None
    if output_override is not None:
        output_override = output_override.expanduser().resolve()
        output_override.parent.mkdir(parents=True, exist_ok=True)

    out_path = map_uniprot_to_hgnc(
        input_csv_path=Path(args.input),
        output_csv_path=output_override,
        config_path=config_path,
        config_section=section,
        column=args.column,
        sep=args.sep,
        encoding=args.encoding,
        log_level=args.log_level,
    )
    out_path = Path(out_path)

    meta_path, errors_path, _ = resolve_cli_sidecar_paths(
        out_path,
        meta_output=args.meta_output,
        errors_output=args.errors_output,
    )

    df = pd.read_csv(out_path, sep=args.sep, encoding=args.encoding)
    row_count, column_count = df.shape

    errors: list[dict[str, str]] = []
    if "uniprot_id" in df.columns and "hgnc_id" in df.columns:
        missing_mask = df["hgnc_id"].isna() | (df["hgnc_id"].astype(str).str.strip() == "")
        for uniprot_id in df.loc[missing_mask, "uniprot_id"].astype(str):
            errors.append(
                {
                    "uniprot_id": uniprot_id,
                    "error": "HGNC identifier missing",
                }
            )

    errors_path.parent.mkdir(parents=True, exist_ok=True)
    with errors_path.open("w", encoding="utf-8") as handle:
        json.dump(errors, handle, ensure_ascii=False, indent=2, sort_keys=True)

    command_parts = [sys.argv[0], *(argv or sys.argv[1:])]
    write_cli_metadata(
        out_path,
        row_count=int(row_count),
        column_count=int(column_count),
        namespace=args,
        command_parts=command_parts,
        meta_path=meta_path,
    )

    print(out_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
