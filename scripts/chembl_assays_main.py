"""Command line entry point for downloading and normalising ChEMBL assays."""

from __future__ import annotations

import argparse
import logging
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd

if __package__ in {None, ""}:
    from _path_utils import ensure_project_root as _ensure_project_root

    _ensure_project_root()

from library.assay_postprocessing import postprocess_assays
from library.assay_validation import AssaysSchema, validate_assays
from library.chembl_client import ChemblClient
from library.chembl_library import get_assays
from library.cli_common import resolve_cli_sidecar_paths, serialise_dataframe
from library.data_profiling import analyze_table_quality
from library.io import read_ids
from library.io_utils import CsvConfig
from library.metadata import write_meta_yaml
from library.normalize_assays import normalize_assays
from library.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)
DEFAULT_LOG_FORMAT = "human"


def _default_output_name(input_path: str) -> str:
    stem = Path(input_path).stem or "output"
    date_suffix = datetime.now().strftime("%Y%m%d")
    return f"output_{stem}_{date_suffix}.csv"


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments for the assay pipeline CLI.

    Args:
        args: A sequence of command-line arguments. If None, `sys.argv` is used.

    Returns:
        An `argparse.Namespace` object containing the parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="input.csv", help="Path to the input CSV file"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Destination CSV file. Defaults to output_<input>_<YYYYMMDD>.csv",
    )
    parser.add_argument(
        "--column", default="assay_chembl_id", help="Column containing assay IDs"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20,
        help="Number of IDs fetched per batch (must be positive)",
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="HTTP timeout in seconds"
    )
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Maximum retry attempts"
    )
    parser.add_argument(
        "--rps", type=float, default=2.0, help="Maximum requests per second"
    )
    parser.add_argument(
        "--base-url",
        default="https://www.ebi.ac.uk/chembl/api/data",
        help="ChEMBL API root",
    )
    parser.add_argument("--sep", default=",", help="CSV delimiter")
    parser.add_argument("--encoding", default="utf-8", help="CSV encoding")
    parser.add_argument(
        "--list-format",
        choices=["json", "pipe"],
        default="json",
        help="Serialization format for list columns",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (e.g. INFO, DEBUG)"
    )
    parser.add_argument(
        "--log-format",
        default=DEFAULT_LOG_FORMAT,
        choices=("human", "json"),
        help="Logging output format (human or json)",
    )
    parser.add_argument(
        "--errors-output", default=None, help="Path to validation error report"
    )
    parser.add_argument(
        "--meta-output", default=None, help="Optional metadata YAML path"
    )
    parsed_args = parser.parse_args(args)
    if parsed_args.chunk_size <= 0:
        parser.error("--chunk-size must be a positive integer")
    return parsed_args


def _prepare_configuration(namespace: argparse.Namespace) -> dict[str, object]:
    config: dict[str, object] = {}
    for key, value in vars(namespace).items():
        if key in {"output", "errors_output", "meta_output"}:
            continue
        if isinstance(value, Path):
            config[key] = str(value)
        else:
            config[key] = value
    return config


def run_pipeline(
    args: argparse.Namespace, *, command_parts: Sequence[str] | None = None
) -> int:
    """Executes the assay acquisition pipeline with the given arguments.

    Args:
        args: An `argparse.Namespace` object containing the pipeline arguments.
        command_parts: A sequence of command-line arguments used to invoke the
            pipeline. If None, `sys.argv` is used.

    Returns:
        An exit code, 0 for success and 1 for failure.
    """

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} does not exist")

    output_path = (
        Path(args.output) if args.output else Path(_default_output_name(args.input))
    )
    meta_path, errors_path, quality_base = resolve_cli_sidecar_paths(
        output_path,
        meta_output=args.meta_output,
        errors_output=args.errors_output,
    )

    csv_cfg = CsvConfig(
        sep=args.sep, encoding=args.encoding, list_format=args.list_format
    )
    assay_ids = read_ids(input_path, args.column, csv_cfg)

    with ChemblClient(
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
        rps=args.rps,
    ) as client:
        assays_df = get_assays(client, assay_ids, chunk_size=args.chunk_size)
    if assays_df.empty:
        LOGGER.warning("No assay data retrieved; writing empty output")
        assays_df = pd.DataFrame(columns=AssaysSchema.ordered_columns())

    processed = postprocess_assays(assays_df)
    normalised = normalize_assays(processed)
    validated = validate_assays(normalised, errors_path=errors_path)

    schema_columns = [
        column
        for column in AssaysSchema.ordered_columns()
        if column in validated.columns
    ]
    extra_columns = sorted(
        [column for column in validated.columns if column not in schema_columns]
    )
    ordered_columns = schema_columns + extra_columns
    if ordered_columns:
        validated = validated[ordered_columns]

    sort_columns = [
        column
        for column in ["document_chembl_id", "target_chembl_id", "assay_chembl_id"]
        if column in validated.columns
    ]
    if sort_columns:
        validated = validated.sort_values(sort_columns).reset_index(drop=True)

    serialised = serialise_dataframe(validated, args.list_format, inplace=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialised.to_csv(output_path, index=False, sep=args.sep, encoding=args.encoding)

    if command_parts is None:
        command_parts = sys.argv

    write_meta_yaml(
        output_path,
        command=" ".join(shlex.quote(part) for part in command_parts),
        config=_prepare_configuration(args),
        row_count=int(len(serialised)),
        column_count=int(len(serialised.columns)),
        meta_path=meta_path,
    )

    analyze_table_quality(serialised, table_name=str(quality_base))

    LOGGER.info("Assay table written to %s", output_path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """The entry point used by the CLI and tests.

    Args:
        argv: A sequence of command-line arguments. If None, `sys.argv` is used.

    Returns:
        An exit code, 0 for success and 1 for failure.
    """

    args = parse_args(argv)
    configure_logging(args.log_level, log_format=args.log_format)
    try:
        cmd_parts = [sys.argv[0], *(argv or sys.argv[1:])]
        return run_pipeline(args, command_parts=cmd_parts)
    except Exception as exc:  # pragma: no cover - safety net
        LOGGER.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
