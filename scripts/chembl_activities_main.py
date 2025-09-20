"""Command line entry point for downloading and normalising ChEMBL activities."""

from __future__ import annotations

if __package__ in {None, ""}:
    from _path_utils import ensure_project_root as _ensure_project_root

    _ensure_project_root()

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence, cast

import pandas as pd

from library.cli_common import (
    ensure_output_dir,
    resolve_cli_sidecar_paths,
    serialise_dataframe,
    write_cli_metadata,
)
from library.activity_validation import ActivitiesSchema, validate_activities
from library.chembl_client import ChemblClient
from library.chembl_library import get_activities
from library.data_profiling import analyze_table_quality
from library.io import read_ids
from library.io_utils import CsvConfig
from library.normalize_activities import normalize_activities
from library.logging_utils import configure_logging
from library.config.chembl import load_chembl_activities_config

LOGGER = logging.getLogger(__name__)
DEFAULT_LOG_FORMAT = "human"


def _default_output_name(input_path: str) -> str:
    stem = Path(input_path).stem or "output"
    date_suffix = datetime.now().strftime("%Y%m%d")
    return f"output_{stem}_{date_suffix}.csv"


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments for the activity pipeline CLI.

    Args:
        args: A sequence of command-line arguments. If None, `sys.argv` is used.

    Returns:
        An `argparse.Namespace` object containing the parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML file providing the chembl_activities section",
    )
    parser.add_argument(
        "--input", default="input.csv", help="Path to the input CSV file"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Destination CSV file. Defaults to output_<input>_<YYYYMMDD>.csv",
    )
    parser.add_argument(
        "--column", default="activity_chembl_id", help="Column containing activity IDs"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Number of IDs fetched per batch (defaults to config)",
    )
    parser.add_argument(
        "--timeout", type=float, default=None, help="HTTP timeout in seconds"
    )
    parser.add_argument(
        "--max-retries", type=int, default=None, help="Maximum retry attempts"
    )
    parser.add_argument(
        "--rps", type=float, default=None, help="Maximum requests per second"
    )
    parser.add_argument(
        "--retry-penalty",
        type=float,
        default=None,
        help=(
            "Fallback sleep in seconds applied after 429 responses without a"
            " Retry-After header"
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="ChEMBL API root (defaults to config)",
    )
    parser.add_argument(
        "--user-agent", default=None, help="User-Agent header (defaults to config)"
    )
    parser.add_argument("--sep", default=None, help="CSV delimiter (defaults to config)")
    parser.add_argument(
        "--encoding", default=None, help="CSV encoding (defaults to config)"
    )
    parser.add_argument(
        "--list-format",
        choices=["json", "pipe"],
        default=None,
        help="Serialization format for list columns (defaults to config)",
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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of identifiers to process",
    )
    parser.add_argument(
        "--dictionary",
        default=None,
        help="Optional dictionary file for downstream enrichment",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and validate the input file without fetching or writing output",
    )
    parsed_args = parser.parse_args(args)

    try:
        cfg = load_chembl_activities_config(parsed_args.config)
    except FileNotFoundError as exc:
        parser.error(f"Configuration file not found: {exc}")
    except ValueError as exc:
        parser.error(f"Failed to load configuration: {exc}")

    if parsed_args.chunk_size is None:
        parsed_args.chunk_size = cfg.chunk_size
    if parsed_args.timeout is None:
        parsed_args.timeout = cfg.network.timeout_sec
    if parsed_args.max_retries is None:
        parsed_args.max_retries = cfg.network.max_retries
    if parsed_args.retry_penalty is None:
        parsed_args.retry_penalty = cfg.network.retry_penalty_sec
    if parsed_args.rps is None:
        parsed_args.rps = cfg.rate_limit.rps
    if parsed_args.base_url is None:
        parsed_args.base_url = cfg.base_url
    if parsed_args.user_agent is None:
        parsed_args.user_agent = cfg.user_agent
    if parsed_args.sep is None:
        parsed_args.sep = cfg.csv.sep
    if parsed_args.encoding is None:
        parsed_args.encoding = cfg.csv.encoding
    if parsed_args.list_format is None:
        parsed_args.list_format = cfg.csv.list_format

    if parsed_args.chunk_size <= 0:
        parser.error("--chunk-size must be a positive integer")
    return parsed_args


def _limited_ids(
    path: Path, column: str, cfg: CsvConfig, limit: int | None
) -> Iterable[str]:
    return cast(Iterable[str], read_ids(path, column, cfg, limit=limit))


def run_pipeline(
    args: argparse.Namespace, *, command_parts: Sequence[str] | None = None
) -> int:
    """Executes the activity acquisition pipeline with the given arguments.

    Args:
        args: An `argparse.Namespace` object containing the pipeline arguments.
        command_parts: A sequence of command-line arguments used to invoke the
            pipeline. If None, `sys.argv` is used.

    Returns:
        An exit code, 0 for success and 1 for failure.
    """

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer")

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

    if args.dry_run:
        count = sum(
            1 for _ in _limited_ids(input_path, args.column, csv_cfg, args.limit)
        )
        LOGGER.info("Dry run complete: %d unique identifiers would be processed", count)
        return 0

    activity_ids = _limited_ids(input_path, args.column, csv_cfg, args.limit)

    client = ChemblClient(
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
        rps=args.rps,
        user_agent=args.user_agent,
        retry_penalty_seconds=args.retry_penalty,
    )

    activities_df = get_activities(client, activity_ids, chunk_size=args.chunk_size)
    if activities_df.empty:
        LOGGER.warning("No activity data retrieved; writing empty output")
        activities_df = pd.DataFrame(columns=ActivitiesSchema.ordered_columns())

    normalised = normalize_activities(activities_df)
    validated = validate_activities(normalised, errors_path=errors_path)

    schema_columns = [
        column
        for column in ActivitiesSchema.ordered_columns()
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
        for column in ["assay_chembl_id", "molecule_chembl_id", "activity_chembl_id"]
        if column in validated.columns
    ]
    if sort_columns:
        validated = validated.sort_values(sort_columns).reset_index(drop=True)

    serialised = serialise_dataframe(validated, args.list_format, inplace=True)
    ensure_output_dir(output_path)
    serialised.to_csv(output_path, index=False, sep=args.sep, encoding=args.encoding)

    write_cli_metadata(
        output_path,
        row_count=int(len(serialised)),
        column_count=int(len(serialised.columns)),
        namespace=args,
        command_parts=command_parts,
        meta_path=meta_path,
    )

    analyze_table_quality(serialised, table_name=str(quality_base))

    LOGGER.info("Activity table written to %s", output_path)
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
