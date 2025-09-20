"""Command line entry point for downloading and normalising ChEMBL activities."""

from __future__ import annotations

if __package__ in {None, ""}:
    from _path_utils import ensure_project_root as _ensure_project_root  # type: ignore[import-not-found]

    _ensure_project_root()

import argparse
import json
import logging
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml

from library.cli_common import (
    ListFormat,
    ensure_output_dir,
    resolve_cli_sidecar_paths,
    serialise_dataframe,
    write_cli_metadata,
)
from library.activity_validation import ActivitiesSchema, validate_activities
from library.chembl_client import ChemblClient
from library.chembl_library import stream_activities
from library.data_profiling import analyze_table_quality
from library.io import read_ids
from library.io_utils import CsvConfig
from library.normalize_activities import normalize_activities
from library.logging_utils import configure_logging

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
        "--retry-penalty",
        type=float,
        default=1.0,
        help=(
            "Fallback sleep in seconds applied after 429 responses without a"
            " Retry-After header"
        ),
    )
    parser.add_argument(
        "--base-url",
        default="https://www.ebi.ac.uk/chembl/api/data",
        help="ChEMBL API root",
    )
    parser.add_argument(
        "--user-agent", default="ChEMBLDataAcquisition/1.0", help="User-Agent header"
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
    if parsed_args.chunk_size <= 0:
        parser.error("--chunk-size must be a positive integer")
    return parsed_args


def _limited_ids(
    path: Path, column: str, cfg: CsvConfig, limit: int | None
) -> Iterable[str]:
    return cast(Iterable[str], read_ids(path, column, cfg, limit=limit))


def _load_existing_metadata(meta_path: Path) -> dict[str, Any]:
    """Load a previously generated metadata file if it exists."""

    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}
    except yaml.YAMLError as exc:
        LOGGER.warning("Failed to parse metadata file %s: %s", meta_path, exc)
        return {}
    if isinstance(payload, Mapping):
        return dict(payload)
    LOGGER.warning("Metadata file %s does not contain a mapping", meta_path)
    return {}


def _load_existing_errors(errors_path: Path) -> list[dict[str, Any]]:
    """Load validation errors from a previous run when resuming."""

    if not errors_path.exists():
        return []
    try:
        with errors_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as exc:
        LOGGER.warning("Failed to parse validation report %s: %s", errors_path, exc)
        return []
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    LOGGER.warning("Validation report %s is not a list; ignoring", errors_path)
    return []


def _consume_error_file(path: Path) -> list[dict[str, Any]]:
    """Return and remove a temporary error file produced during validation."""

    if not path.exists():
        return []
    errors = _load_existing_errors(path)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    return errors


def _skip_processed_ids(
    loader: Callable[[], Iterable[str]], last_processed: str | None
) -> Iterator[str]:
    """Skip identifiers processed in a previous run using checkpoint metadata."""

    if last_processed is None:
        yield from loader()
        return

    iterator = iter(loader())
    for identifier in iterator:
        if identifier == last_processed:
            LOGGER.info(
                "Resuming after previously processed identifier %s", last_processed
            )
            break
    else:
        LOGGER.warning(
            "Previously processed identifier %s not found; restarting from the beginning",
            last_processed,
        )
        yield from loader()
        return

    # Continue from the element immediately after ``last_processed``.
    for identifier in iterator:
        yield identifier


def _prepare_activity_chunk(
    chunk: pd.DataFrame,
    *,
    list_format: ListFormat,
    errors_path: Path,
    error_accumulator: list[dict[str, Any]],
    ordered_columns: list[str] | None,
) -> tuple[pd.DataFrame, list[str] | None, str | None]:
    """Normalise, validate, and serialise a single chunk of activity records."""

    if chunk.empty:
        return pd.DataFrame(), ordered_columns, None

    normalised = normalize_activities(chunk)

    temp_errors = errors_path.with_name(f"{errors_path.name}.part")
    validated = validate_activities(normalised, errors_path=temp_errors)
    error_accumulator.extend(_consume_error_file(temp_errors))

    if validated.empty:
        return pd.DataFrame(), ordered_columns, None

    if ordered_columns is None:
        schema_columns = [
            column
            for column in ActivitiesSchema.ordered_columns()
            if column in validated.columns
        ]
        extra_columns = sorted(
            [column for column in validated.columns if column not in schema_columns]
        )
        ordered_columns = schema_columns + extra_columns
    else:
        missing_columns = [
            column for column in validated.columns if column not in ordered_columns
        ]
        if missing_columns:
            ordered_columns = [*ordered_columns, *sorted(missing_columns)]

    prepared = validated.reindex(
        columns=ordered_columns, fill_value=cast(object, pd.NA)
    )

    sort_columns = [
        column
        for column in ["assay_chembl_id", "molecule_chembl_id", "activity_chembl_id"]
        if column in prepared.columns
    ]
    if sort_columns:
        prepared = prepared.sort_values(sort_columns).reset_index(drop=True)

    serialised = serialise_dataframe(prepared, list_format, inplace=True)

    last_identifier: str | None = None
    if "activity_chembl_id" in serialised.columns and not serialised.empty:
        last_identifier = str(serialised["activity_chembl_id"].iloc[-1])

    return serialised, ordered_columns, last_identifier


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

    list_format = cast(ListFormat, args.list_format)

    csv_cfg = CsvConfig(sep=args.sep, encoding=args.encoding, list_format=list_format)

    if args.dry_run:
        count = sum(
            1 for _ in _limited_ids(input_path, args.column, csv_cfg, args.limit)
        )
        LOGGER.info("Dry run complete: %d unique identifiers would be processed", count)
        return 0

    def id_loader() -> Iterable[str]:
        return _limited_ids(input_path, args.column, csv_cfg, args.limit)

    existing_metadata = _load_existing_metadata(meta_path)
    progress_meta = existing_metadata.get("progress")
    resume_id: str | None = None
    ordered_columns: list[str] | None = None
    if isinstance(progress_meta, Mapping):
        raw_last = progress_meta.get("last_id")
        if isinstance(raw_last, str) and raw_last:
            resume_id = raw_last
        column_order = progress_meta.get("column_order")
        if isinstance(column_order, list) and all(
            isinstance(item, str) for item in column_order
        ):
            ordered_columns = list(column_order)

    total_rows = existing_metadata.get("rows")
    if not isinstance(total_rows, int) or resume_id is None:
        total_rows = 0

    aggregated_errors = _load_existing_errors(errors_path)
    last_identifier = resume_id

    if resume_id is None:
        aggregated_errors = []
        if errors_path.exists():
            errors_path.unlink()

    if resume_id is None and output_path.exists():
        LOGGER.info("Starting fresh run and replacing existing output %s", output_path)
        output_path.unlink()

    ensure_output_dir(output_path)
    header_written = output_path.exists() and output_path.stat().st_size > 0
    if header_written and resume_id is None:
        header_written = False

    client = ChemblClient(
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
        rps=args.rps,
        user_agent=args.user_agent,
        retry_penalty_seconds=args.retry_penalty,
    )

    identifier_stream = _skip_processed_ids(id_loader, resume_id)

    for chunk in stream_activities(
        client, identifier_stream, chunk_size=args.chunk_size
    ):
        serialised, ordered_columns, chunk_last = _prepare_activity_chunk(
            chunk,
            list_format=list_format,
            errors_path=errors_path,
            error_accumulator=aggregated_errors,
            ordered_columns=ordered_columns,
        )

        if serialised.empty:
            if chunk_last is not None:
                last_identifier = chunk_last
            continue

        serialised.to_csv(
            output_path,
            mode="a",
            header=not header_written,
            index=False,
            sep=args.sep,
            encoding=args.encoding,
        )
        header_written = True
        total_rows += int(len(serialised))
        if chunk_last is not None:
            last_identifier = chunk_last

        column_count = len(ordered_columns) if ordered_columns is not None else 0
        progress_payload = {
            "progress": {
                "last_id": last_identifier,
                "column_order": ordered_columns or [],
            }
        }
        write_cli_metadata(
            output_path,
            row_count=total_rows,
            column_count=column_count,
            namespace=args,
            command_parts=command_parts,
            meta_path=meta_path,
            extra=progress_payload,
        )

    if not header_written:
        LOGGER.warning("No activity data retrieved; writing empty output")
        empty_frame = pd.DataFrame(columns=ActivitiesSchema.ordered_columns())
        empty_frame.to_csv(
            output_path,
            index=False,
            sep=args.sep,
            encoding=args.encoding,
        )
        ordered_columns = list(empty_frame.columns)
        column_count = len(ordered_columns)
        progress_payload = {
            "progress": {"last_id": last_identifier, "column_order": ordered_columns}
        }
        write_cli_metadata(
            output_path,
            row_count=0,
            column_count=column_count,
            namespace=args,
            command_parts=command_parts,
            meta_path=meta_path,
            extra=progress_payload,
        )
    else:
        column_count = len(ordered_columns) if ordered_columns is not None else 0
        progress_payload = {
            "progress": {
                "last_id": last_identifier,
                "column_order": ordered_columns or [],
            }
        }
        write_cli_metadata(
            output_path,
            row_count=total_rows,
            column_count=column_count,
            namespace=args,
            command_parts=command_parts,
            meta_path=meta_path,
            extra=progress_payload,
        )

    if aggregated_errors:
        errors_path.parent.mkdir(parents=True, exist_ok=True)
        with errors_path.open("w", encoding="utf-8") as handle:
            json.dump(aggregated_errors, handle, ensure_ascii=False, indent=2)
    elif errors_path.exists():
        errors_path.unlink()

    analyze_table_quality(
        output_path,
        table_name=str(quality_base),
        separator=args.sep,
        encoding=args.encoding,
    )

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
