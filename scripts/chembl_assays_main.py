"""Command line entry point for downloading and normalising ChEMBL assays."""

from __future__ import annotations

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

if __package__ in {None, ""}:
    from _path_utils import ensure_project_root as _ensure_project_root  # type: ignore[import-not-found]

    _ensure_project_root()

from library.assay_postprocessing import postprocess_assays
from library.assay_validation import AssaysSchema, validate_assays
from library.chembl_client import ChemblClient

from library.chembl_library import get_assays
from library.cli_common import (
    resolve_cli_sidecar_paths,
    serialise_dataframe,
    write_cli_metadata,

)
from library.data_profiling import analyze_table_quality
from library.io import read_ids
from library.io_utils import CsvConfig
from library.normalize_assays import normalize_assays
from library.logging_utils import configure_logging
from library.config.chembl import load_chembl_assays_config

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
        "--config",
        default="config.yaml",
        help="Path to the YAML file providing the chembl_assays section",
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
        "--column", default="assay_chembl_id", help="Column containing assay IDs"
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
    parsed_args = parser.parse_args(args)

    try:
        cfg = load_chembl_assays_config(parsed_args.config)
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


def _load_existing_metadata(meta_path: Path) -> dict[str, Any]:
    """Load and normalise metadata from previous runs if available."""

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
    """Load validation errors persisted by a previous run."""

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
    """Consume and delete a temporary validation error file."""

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
    """Advance ``loader`` past ``last_processed`` for resumable downloads."""

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

    for identifier in iterator:
        yield identifier


def _prepare_assay_chunk(
    chunk: pd.DataFrame,
    *,
    list_format: ListFormat,
    errors_path: Path,
    error_accumulator: list[dict[str, Any]],
    ordered_columns: list[str] | None,
) -> tuple[pd.DataFrame, list[str] | None, str | None]:
    """Normalise, validate, and serialise a chunk of assay records."""

    if chunk.empty:
        return pd.DataFrame(), ordered_columns, None

    processed = postprocess_assays(chunk)
    normalised = normalize_assays(processed)

    temp_errors = errors_path.with_name(f"{errors_path.name}.part")
    validated = validate_assays(normalised, errors_path=temp_errors)
    error_accumulator.extend(_consume_error_file(temp_errors))

    if validated.empty:
        return pd.DataFrame(), ordered_columns, None

    if ordered_columns is None:
        schema_columns = [
            column
            for column in AssaysSchema.ordered_columns()
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
        for column in ["document_chembl_id", "target_chembl_id", "assay_chembl_id"]
        if column in prepared.columns
    ]
    if sort_columns:
        prepared = prepared.sort_values(sort_columns).reset_index(drop=True)

    serialised = serialise_dataframe(prepared, list_format, inplace=True)

    last_identifier: str | None = None
    if "assay_chembl_id" in serialised.columns and not serialised.empty:
        last_identifier = str(serialised["assay_chembl_id"].iloc[-1])

    return serialised, ordered_columns, last_identifier



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

    list_format = cast(ListFormat, args.list_format)

    csv_cfg = CsvConfig(sep=args.sep, encoding=args.encoding, list_format=list_format)

    def id_loader() -> Iterable[str]:
        return read_ids(input_path, args.column, csv_cfg)

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    header_written = output_path.exists() and output_path.stat().st_size > 0
    if header_written and resume_id is None:
        header_written = False

    client = ChemblClient(
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
        rps=args.rps,
        retry_penalty_seconds=args.retry_penalty,
        user_agent=args.user_agent,
    )

    identifier_stream = _skip_processed_ids(id_loader, resume_id)

    command_sequence = command_parts if command_parts is not None else tuple(sys.argv)
    command = " ".join(shlex.quote(part) for part in command_sequence)
    config = _prepare_configuration(args)

    for chunk in stream_assays(client, identifier_stream, chunk_size=args.chunk_size):
        serialised, ordered_columns, chunk_last = _prepare_assay_chunk(
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


    write_cli_metadata(
        output_path,
        row_count=int(len(serialised)),
        column_count=int(len(serialised.columns)),
        namespace=args,
        command_parts=command_parts,
        meta_path=meta_path,
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
        write_meta_yaml(
            output_path,
            command=command,
            config=config,
            row_count=total_rows,
            column_count=column_count,
            meta_path=meta_path,
            extra=progress_payload,
        )

    if not header_written:
        LOGGER.warning("No assay data retrieved; writing empty output")
        empty_frame = pd.DataFrame(columns=AssaysSchema.ordered_columns())
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
        write_meta_yaml(
            output_path,
            command=command,
            config=config,
            row_count=0,
            column_count=column_count,
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
        write_meta_yaml(
            output_path,
            command=command,
            config=config,
            row_count=total_rows,
            column_count=column_count,
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
