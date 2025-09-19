"""CLI for downloading and normalising ChEMBL molecule metadata."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence, cast

import pandas as pd

if __package__ in {None, ""}:
    from _path_utils import ensure_project_root as _ensure_project_root

    _ensure_project_root()

from library.cli_common import (
    ensure_output_dir,
    resolve_cli_sidecar_paths,
    serialise_dataframe,
    write_cli_metadata,
)
from library.chembl_client import ChemblClient
from library.chembl_library import get_testitems
from library.data_profiling import analyze_table_quality
from library.io import read_ids
from library.io_utils import CsvConfig, serialise_cell
from library.normalize_testitems import normalize_testitems
from library.testitem_library import (
    PUBCHEM_BASE_URL,
    PUBCHEM_DEFAULT_BACKOFF,
    PUBCHEM_DEFAULT_MAX_RETRIES,
    PUBCHEM_DEFAULT_RETRY_PENALTY,
    PUBCHEM_DEFAULT_RPS,
    PUBCHEM_PROPERTY_COLUMNS,
    add_pubchem_data,
)
from library.testitem_validation import TestitemsSchema, validate_testitems
from library.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)
DEFAULT_LOG_FORMAT = "human"

REQUIRED_ENRICHED_COLUMNS: tuple[str, ...] = (
    "salt_chembl_id",
    "parent_chembl_id",
    *PUBCHEM_PROPERTY_COLUMNS,
)


def _default_output_name(input_path: str) -> str:
    stem = Path(input_path).stem or "output"
    date_suffix = datetime.now().strftime("%Y%m%d")
    return f"output_{stem}_{date_suffix}.csv"


def _serialise_value(value: object, list_format: str) -> object:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, list):
        if list_format == "pipe":
            return "|".join(
                json.dumps(item, ensure_ascii=False, sort_keys=True) for item in value
            )
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _serialise_complex_columns(df: pd.DataFrame, list_format: str) -> pd.DataFrame:
    result = df.copy()
    for column in result.columns:
        result[column] = result[column].map(
            lambda value: serialise_cell(value, list_format)
        )
    return result


def _ensure_output_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Ensure ``df`` exposes ``columns`` by filling missing ones with ``pd.NA``.

    The helper returns the original ``DataFrame`` when all requested columns are
    already present; otherwise it returns a shallow copy where the missing
    columns have been initialised with ``pd.NA`` values.  This behaviour keeps
    downstream validation deterministic by guaranteeing that optional
    descriptors such as ``salt_chembl_id`` and the PubChem enrichment fields
    always exist, even if the upstream APIs did not return data for them.
    """

    missing = [column for column in columns if column not in df.columns]
    if not missing:
        return df
    enriched = df.copy()
    for column in missing:
        enriched[column] = pd.NA
    return enriched


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments for the molecule pipeline CLI.

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
        "--column", default="molecule_chembl_id", help="Column containing molecule IDs"
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
        "--smiles-column",
        default="canonical_smiles",
        help="Column containing SMILES strings for PubChem enrichment",
    )
    parser.add_argument(
        "--pubchem-timeout",
        type=float,
        default=10.0,
        help="PubChem HTTP timeout in seconds",
    )
    parser.add_argument(
        "--pubchem-max-retries",
        type=int,
        default=PUBCHEM_DEFAULT_MAX_RETRIES,
        help="Maximum retry attempts for PubChem requests",
    )
    parser.add_argument(
        "--pubchem-rps",
        type=float,
        default=PUBCHEM_DEFAULT_RPS,
        help="Maximum PubChem requests per second",
    )
    parser.add_argument(
        "--pubchem-backoff",
        type=float,
        default=PUBCHEM_DEFAULT_BACKOFF,
        help="Exponential backoff multiplier for PubChem retries",
    )
    parser.add_argument(
        "--pubchem-retry-penalty",
        type=float,
        default=PUBCHEM_DEFAULT_RETRY_PENALTY,
        help="Additional sleep in seconds applied after PubChem retries",
    )
    parser.add_argument(
        "--pubchem-base-url",
        default=PUBCHEM_BASE_URL,
        help="PubChem PUG REST base URL",
    )
    parser.add_argument(
        "--pubchem-user-agent",
        default="ChEMBLDataAcquisition/1.0",
        help="User-Agent header used for PubChem requests",
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
) -> Iterator[str]:
    return cast(Iterator[str], read_ids(path, column, cfg, limit=limit))


def run_pipeline(
    args: argparse.Namespace, *, command_parts: Sequence[str] | None = None
) -> int:
    """Executes the molecule acquisition pipeline with the given arguments.

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

    molecule_ids = _limited_ids(input_path, args.column, csv_cfg, args.limit)

    with ChemblClient(
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
        rps=args.rps,
        user_agent=args.user_agent,
    ) as client:
        molecules_df = get_testitems(client, molecule_ids, chunk_size=args.chunk_size)
    if molecules_df.empty:
        LOGGER.warning("No molecule data retrieved; writing empty output")
        molecules_df = pd.DataFrame(columns=TestitemsSchema.ordered_columns())

    normalised = normalize_testitems(molecules_df)
    pubchem_http_client_config = {
        "max_retries": args.pubchem_max_retries,
        "rps": args.pubchem_rps,
        "backoff_multiplier": args.pubchem_backoff,
        "retry_penalty_seconds": args.pubchem_retry_penalty,
    }
    LOGGER.debug(
        "Using PubChem HTTP client configuration: %s", pubchem_http_client_config
    )

    enriched = add_pubchem_data(
        normalised,
        smiles_column=args.smiles_column,
        timeout=args.pubchem_timeout,
        base_url=args.pubchem_base_url,
        user_agent=args.pubchem_user_agent,
        http_client_config=pubchem_http_client_config,
    )
    enriched = _ensure_output_columns(enriched, REQUIRED_ENRICHED_COLUMNS)
    validated = validate_testitems(enriched, errors_path=errors_path)
    validated = _ensure_output_columns(validated, REQUIRED_ENRICHED_COLUMNS)

    schema_columns = [
        column
        for column in TestitemsSchema.ordered_columns()
        if column in validated.columns
    ]
    extra_columns = sorted(
        [column for column in validated.columns if column not in schema_columns]
    )
    ordered_columns = schema_columns + extra_columns
    if ordered_columns:
        validated = validated[ordered_columns]

    sort_columns = []
    if "salt_chembl_id" in validated.columns:
        sort_columns.append("salt_chembl_id")
    if "molecule_chembl_id" in validated.columns:
        sort_columns.append("molecule_chembl_id")
    if sort_columns:
        validated = validated.sort_values(sort_columns, na_position="last").reset_index(
            drop=True
        )

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

    LOGGER.info("Molecule table written to %s", output_path)
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
