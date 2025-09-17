"""CLI for downloading and normalising ChEMBL molecule metadata."""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

if __package__ in {None, ""}:
    from _path_utils import ensure_project_root as _ensure_project_root

    _ensure_project_root()

from library.chembl_client import ChemblClient
from library.chembl_library import get_testitems
from library.data_profiling import analyze_table_quality
from library.io import read_ids
from library.io_utils import CsvConfig, serialise_cell
from library.metadata import write_meta_yaml
from library.normalize_testitems import normalize_testitems
from library.testitem_library import (
    PUBCHEM_BASE_URL,
    PUBCHEM_DEFAULT_BACKOFF,
    PUBCHEM_DEFAULT_MAX_RETRIES,
    PUBCHEM_DEFAULT_RETRY_PENALTY,
    PUBCHEM_DEFAULT_RPS,
    add_pubchem_data,
)
from library.testitem_validation import TestitemsSchema, validate_testitems
from library.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)
DEFAULT_LOG_FORMAT = "human"


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


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the molecule pipeline CLI."""

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
        "--chunk-size", type=int, default=20, help="Number of IDs fetched per batch"
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
    return parser.parse_args(args)


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


def _limited_ids(
    path: Path, column: str, cfg: CsvConfig, limit: int | None
) -> Iterable[str]:
    return read_ids(path, column, cfg, limit=limit)


def run_pipeline(
    args: argparse.Namespace, *, command_parts: Sequence[str] | None = None
) -> int:
    """Execute the molecule acquisition pipeline with ``args``."""

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} does not exist")

    output_path = (
        Path(args.output) if args.output else Path(_default_output_name(args.input))
    )
    errors_path = (
        Path(args.errors_output)
        if args.errors_output
        else output_path.with_suffix(f"{output_path.suffix}.errors.json")
    )
    meta_path = Path(args.meta_output) if args.meta_output else None

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

    client = ChemblClient(
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
        rps=args.rps,
        user_agent=args.user_agent,
    )

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
    validated = validate_testitems(enriched, errors_path=errors_path)

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

    serialised = _serialise_complex_columns(validated, args.list_format)
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

    analyze_table_quality(serialised, table_name=str(output_path.with_suffix("")))

    LOGGER.info("Molecule table written to %s", output_path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by the CLI and tests."""

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
