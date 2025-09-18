"""Command line interface for downloading ChEMBL tissue metadata.

The utility accepts either a single ChEMBL tissue identifier via
``--chembl-id`` or a CSV file containing multiple identifiers.  Retrieved
records are written as a JSON array while logging progress and validation
issues.

Algorithm Notes
---------------
1. Parse command line options and configure structured logging.
2. Collect tissue identifiers from CLI arguments and optional CSV input.
3. Iterate over the identifiers, calling
   :func:`library.chembl_tissue_client.fetch_tissue_record` for each entry.
4. Serialise the resulting list of dictionaries into a UTF-8 encoded JSON file.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import requests

ROOT = Path(__file__).resolve().parents[1]
LIB_DIR = ROOT / "library"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from library.chembl_tissue_client import (  # noqa: E402
    TissueConfig,
    TissueNotFoundError,
    create_http_client,
    fetch_tissue_record,
    normalise_tissue_id,
)
from library.http_client import CacheConfig  # noqa: E402
from library.logging_utils import configure_logging  # noqa: E402

LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT = "input.csv"
DEFAULT_ENCODING = "utf-8"
DEFAULT_SEP = ","
DEFAULT_COLUMN = "tissue_chembl_id"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "human"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RPS = 2.0
DEFAULT_BASE_URL = TissueConfig().base_url


def _build_parser() -> argparse.ArgumentParser:
    """Return the command line parser used by :func:`main`."""

    parser = argparse.ArgumentParser(
        description="Download ChEMBL tissue metadata and serialise as JSON.",
    )
    parser.add_argument(
        "--chembl-id",
        help="Single tissue ChEMBL identifier to retrieve.",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=(
            "Path to CSV file containing tissue identifiers. Defaults to"
            f" {DEFAULT_INPUT}."
        ),
    )
    parser.add_argument(
        "--column",
        default=DEFAULT_COLUMN,
        help="Name of the CSV column containing tissue identifiers.",
    )
    parser.add_argument(
        "--output",
        help=(
            "Destination JSON file. When omitted the file is named "
            "output_<input>_<YYYYMMDD>.json in the current directory."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        help="Logging level (e.g. INFO, DEBUG).",
    )
    parser.add_argument(
        "--log-format",
        choices=["human", "json"],
        default=DEFAULT_LOG_FORMAT,
        help="Logging output format.",
    )
    parser.add_argument(
        "--sep",
        default=DEFAULT_SEP,
        help="CSV delimiter used when reading --input.",
    )
    parser.add_argument(
        "--encoding",
        default=DEFAULT_ENCODING,
        help="Encoding of the input CSV file and the generated JSON output.",
    )
    parser.add_argument(
        "--dictionary",
        help=(
            "Optional dictionary file path (reserved for compatibility with"
            " other tools)."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL for the ChEMBL API.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum number of retry attempts for failed HTTP requests.",
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=DEFAULT_RPS,
        help="Requests per second limit for the ChEMBL API.",
    )
    parser.add_argument(
        "--cache-path",
        help="Enable persistent HTTP caching at the given filesystem path.",
    )
    parser.add_argument(
        "--cache-ttl",
        type=float,
        default=0.0,
        help="Time-to-live for cached responses in seconds (requires --cache-path).",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip identifiers that fail validation or are missing in ChEMBL.",
    )
    return parser


def _default_output_path(input_path: Path) -> Path:
    """Return the default output path derived from ``input_path``."""

    date_str = datetime.utcnow().strftime("%Y%m%d")
    stem = input_path.stem or "input"
    filename = f"output_{stem}_{date_str}.json"
    return Path(filename)


def _read_identifiers_from_csv(
    csv_path: Path,
    *,
    column: str,
    sep: str,
    encoding: str,
) -> list[str]:
    """Return identifiers from ``csv_path`` extracted from ``column``."""

    with csv_path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=sep)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(
                f"Input file {csv_path} does not contain required column {column!r}"
            )
        values: list[str] = []
        for row in reader:
            raw_value = row.get(column)
            if raw_value is None:
                continue
            cleaned = raw_value.strip()
            if cleaned:
                values.append(cleaned)
    return values


def _collect_identifiers(
    args: argparse.Namespace,
) -> list[str]:
    """Collect tissue identifiers from CLI arguments and optional CSV input."""

    identifiers: list[str] = []
    if args.chembl_id:
        identifiers.append(args.chembl_id)
    input_path = Path(args.input)
    if input_path.exists():
        identifiers.extend(
            _read_identifiers_from_csv(
                input_path, column=args.column, sep=args.sep, encoding=args.encoding
            )
        )
    elif not args.chembl_id:
        raise FileNotFoundError(
            f"Input file {input_path} not found and no --chembl-id provided"
        )
    return identifiers


def _build_cache_config(args: argparse.Namespace) -> CacheConfig | None:
    """Construct a :class:`CacheConfig` instance from CLI arguments."""

    if not args.cache_path:
        return None
    if args.cache_ttl <= 0:
        LOGGER.warning(
            "Ignoring cache configuration because --cache-ttl is not positive",
        )
        return None
    return CacheConfig(enabled=True, path=args.cache_path, ttl_seconds=args.cache_ttl)


def _write_output(
    records: Sequence[Mapping[str, object]],
    output_path: Path,
    *,
    encoding: str,
) -> None:
    """Serialise ``records`` to ``output_path`` as UTF-8 encoded JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding=encoding) as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Program entry point returning an exit status code."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level, log_format=args.log_format)

    if args.dictionary:
        LOGGER.info("Dictionary parameter is currently unused: %s", args.dictionary)

    try:
        raw_ids = _collect_identifiers(args)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    normalised_ids: list[str] = []
    seen: set[str] = set()
    for raw in raw_ids:
        try:
            tissue_id = normalise_tissue_id(raw)
        except ValueError as exc:
            if args.skip_missing:
                LOGGER.warning("Skipping invalid identifier %r: %s", raw, exc)
                continue
            LOGGER.error("Invalid identifier %r: %s", raw, exc)
            return 1
        if tissue_id in seen:
            LOGGER.debug("Skipping duplicate identifier %s", tissue_id)
            continue
        seen.add(tissue_id)
        normalised_ids.append(tissue_id)

    if not normalised_ids:
        LOGGER.error("No valid tissue identifiers supplied")
        return 1

    cache_config = _build_cache_config(args)
    config = TissueConfig(
        base_url=args.base_url,
        timeout_sec=args.timeout,
        max_retries=args.max_retries,
        rps=args.rps,
        cache=cache_config,
    )
    client = create_http_client(config)

    records: list[dict[str, object]] = []
    skipped = 0
    for tissue_id in normalised_ids:
        try:
            records.append(fetch_tissue_record(tissue_id, config=config, client=client))
        except TissueNotFoundError as exc:
            if args.skip_missing:
                LOGGER.warning("%s", exc)
                skipped += 1
                continue
            LOGGER.error("%s", exc)
            return 1
        except ValueError as exc:
            if args.skip_missing:
                LOGGER.warning("Skipping %s due to invalid payload: %s", tissue_id, exc)
                skipped += 1
                continue
            LOGGER.error("Validation failed for %s: %s", tissue_id, exc)
            return 1
        except requests.RequestException as exc:
            LOGGER.error("Network error while fetching %s: %s", tissue_id, exc)
            return 1

    if not records:
        LOGGER.error("No tissue records retrieved")
        return 1

    output_path = (
        Path(args.output) if args.output else _default_output_path(Path(args.input))
    )
    try:
        _write_output(records, output_path, encoding=args.encoding)
    except OSError as exc:
        LOGGER.error("Failed to write output %s: %s", output_path, exc)
        return 1

    LOGGER.info(
        "Wrote %d tissue records to %s%s",
        len(records),
        output_path,
        f" (skipped {skipped})" if skipped else "",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
