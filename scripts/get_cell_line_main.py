"""Command line interface for downloading ChEMBL cell line metadata."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
LIB_DIR = ROOT / "library"
if __package__ in {None, ""}:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(LIB_DIR) not in sys.path:
        sys.path.insert(0, str(LIB_DIR))

from library.chembl_cell_lines import (  # noqa: E402
    CellLineClient,
    CellLineConfig,
    CellLineError,
)
from library.logging_utils import configure_logging  # noqa: E402

LOGGER = logging.getLogger(__name__)

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_SEP = ","
DEFAULT_ENCODING = "utf-8"
DEFAULT_COLUMN = "cell_chembl_id"


def _iter_unique(values: Iterable[str]) -> Iterable[str]:
    """Yield unique values from ``values`` while preserving order."""

    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            yield value


def _load_ids_from_csv(path: Path, column: str, sep: str, encoding: str) -> list[str]:
    """Return cell line identifiers from ``column`` in ``path``."""

    import pandas as pd

    df = pd.read_csv(path, sep=sep, encoding=encoding)
    if column not in df.columns:
        raise ValueError(f"Missing column '{column}' in {path}")
    series = df[column].dropna()
    return [str(item).strip() for item in series if str(item).strip()]


def _run(args: argparse.Namespace) -> None:
    """Fetch requested cell line metadata and persist it to disk."""

    if not args.sep:
        raise ValueError("--sep must be a non-empty delimiter")
    if not args.encoding:
        raise ValueError("--encoding must be provided")
    if not args.column or not str(args.column).strip():
        raise ValueError("--column must name a non-empty column")

    ids = list(args.cell_line_ids)
    if args.input:
        input_path = Path(args.input).expanduser()
        if not input_path.exists() or not input_path.is_file():
            raise FileNotFoundError(f"Input file {input_path.resolve()} does not exist")
        ids.extend(
            _load_ids_from_csv(
                input_path.resolve(), args.column, args.sep, args.encoding
            )
        )

    ids = list(_iter_unique(i for i in ids if i))
    if not ids:
        raise ValueError("No cell line identifiers provided")

    output_path = (
        Path(
            args.output
            if args.output
            else f"output_input_{datetime.utcnow():%Y%m%d}.json"
        )
        .expanduser()
        .resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = CellLineClient(
        CellLineConfig(
            base_url=args.base_url,
        )
    )
    with output_path.open("w", encoding=args.encoding) as handle:
        for identifier in ids:
            record = client.fetch_cell_line(identifier)
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    print(output_path)


def main(argv: Sequence[str] | None = None) -> None:
    """Parses command-line arguments and downloads cell line records.

    Args:
        argv: A sequence of command-line arguments. If None, `sys.argv` is used.
    """

    parser = argparse.ArgumentParser(
        description="Download metadata for one or more ChEMBL cell lines",
    )
    parser.add_argument(
        "--cell-line-id",
        dest="cell_line_ids",
        action="append",
        default=[],
        help="ChEMBL cell line identifier to fetch (repeat for multiple IDs)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="CSV file containing cell line identifiers",
    )
    parser.add_argument(
        "--column",
        default=DEFAULT_COLUMN,
        help=f"Name of the column containing identifiers (default: {DEFAULT_COLUMN})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the JSON lines output",
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        help="Logging level",
    )
    parser.add_argument(
        "--sep",
        default=DEFAULT_SEP,
        help="Column separator for CSV input",
    )
    parser.add_argument(
        "--encoding",
        default=DEFAULT_ENCODING,
        help="File encoding for CSV input",
    )
    parser.add_argument(
        "--base-url",
        default=CellLineConfig.base_url,
        help="Override the ChEMBL API base URL",
    )
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    try:
        _run(args)
    except (FileNotFoundError, ValueError, CellLineError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception("Unexpected error while downloading cell line metadata")
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
