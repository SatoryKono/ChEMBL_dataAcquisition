"""Command line interface for downloading ChEMBL target metadata."""

from __future__ import annotations

 
 
import csv
 
import logging
 
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Sequence

if __package__ in {None, ""}:
    from _path_utils import ensure_project_root as _ensure_project_root

    _ensure_project_root()

from chembl_targets import TargetConfig, fetch_targets  # noqa: E402
from library.cli_common import (  # noqa: E402
    analyze_table_quality,
    ensure_output_dir,
    resolve_cli_sidecar_paths,
    serialise_dataframe,
    write_cli_metadata,
)
from library.io import read_ids  # noqa: E402
from library.io_utils import CsvConfig  # noqa: E402
from library.logging_utils import configure_logging  # noqa: E402

LOGGER = logging.getLogger(__name__)

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_SEP = ","
DEFAULT_ENCODING = "utf-8-sig"
DEFAULT_LOG_FORMAT = "human"
DEFAULT_INPUT = "input.csv"
DEFAULT_COLUMN = "target_chembl_id"
STREAM_BATCH_SIZE = 200


def _default_output_name(input_path: str) -> str:
    """Derive the default output file name from ``input_path``."""

    stem = Path(input_path).stem or "input"
    date_suffix = datetime.utcnow().strftime("%Y%m%d")
    return f"output_{stem}_{date_suffix}.csv"


def _chunked(iterable: Iterable[str], size: int) -> Iterator[list[str]]:
    """Yield fixed-size chunks from ``iterable``.

    Parameters
    ----------
    iterable:
        The source iterable providing identifier values.
    size:
        Maximum number of identifiers to include in a single chunk. Must be a
        positive integer.

    Yields
    ------
    list[str]
        Lists containing up to ``size`` identifiers, preserving the original
        order.

    Raises
    ------
    ValueError
        If ``size`` is not a positive integer.
    """

    if size <= 0:
        msg = "size must be a positive integer"
        raise ValueError(msg)

    batch: list[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _write_header(path: Path, columns: Sequence[str], cfg: CsvConfig) -> None:
    """Write only the CSV header to ``path`` using ``columns`` order."""

    with path.open("w", encoding=cfg.encoding, newline="") as handle:
        writer = csv.writer(handle, delimiter=cfg.sep)
        writer.writerow(columns)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Create an argument parser and return parsed ``argv``."""

    parser = argparse.ArgumentParser(description="Download ChEMBL target data")
    parser.add_argument(
        "--input", default=DEFAULT_INPUT, help="Input CSV file containing identifiers"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV file. Defaults to output_<input>_<YYYYMMDD>.csv",
    )
    parser.add_argument(
        "--column",
        default=DEFAULT_COLUMN,
        help="Name of the column providing ChEMBL target identifiers",
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        help="Logging level (e.g. DEBUG, INFO)",
    )
    parser.add_argument(
        "--log-format",
        default=DEFAULT_LOG_FORMAT,
        choices=("human", "json"),
        help="Logging output format",
    )
    parser.add_argument(
        "--sep",
        default=DEFAULT_SEP,
        help="CSV delimiter used for reading input and writing output",
    )
    parser.add_argument(
        "--encoding",
        default=DEFAULT_ENCODING,
        help="Text encoding for input and output CSV files",
    )
    parser.add_argument(
        "--list-format",
        choices=("json", "pipe"),
        default="json",
        help="Serialisation format for list-like columns",
    )
    parser.add_argument(
        "--meta-output",
        default=None,
        help="Optional path for the generated .meta.yaml file",
    )
    return parser.parse_args(argv)


def _run(args: argparse.Namespace) -> None:
    """Execute the target metadata export using parsed CLI arguments."""

    if not args.column or not str(args.column).strip():
        raise ValueError("--column must name a non-empty column")
    if not args.sep:
        raise ValueError("--sep must be a non-empty delimiter")
    if not args.encoding:
        raise ValueError("--encoding must be provided")

    input_path = Path(args.input).expanduser()
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"Input file {input_path.resolve()} does not exist")
    input_path = input_path.resolve()

    output_candidate = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_name(_default_output_name(args.input))
    )
    output_path = ensure_output_dir(output_candidate)

    csv_cfg = CsvConfig(
        sep=args.sep, encoding=args.encoding, list_format=args.list_format
    )
 
    identifiers = read_ids(input_path, args.column, csv_cfg)
 
    try:
        identifiers = list(read_ids(input_path, args.column, csv_cfg))
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
 

    target_cfg = TargetConfig(
        output_sep=args.sep,
        output_encoding=args.encoding,
        list_format=args.list_format,
    )
    row_count = 0
    wrote_header = False
    columns: list[str] | None = None

    for batch in _chunked(identifiers, STREAM_BATCH_SIZE):
        batch_frame = fetch_targets(batch, target_cfg)
        serialised = serialise_dataframe(batch_frame, args.list_format, inplace=True)

        if columns is None:
            columns = list(serialised.columns) or list(target_cfg.columns)
        serialised = serialised.loc[:, columns]

        mode = "w" if not wrote_header else "a"
        serialised.to_csv(
            output_path,
            sep=args.sep,
            encoding=args.encoding,
            index=False,
            mode=mode,
            header=not wrote_header,
        )
        wrote_header = True
        row_count += int(serialised.shape[0])

    if columns is None:
        columns = list(target_cfg.columns)

    if not wrote_header:
        _write_header(output_path, columns, csv_cfg)

    meta_path, _, quality_base = resolve_cli_sidecar_paths(
        output_path,
        meta_output=args.meta_output,
    )
    analyze_table_quality(
        output_path,
        table_name=str(quality_base),
        separator=args.sep,
        encoding=args.encoding,
    )

    write_cli_metadata(
        output_path,
        row_count=row_count,
        column_count=int(len(columns)),
        namespace=args,
        meta_path=meta_path,
    )

    print(output_path)


def main(argv: Sequence[str] | None = None) -> None:
    """Parse command-line arguments and run the target data download."""

    args = parse_args(argv)
    configure_logging(args.log_level, log_format=args.log_format)

    try:
        _run(args)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception("Unexpected error while downloading target metadata")
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
