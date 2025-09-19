"""Command line interface for downloading ChEMBL target metadata."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence

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
from library.io_utils import CsvConfig, write_rows  # noqa: E402
from library.logging_utils import configure_logging  # noqa: E402

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_SEP = ","
DEFAULT_ENCODING = "utf-8-sig"
DEFAULT_LOG_FORMAT = "human"
DEFAULT_INPUT = "input.csv"
DEFAULT_COLUMN = "target_chembl_id"

def _default_output_name(input_path: str) -> str:
    """Derive the default output file name from ``input_path``."""

    stem = Path(input_path).stem or "input"
    date_suffix = datetime.utcnow().strftime("%Y%m%d")
    return f"output_{stem}_{date_suffix}.csv"


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


def main(argv: Sequence[str] | None = None) -> None:
    """Parse command-line arguments and run the target data download."""

    args = parse_args(argv)
    configure_logging(args.log_level, log_format=args.log_format)

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} does not exist")

    output_candidate = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_name(_default_output_name(args.input))
    )
    output_path = ensure_output_dir(output_candidate)

    csv_cfg = CsvConfig(
        sep=args.sep, encoding=args.encoding, list_format=args.list_format
    )
    identifiers = list(read_ids(input_path, args.column, csv_cfg))

    target_cfg = TargetConfig(
        output_sep=args.sep,
        output_encoding=args.encoding,
        list_format=args.list_format,
    )
    result = fetch_targets(identifiers, target_cfg)
    serialised = serialise_dataframe(result, args.list_format)

    columns = list(serialised.columns) or list(target_cfg.columns)
    rows = serialised.to_dict(orient="records")
    write_rows(output_path, rows, columns, csv_cfg)

    meta_path, _, quality_base = resolve_cli_sidecar_paths(
        output_path,
        meta_output=args.meta_output,
    )
    analyze_table_quality(serialised, table_name=str(quality_base))

    write_cli_metadata(
        output_path,
        row_count=int(serialised.shape[0]),
        column_count=int(len(columns)),
        namespace=args,
        meta_path=meta_path,
    )

    print(output_path)


if __name__ == "__main__":  # pragma: no cover
    main()
