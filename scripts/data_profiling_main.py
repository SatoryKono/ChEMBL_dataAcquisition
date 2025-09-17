"""Command line interface for generating data profiling reports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    from _path_utils import ensure_project_root as _ensure_project_root

    _ensure_project_root()

from library.data_profiling import analyze_table_quality  # noqa: E402
from library.logging_utils import configure_logging  # noqa: E402

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "human"


def main(argv: Sequence[str] | None = None) -> None:
    """Run the profiling utility on a CSV file."""

    parser = argparse.ArgumentParser(
        description="Generate quality and correlation reports for a table"
    )
    parser.add_argument("--input", default="input.csv", help="Path to input CSV file")
    parser.add_argument(
        "--output-prefix",
        help=(
            "Prefix for output reports. Defaults to the input file stem in the current"
            " directory"
        ),
    )
    parser.add_argument(
        "--log-level", default=DEFAULT_LOG_LEVEL, help="Logging verbosity level"
    )
    parser.add_argument(
        "--log-format",
        default=DEFAULT_LOG_FORMAT,
        choices=("human", "json"),
        help="Logging output format (human or json)",
    )
    args = parser.parse_args(argv)

    configure_logging(args.log_level, log_format=args.log_format)

    table_name = args.output_prefix or str(Path(args.input).with_suffix(""))
    analyze_table_quality(args.input, table_name=table_name)


if __name__ == "__main__":  # pragma: no cover
    main()
