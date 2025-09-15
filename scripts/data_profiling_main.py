"""Command line interface for generating data profiling reports."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from library.data_profiling import analyze_table_quality

DEFAULT_LOG_LEVEL = "INFO"


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
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    table_name = args.output_prefix or str(Path(args.input).with_suffix(""))
    analyze_table_quality(args.input, table_name=table_name)


if __name__ == "__main__":  # pragma: no cover
    main()
