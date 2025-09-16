"""Command line interface for generating data profiling reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from library.data_profiling import analyze_table_quality  # noqa: E402
from library.logging_utils import configure_logging  # noqa: E402

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

    configure_logging(args.log_level)

    table_name = args.output_prefix or str(Path(args.input).with_suffix(""))
    analyze_table_quality(args.input, table_name=table_name)


if __name__ == "__main__":  # pragma: no cover
    main()
