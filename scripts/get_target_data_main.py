"""Command line interface for downloading ChEMBL target metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    from _path_utils import ensure_project_root as _ensure_project_root

    _ensure_project_root()

from chembl_targets import TargetConfig, fetch_targets  # noqa: E402
from data_profiling import analyze_table_quality  # noqa: E402
from library.logging_utils import configure_logging  # noqa: E402

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_SEP = ","
DEFAULT_ENCODING = "utf-8-sig"
DEFAULT_LOG_FORMAT = "human"


def main(argv: Sequence[str] | None = None) -> None:
    """Parses command-line arguments and runs the target data download.

    Args:
        argv: An optional list of command-line arguments. If None, the
            arguments are taken from `sys.argv`.
    """
    parser = argparse.ArgumentParser(description="Download ChEMBL target data")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument(
        "--column", default="target_chembl_id", help="Column with target IDs"
    )
    parser.add_argument("--log-level", default=DEFAULT_LOG_LEVEL)
    parser.add_argument(
        "--log-format",
        default=DEFAULT_LOG_FORMAT,
        choices=("human", "json"),
        help="Logging output format (human or json)",
    )
    parser.add_argument("--sep", default=DEFAULT_SEP)
    parser.add_argument("--encoding", default=DEFAULT_ENCODING)
    args = parser.parse_args(argv)

    configure_logging(args.log_level, log_format=args.log_format)

    import pandas as pd

    df = pd.read_csv(args.input, sep=args.sep, encoding=args.encoding)
    if args.column not in df.columns:
        raise ValueError(f"missing column {args.column}")
    ids = df[args.column].dropna().astype(str).tolist()

    cfg = TargetConfig(output_sep=args.sep, output_encoding=args.encoding)
    result = fetch_targets(ids, cfg)
    result.to_csv(
        args.output, index=False, sep=cfg.output_sep, encoding=cfg.output_encoding
    )
    analyze_table_quality(result, table_name=str(Path(args.output).with_suffix("")))

    print(args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
