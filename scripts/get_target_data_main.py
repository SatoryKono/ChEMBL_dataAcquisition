"""Command line interface for downloading ChEMBL target metadata."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
LIB_DIR = ROOT / "library"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from chembl_targets import TargetConfig, fetch_targets  # noqa: E402

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_SEP = ","
DEFAULT_ENCODING = "utf-8-sig"


def main(argv: Sequence[str] | None = None) -> None:
    """Parse command-line arguments and run the target data download.

    Parameters
    ----------
    argv:
        Optional list of command line arguments. If not provided, `sys.argv`
        will be used.
    """
    parser = argparse.ArgumentParser(description="Download ChEMBL target data")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument(
        "--column", default="target_chembl_id", help="Column with target IDs"
    )
    parser.add_argument("--log-level", default=DEFAULT_LOG_LEVEL)
    parser.add_argument("--sep", default=DEFAULT_SEP)
    parser.add_argument("--encoding", default=DEFAULT_ENCODING)
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

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

    print(args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
