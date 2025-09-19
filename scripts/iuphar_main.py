"""CLI tool to map UniProt accessions to IUPHAR classifications."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

if __package__ in {None, ""}:
    from _path_utils import ensure_project_root as _ensure_project_root

    _ensure_project_root()

from library.config.contact import load_contact_config
from library.iuphar import ApiCfg, IUPHARData, RetryCfg, init_session
from library.logging_utils import configure_logging

DEFAULT_LOG_FORMAT = "human"


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        An `argparse.Namespace` object containing the parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="Path to _IUPHAR_target.csv")
    parser.add_argument("--family", required=True, help="Path to _IUPHAR_family.csv")
    parser.add_argument(
        "--input", default="input.csv", help="Input CSV with uniprot_id column"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: output_<input>_<YYYYMMDD>.csv)",
    )
    parser.add_argument("--sep", default=",", help="CSV delimiter")
    parser.add_argument("--encoding", default="utf-8", help="File encoding")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument(
        "--log-format",
        default=DEFAULT_LOG_FORMAT,
        choices=("human", "json"),
        help="Logging output format (human or json)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file providing contact details",
    )
    return parser.parse_args()


def configure_http_session(config_path: str | Path) -> None:
    """Initialise the shared HTTP session using contact information."""

    try:
        contact_cfg = load_contact_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Failed to load contact configuration: {exc}") from exc

    init_session(ApiCfg(user_agent=contact_cfg.effective_user_agent()), RetryCfg())


def main() -> None:
    """The main entry point for the script."""

    args = parse_args()
    configure_logging(args.log_level, log_format=args.log_format)
    configure_http_session(args.config)
    if args.output is None:
        date = dt.date.today().strftime("%Y%m%d")
        stem = Path(args.input).stem
        args.output = f"output_{stem}_{date}.csv"

    data = IUPHARData.from_files(args.target, args.family, encoding=args.encoding)
    data.map_uniprot_file(
        args.input,
        args.output,
        encoding=args.encoding,
        sep=args.sep,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
