"""CLI for verifying dependency declarations against pinned constraints."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Sequence

from _path_utils import ensure_project_root

_DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments for the dependency verification CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate that pyproject.toml declares version bounds compatible with "
            "constraints.txt."
        )
    )
    parser.add_argument(
        "--pyproject",
        default=Path("pyproject.toml"),
        type=Path,
        help="Path to the pyproject.toml file to inspect.",
    )
    parser.add_argument(
        "--constraints-file",
        default=Path("constraints.txt"),
        type=Path,
        help="Path to the pinned constraints file.",
    )
    parser.add_argument(
        "--extras",
        nargs="*",
        default=("dev",),
        help=(
            "Optional extras that should also be validated. Pass the flag without "
            "values to skip extra validation."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=sorted(_LOG_LEVELS),
        default=_DEFAULT_LOG_LEVEL,
        help="Logging verbosity (defaults to the LOG_LEVEL environment variable).",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    """Configures logging for the CLI."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the dependency verification CLI."""

    args = parse_args(argv)
    configure_logging(args.log_level)
    ensure_project_root()
    from library.dependency_tools import validate_pyproject_constraints

    logger = logging.getLogger(__name__)
    extras: Sequence[str] | None = args.extras
    if args.extras == []:
        extras = ()
    try:
        validate_pyproject_constraints(
            args.pyproject, args.constraints_file, extras=extras
        )
    except FileNotFoundError as exc:
        logger.error("Required file '%s' could not be found.", exc.filename)
        raise SystemExit(1) from exc
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
