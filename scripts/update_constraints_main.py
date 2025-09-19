"""CLI for synchronizing the constraints file with the lock file."""

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
    """Parses command-line arguments for the synchronization tool.

    Args:
        argv: A sequence of command-line arguments. If None, `sys.argv` is used.

    Returns:
        An `argparse.Namespace` object containing the parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Generate constraints.txt from requirements.lock, ensuring that "
            "both files stay in sync."
        )
    )
    parser.add_argument(
        "--lock-file",
        default=Path("requirements.lock"),
        type=Path,
        help="Path to the fully pinned requirements file.",
    )
    parser.add_argument(
        "--constraints-file",
        default=Path("constraints.txt"),
        type=Path,
        help="Destination path for the generated constraints file.",
    )
    parser.add_argument(
        "--log-level",
        default=_DEFAULT_LOG_LEVEL,
        choices=sorted(_LOG_LEVELS),
        help="Logging verbosity (defaults to the LOG_LEVEL environment variable)",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    """Configures the logging framework for the CLI.

    Args:
        level: The logging level to set.
    """

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def main(argv: Sequence[str] | None = None) -> None:
    """The entry point for the constraints synchronization CLI.

    Args:
        argv: A sequence of command-line arguments. If None, `sys.argv` is used.
    """

    args = parse_args(argv)
    configure_logging(args.log_level)
    ensure_project_root()
    from library.dependency_tools import synchronize_constraints

    logger = logging.getLogger(__name__)
    try:
        synchronize_constraints(args.lock_file, args.constraints_file)
    except FileNotFoundError as exc:
        logger.error("Lock file '%s' could not be found", exc.filename)
        raise SystemExit(1) from exc
    except OSError as exc:
        logger.error(
            "Failed to write constraints file '%s': %s", args.constraints_file, exc
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
