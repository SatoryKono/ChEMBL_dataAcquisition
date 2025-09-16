"""Command line interface for :mod:`chembl2uniprot`.

Example
-------
Run the mapper on ``input.csv`` using the project's bundled configuration and
write the result to ``output.csv``::

    python scripts/chembl2uniprot_main.py --input input.csv --output output.csv

To use a standalone configuration file ``my_config.yaml``::

    python scripts/chembl2uniprot_main.py \
        --input input.csv \
        --output output.csv \
        --config my_config.yaml \
        --log-level INFO \
        --sep , \
        --encoding utf-8
"""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in {None, ""}:
    from _path_utils import ensure_project_root as _ensure_project_root

    _ensure_project_root()

from chembl2uniprot.mapping import map_chembl_to_uniprot  # noqa: E402
from library.logging_utils import configure_logging  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_SEP = ","
DEFAULT_ENCODING = "utf-8"
DEFAULT_LOG_FORMAT = "human"


def main(argv: list[str] | None = None) -> None:
    """Entry point for the command line interface.

    Parameters
    ----------
    argv:
        Optional list of command line arguments. When ``None`` the arguments
        provided via the command line are used.
    """

    parser = argparse.ArgumentParser(description="Map ChEMBL IDs to UniProt IDs")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=False, help="Path to output CSV file")
    parser.add_argument(
        "--config",
        required=False,
        help="Path to YAML configuration file; defaults to the project config",
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        help="Logging level (e.g. INFO, DEBUG)",
    )
    parser.add_argument(
        "--log-format",
        default=DEFAULT_LOG_FORMAT,
        choices=["human", "json"],
        help="Logging output format",
    )
    parser.add_argument(
        "--sep",
        default=DEFAULT_SEP,
        help="CSV field separator",
    )
    parser.add_argument(
        "--encoding",
        default=DEFAULT_ENCODING,
        help="File encoding for CSV input and output",
    )
    args = parser.parse_args(argv)

    configure_logging(args.log_level, log_format=args.log_format)

    schema = ROOT / "schemas" / "config.schema.json"
    if args.config:
        config_path = Path(args.config)
        schema_path = config_path.with_name("config.schema.json")
        output = map_chembl_to_uniprot(
            input_csv_path=Path(args.input),
            output_csv_path=Path(args.output) if args.output else None,
            config_path=config_path,
            schema_path=schema_path,
        )
    else:
        cfg_path = ROOT / "config.yaml"
        output = map_chembl_to_uniprot(
            input_csv_path=Path(args.input),
            output_csv_path=Path(args.output) if args.output else None,
            config_path=cfg_path,
            schema_path=schema,
            config_section="chembl2uniprot",
        )

    print(output)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
