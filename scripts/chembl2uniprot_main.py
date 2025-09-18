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

ROOT = Path(__file__).resolve().parents[1]


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
        default=None,
        help="Logging level (e.g. INFO, DEBUG)",
    )
    parser.add_argument(
        "--log-format",
        default=None,
        choices=["human", "json"],
        help="Logging output format",
    )
    parser.add_argument(
        "--sep",
        default=None,
        help="CSV field separator",
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help="File encoding for CSV input and output",
    )
    args = parser.parse_args(argv)

    schema = ROOT / "schemas" / "config.schema.json"
    runtime_overrides = {
        key: value
        for key, value in {
            "log_level": args.log_level,
            "log_format": args.log_format,
            "sep": args.sep,
            "encoding": args.encoding,
        }.items()
        if value is not None
    }
    if args.config:
        config_path = Path(args.config)
        schema_path = config_path.with_name("config.schema.json")
        output = map_chembl_to_uniprot(
            input_csv_path=Path(args.input),
            output_csv_path=Path(args.output) if args.output else None,
            config_path=config_path,
            schema_path=schema_path,
            **runtime_overrides,
        )
    else:
        cfg_path = ROOT / "config.yaml"
        output = map_chembl_to_uniprot(
            input_csv_path=Path(args.input),
            output_csv_path=Path(args.output) if args.output else None,
            config_path=cfg_path,
            schema_path=schema,
            config_section="chembl2uniprot",
            **runtime_overrides,
        )

    print(output)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
