"""Command line interface for :mod:`chembl2uniprot`.

Example
-------
Run the mapper on ``input.csv`` using the built-in default configuration and
write the result to ``output.csv``::

    python -m chembl2uniprot --input input.csv --output output.csv

To use a custom configuration file ``config.yaml``::

    python -m chembl2uniprot --input input.csv --output output.csv --config config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from importlib import resources

from .mapping import map_chembl_to_uniprot


def main(argv: list[str] | None = None) -> None:
    """Run the CLI with ``argv`` arguments."""

    parser = argparse.ArgumentParser(description="Map ChEMBL IDs to UniProt IDs")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=False, help="Path to output CSV file")
    parser.add_argument(
        "--config",
        required=False,
        help="Path to YAML configuration file; defaults to a built-in config",
    )
    args = parser.parse_args(argv)

    if args.config:
        config_path = Path(args.config)
        output = map_chembl_to_uniprot(
            input_csv_path=Path(args.input),
            output_csv_path=Path(args.output) if args.output else None,
            config_path=config_path,
        )
    else:
        # Fall back to the package's default configuration file.
        with resources.as_file(
            resources.files("chembl2uniprot") / "default_config.yaml"
        ) as cfg_path:
            output = map_chembl_to_uniprot(
                input_csv_path=Path(args.input),
                output_csv_path=Path(args.output) if args.output else None,
                config_path=cfg_path,
            )

    print(output)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
