"""Command line interface for :mod:`chembl2uniprot`.

Example
-------
Run the mapper on ``input.csv`` using ``config.yaml`` and write the result to
``output.csv``::

    python -m chembl2uniprot --input input.csv --output output.csv --config config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .mapping import map_chembl_to_uniprot


def main(argv: list[str] | None = None) -> None:
    """Entry point for the command line interface.

    Parameters
    ----------
    argv:
        Optional list of command line arguments.  When ``None`` the arguments
        are taken from :data:`sys.argv`.
    """

    parser = argparse.ArgumentParser(description="Map ChEMBL IDs to UniProt IDs")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=False, help="Path to output CSV file")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to YAML configuration file"
    )
    args = parser.parse_args(argv)

    output = map_chembl_to_uniprot(
        input_csv_path=Path(args.input),
        output_csv_path=Path(args.output) if args.output else None,
        config_path=Path(args.config),
    )
    print(output)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
