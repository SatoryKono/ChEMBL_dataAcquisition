# ChEMBL data acquisition

A small utility for mapping [ChEMBL](https://www.ebi.ac.uk/chembl/) identifiers to
[UniProt](https://www.uniprot.org/) accessions.  The mapping is performed via the
UniProt ID mapping service and the result is written as a new column to the input
CSV file.  The project exposes a reusable Python library and a command line
interface.

## Requirements

- Python >= 3.10 (tested with Python 3.12)
- pandas >= 1.5
- requests >= 2.31
- PyYAML >= 6.0
- jsonschema >= 4.17
- tqdm >= 4.66
- tenacity >= 8.2

Development and testing additionally require:

- pytest >= 7.4
- requests-mock >= 1.11
- black >= 23.0
- ruff >= 0.1
- mypy >= 1.4

## Installation

Create and activate a virtual environment and install the package along with
development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows use `.venv\Scripts\activate`
pip install -e .[dev]
```

## Project structure

```
chembl2uniprot/      # Library with configuration and mapping utilities
    __main__.py      # CLI entry point: ``python -m chembl2uniprot``
    config.py        # YAML loading and validation
    mapping.py       # Core mapping logic
    default_config.yaml  # Built-in configuration
    config.schema.json   # JSON schema for configuration validation
tests/
    data/            # Sample config and CSV files used in tests
    test_mapping.py  # Unit tests
pyproject.toml       # Build system and dependency declaration
mypy.ini             # Type checker configuration
```

## Usage

1. The package ships with a default configuration so the mapper can be run out of
   the box:

   ```bash
   python -m chembl2uniprot --input input.csv --output output.csv
   ```

   To use a custom configuration file (see ``tests/data/config/valid.yaml`` for an
   example):

   ```bash
   python -m chembl2uniprot --input input.csv --output output.csv --config config.yaml
   ```

When ``--output`` is omitted the result is written next to ``input.csv`` with the
suffix ``_with_uniprot.csv``.  The mapped UniProt identifiers are stored in a new
column defined by the configuration file.

The mapping function can also be used programmatically:

```python
from chembl2uniprot import map_chembl_to_uniprot
map_chembl_to_uniprot("input.csv", "output.csv", "config.yaml")
```

## Testing and quality checks

After making changes, run the following tools:

```bash
ruff .
black --check .
python -m mypy .
pytest
```

## License

This project is provided under the MIT license.
