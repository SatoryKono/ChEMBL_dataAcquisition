# ChEMBL Tissue Retrieval Utilities

This document describes the reusable library function and accompanying command
line interface for downloading tissue metadata from the public
[ChEMBL](https://www.ebi.ac.uk/chembl/) API.

## Dependencies

The functionality relies on the following project dependencies (minimum
versions match those declared in `pyproject.toml`):

- Python 3.10+
- `requests` ≥ 2.31
- `requests-cache` ≥ 1.0 (optional, enables local HTTP caching)
- `tenacity` ≥ 8.2
- `pandas` ≥ 1.5 (only needed when working with other project components)

For development and testing we recommend installing the optional dependencies:

- `pytest` ≥ 7.4
- `requests-mock` ≥ 1.11
- `black` ≥ 23.0
- `ruff` ≥ 0.1
- `mypy` ≥ 1.4

## Installation

Create a virtual environment and install the project in editable mode together
with the development extras:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

The CLI uses the project's logging helpers which automatically integrate with
the configured logging format.

## Library Usage

The module `library.chembl_tissue_client` exposes the high level helper
`fetch_tissue_record`:

```python
from library.chembl_tissue_client import fetch_tissue_record

record = fetch_tissue_record("CHEMBL3988026")
print(record["pref_name"])  # "Uterine cervix"
```

`fetch_tissue_record` accepts optional `TissueConfig` and `HttpClient`
instances.  Use these parameters to customise timeouts, caching behaviour and
rate limiting.  When fetching multiple identifiers you may prefer the
`fetch_tissues` helper which deduplicates identifiers and returns a list of
payload dictionaries.

## Command Line Interface

The script `scripts/chembl_tissue_main.py` wraps the library helper in a
repeatable CLI.  It can download a single tissue record or process a CSV file
containing multiple identifiers.

### Basic Usage

Fetch a single record and print the location of the generated JSON file:

```bash
python scripts/chembl_tissue_main.py \
    --chembl-id CHEMBL3988026 \
    --output data/output/tissue.json
```

Process a CSV file with a column named `tissue_chembl_id` and skip identifiers
that are not available in ChEMBL:

```bash
python scripts/chembl_tissue_main.py \
    --input data/tissues.csv \
    --output data/output/tissues.json \
    --skip-missing
```

### Configuration Flags

- `--log-level`: logging verbosity (default `INFO`).
- `--log-format`: either `human` or `json`.
- `--sep` / `--encoding`: configure the CSV parser.
- `--base-url`, `--timeout`, `--max-retries`, `--rps`: network tuning options.
- `--cache-path` and `--cache-ttl`: enable persistent HTTP caching.
- `--skip-missing`: continue when identifiers are invalid or not present.

If no `--output` path is provided, the script writes to
`output_<input_name>_<YYYYMMDD>.json` in the current working directory.

## Running Tests

Execute the targeted unit tests via `pytest`:

```bash
pytest tests/test_chembl_tissue_client.py tests/test_chembl_tissue_main.py
```

The complete project test suite (`pytest`), formatter (`black`), linter
(`ruff`) and type checker (`mypy`) should all succeed before submitting
changes:

```bash
pytest
black --check .
ruff check .
mypy .
```
