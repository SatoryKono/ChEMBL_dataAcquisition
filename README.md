# Data Acquisition and Normalization Pipeline

This repository contains a suite of Python scripts and a reusable library for acquiring, mapping, and normalizing biological data from various sources, including ChEMBL, UniProt, HGNC, and the IUPHAR/BPS Guide to PHARMACOLOGY (GtoPdb).

The primary goal of this project is to provide a deterministic and configurable pipeline for creating a unified dataset of protein targets, their classifications, and related annotations.

## Features

*   **Modular Architecture:** The project is divided into a `library` of reusable components and a set of `scripts` that provide command-line interfaces for common tasks.
*   **Comprehensive Data Acquisition:** Fetch data from multiple sources, including ChEMBL, UniProt, HGNC, and GtoPdb.
*   **Robust ID Mapping:** Map identifiers between different databases (e.g., ChEMBL to UniProt, UniProt to HGNC).
*   **Data Normalization:** Normalize and enrich data to produce a clean, consistent, and deterministic dataset.
*   **Protein Classification:** Classify proteins into a hierarchical system based on signals from UniProt and IUPHAR.
*   **Configurable:** The pipeline's behavior can be customized through a central YAML configuration file.
*   **Extensible:** The modular design makes it easy to add new data sources or processing steps.

## Requirements

The project targets Python 3.12. [`requirements.lock`](requirements.lock) is the
single source of truth for fully pinned dependencies. The compact
[`constraints.txt`](constraints.txt) file is generated from the lock file and is
used during installation to apply the exact versions resolved by `pip-compile`.
`pyproject.toml` mirrors these versions by declaring the pinned releases as the
minimum supported versions.

### Runtime dependencies

| Package        | Pinned version |
| -------------- | -------------- |
| pandas         | 2.3.2          |
| requests       | 2.32.5         |
| PyYAML         | 6.0.2          |
| jsonschema     | 4.25.1         |
| tqdm           | 4.67.1         |
| tenacity       | 9.1.2          |
| pydantic       | 2.11.9         |
| requests-cache | 1.2.1          |
| packaging      | 25.0           |

### Development dependencies

| Package         | Pinned version |
| --------------- | -------------- |
| pytest          | 8.4.2          |
| requests-mock   | 1.12.1         |
| hypothesis      | 6.138.17       |
| black           | 25.1.0         |
| ruff            | 0.13.0         |
| mypy            | 1.18.1         |
| pandas-stubs    | 2.3.2.250827   |
| types-PyYAML    | 6.0.12.20250915 |
| types-requests  | 2.32.4.20250913 |
| types-jsonschema| 4.25.1.20250822 |
| types-pytz      | 2025.2.0.20250809 |

To refresh these pins run `pip-compile` to update `requirements.lock`, regenerate
the short constraints file, and verify that `pyproject.toml` still reflects the
resolved versions:

```bash
pip-compile --resolver=backtracking pyproject.toml --output-file requirements.lock
python scripts/update_constraints_main.py
python scripts/check_dependency_versions_main.py
```

`pip-compile` is part of `pip-tools` and can be installed in a dedicated update
environment. The verification step ensures that the lower bounds in
`pyproject.toml` match the pinned versions in `constraints.txt`.

## Installation

1.  Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

2.  Upgrade ``pip`` and install the project with the pinned dependencies:
    ```bash
    python -m pip install --upgrade pip
    pip install --constraint constraints.txt .[dev]
    ```
    Omit ``.[dev]`` if you only need the runtime dependencies.

    To confirm that the installation metadata remains in sync with the lock
    files, run:

    ```bash
    python scripts/check_dependency_versions_main.py
    ```

3.  Install the pre-commit hooks to ensure consistent formatting, linting, type
    checking, and tests before each commit:
    ```bash
    pre-commit install
    ```

## Testing

The project ships with an extensive pytest suite that exercises the library
modules and command-line entry points. To run all tests locally use:

```bash
pytest -q
```

During development it can be helpful to abort quickly on the first failure and
inspect the slowest tests:

```bash
pytest --maxfail=1 --durations=10
```

## Code Quality Checks

Static analysis and formatting are enforced via pre-commit hooks. To run the
individual tools manually execute:

```bash
ruff check .
ruff format .  # or `black .` to match the configured formatter
mypy --strict --ignore-missing-imports .
pytest
```

Alternatively, execute all checks in one go with:

```bash
pre-commit run --all-files
```

## Project Structure

The repository is organized as follows:

```
├── config.yaml                # Main configuration file
├── data/                      # Input and output data
├── docs/                      # Project documentation
├── library/                   # Core Python library
│   ├── chembl2uniprot/        # ChEMBL to UniProt mapping
│   ├── uniprot_enrich/        # UniProt data enrichment
│   ├── ...                    # Other library modules
├── schemas/                   # JSON schemas for configuration validation
├── scripts/                   # Command-line interface scripts
├── tests/                     # Unit and integration tests
├── pyproject.toml             # Build system and dependency declaration
└── README.md                  # This file
```

## Usage

The pipeline is primarily driven by the scripts in the `scripts/` directory. Each script provides a command-line interface for a specific task.

### Configuration

The pipeline's behavior is controlled by the `config.yaml` file. This file contains settings for API endpoints, network parameters, data processing options, and output formats. A JSON schema for this file is provided in `schemas/config.schema.json`.

#### Environment variable overrides

Configuration values can be overridden at runtime via environment variables. The recommended pattern prefixes variables with `CHEMBL_DA__` followed by the uppercase configuration path where nested keys are separated by double underscores. For example, to increase the retry budget for the bundled `chembl2uniprot` configuration section you can run:

```bash
export CHEMBL_DA__CHEMBL2UNIPROT__RETRY__MAX_ATTEMPTS=8
export CHEMBL_DA__CHEMBL2UNIPROT__RETRY__BACKOFF_SEC=2
python scripts/chembl2uniprot_main.py --input data/input/targets.csv
```

The loader is case-insensitive and coerces primitive values automatically, so the strings above are parsed as integers and floats. When working with standalone configuration files that only contain the mapping schema (for example copies of `tests/data/config/valid.yaml`), the legacy `CHEMBL_` prefix remains supported:

```bash
export CHEMBL_BATCH__SIZE=10  # Equivalent to CHEMBL_DA__BATCH__SIZE
python scripts/chembl2uniprot_main.py --config my_config.yaml
```

The unified target pipeline honours the same convention. For instance, to switch the serialisation format without editing `config.yaml` use:

```bash
export CHEMBL_DA__PIPELINE__LIST_FORMAT=pipe
export CHEMBL_DA__PIPELINE__IUPHAR__APPROVED_ONLY=true
python scripts/pipeline_targets_main.py --input data/input/targets.csv --output data/output/final_targets.csv
```

Always define the environment variables in the shell session before launching the CLI so that the overrides are visible to the Python process.

#### HTTP retry configuration

Network-bound scripts rely on `library.http_client.HttpClient`, which retries transient HTTP errors listed in `status_forcelist`. The default value, exposed as `library.http_client.DEFAULT_STATUS_FORCELIST`, targets rate limits and server-side failures (408, 409, 429, 500, 502, 503, 504) and intentionally skips `404 Not Found`. Retrying missing resources usually wastes the retry budget and slows down processing, so only opt in when a specific API is known to return temporary 404 responses. To enable this behaviour, provide a custom list in the configuration file, for example:

```yaml
pipeline:
  status_forcelist: [404, 408, 409, 429, 500, 502, 503, 504]
```

or construct an HTTP client directly with `HttpClient(..., status_forcelist=DEFAULT_STATUS_FORCELIST | {404})` for carefully scoped scripts.

### Running the Pipeline

The main entry point for the unified pipeline is `scripts/pipeline_targets_main.py`. This script orchestrates the entire data acquisition and normalization process.

```bash
python scripts/pipeline_targets_main.py \
    --input data/input/targets.csv \
    --output data/output/final_targets.csv \
    --id-column target_chembl_id \
    --with-orthologs \
    --with-isoforms \
    --iuphar-target data/input/iuphar_target.csv \
    --iuphar-family data/input/iuphar_family.csv
```

### Individual Scripts

The `scripts/` directory contains several other scripts for performing specific tasks:

*   `chembl2uniprot_main.py`: Map ChEMBL IDs to UniProt accessions.
*   `chembl_tissue_main.py`: Download tissue metadata directly from ChEMBL.
*   `get_target_data_main.py`: Download target metadata from ChEMBL.
*   `get_cell_line_main.py`: Download metadata for specific ChEMBL cell lines and
    serialise the records as JSON lines.
*   `get_uniprot_target_data.py`: Retrieve and normalize detailed information about UniProt targets.
    The input CSV must expose a `uniprot_id` column unless `--column` is used to
    point at an alternative header.
*   `get_hgnc_by_uniprot.py`: Map UniProt accessions to HGNC identifiers.
*   `dump_gtop_target.py`: Download comprehensive GtoPdb target information.
*   `protein_classify_main.py`: Classify proteins based on UniProt data.
*   `uniprot_enrich_main.py`: Enrich a CSV file with additional UniProt annotations.
*   `chembl_assays_main.py`: Retrieve, validate, and export ChEMBL assay metadata with quality reports.
*   `chembl_activities_main.py`: Stream activity identifiers, fetch ChEMBL activity records, normalise/validate them, and emit quality reports.

For detailed usage information for each script, run it with the `--help` flag.
All command line entry points accept `--log-format` to switch between the
default human-readable output and structured JSON logs, complementing the
existing `--log-level` control.

#### `chembl_testitems_main.py`

The `chembl_testitems_main.py` CLI fetches molecule metadata from ChEMBL and
optionally enriches the results with PubChem descriptors.  In addition to the
existing `--pubchem-timeout`, `--pubchem-base-url`, and `--pubchem-user-agent`
flags, the following options fine-tune the PubChem HTTP client:

* `--pubchem-max-retries` – maximum retry attempts before giving up (default:
  `3`).
* `--pubchem-rps` – allowed PubChem requests per second (default: `5.0`).
* `--pubchem-backoff` – exponential backoff multiplier applied between
  retries (default: `1.0`).
* `--pubchem-retry-penalty` – additional cooldown in seconds added after each
  retry cycle (default: `5.0`).

Combine these parameters to comply with local rate limits or API usage
guidelines.  All supplied values are captured in the CLI metadata sidecar to
aid reproducibility.

## Library

The `library/` directory contains the core logic of the pipeline, organized into several modules:

*   `chembl_targets.py`: Utilities for downloading and normalizing ChEMBL target records.
*   `uniprot_client.py`: A client for the UniProt REST API.
*   `uniprot_normalize.py`: Functions for normalizing UniProt data.
*   `hgnc_client.py`: A client for the HGNC REST API.
*   `gtop_client.py`: A client for the IUPHAR/BPS Guide to PHARMACOLOGY API.
*   `iuphar.py`: Utilities for working with IUPHAR data and performing classifications.
*   `orthologs.py`: Clients for retrieving gene orthology information from Ensembl and OMA.
*   `protein_classifier.py`: Deterministic protein classification based on UniProt data.
*   `pipeline_targets.py`: The main orchestration logic for the unified pipeline.

These modules can be used programmatically to build custom data processing workflows.

## Testing

The project includes a suite of unit and integration tests. To run the tests, use `pytest`:

```bash
pytest
```

To ensure code quality, the following tools are used:

*   `ruff format` (compatible with the Black profile) for code formatting
*   `ruff` for linting
*   `mypy` for static type checking in strict mode

You can run these checks with the following commands:

```bash
ruff format --check .
ruff check .
mypy --strict
```

The formatting and linting configuration is centralised in `pyproject.toml`. Both
`black` and `ruff` use a line length of 88 characters and target Python 3.10
syntax, ensuring the same limits apply regardless of which tool is run.

Strict type checking is being rolled out incrementally. The `mypy --strict`
invocation currently validates the `scripts/chembl_testitems_main.py` entry
point, providing a template for migrating additional modules to strict typing.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
#### ChEMBL activities extraction

The `chembl_activities_main.py` script orchestrates the end-to-end retrieval and
validation of ChEMBL activity records. Key features include streaming input ID
reading, optional limits for sampling large files, resilient API access with a
configurable User-Agent, deterministic normalisation, schema-based validation
with JSON sidecar reports, and automatic quality profiling alongside metadata
sidecars.

Example commands:

```bash
# Inspect the input file without making API calls or writing outputs
python scripts/chembl_activities_main.py --input activities.csv --dry-run

# Download, normalise, and validate activities with explicit limits and output paths
python scripts/chembl_activities_main.py \
    --input activities.csv \
    --output output_activities.csv \
    --column activity_chembl_id \
    --limit 1000 \
    --chunk-size 10 \
    --log-level DEBUG \
    --log-format json
```

The `--chunk-size` option must be a positive integer; zero or negative values
are rejected during argument parsing.

Validation errors are persisted to `<output_filename>.errors.json`, where
`<output_filename>` includes the complete original name (for example,
`dataset.tar.gz.errors.json`). Dataset metadata is written to
`<output_filename>.meta.yaml`, and quality plus correlation reports are produced
alongside the main CSV file using the same base filename.

### Performance smoke testing

`chembl_activities_main.py` is the preferred entry point for quick performance
smoke checks because it supports both `--dry-run` and `--limit` arguments. The
snippet below exercises the CLI against the bundled
`tests/data/activities_input.csv` file without making network calls while
capturing the runtime via `time.perf_counter`:

```bash
python - <<'PY'
import pathlib
import runpy
import sys
import time

sys.path.insert(0, str(pathlib.Path('scripts').resolve()))
sys.argv = [
    'chembl_activities_main.py',
    '--input',
    'tests/data/activities_input.csv',
    '--dry-run',
    '--limit',
    '50',
]
start = time.perf_counter()
try:
    runpy.run_path('scripts/chembl_activities_main.py', run_name='__main__')
except SystemExit as exc:
    if exc.code not in (0, None):
        raise
elapsed = time.perf_counter() - start
print(f"Dry-run completed in {elapsed:.3f}s")
PY
```

Engineers can drop the `--dry-run` flag and keep a small `--limit` (for example
`--limit 5`) to perform a short end-to-end request against the live API:

```bash
python scripts/chembl_activities_main.py \
    --input tests/data/activities_input.csv \
    --output output/activities_smoke.csv \
    --column activity_chembl_id \
    --limit 5 \
    --chunk-size 5 \
    --log-level INFO
```

As above, ensure that `--chunk-size` is set to a positive integer before
running the command.

Add the first command to the CI smoke test job to guard against regressions in
argument parsing and input handling without depending on external services.

### CSV serialisation guidelines

The command-line interfaces rely on :func:`library.cli_common.serialise_dataframe`
before exporting tables with :meth:`pandas.DataFrame.to_csv`. The helper only
materialises object-like columns and accepts ``inplace=True`` to mutate the
input DataFrame, which halves the temporary memory footprint when the original
object is no longer needed. Nevertheless, the final CSV export still requires
the entire table to reside in memory because pandas buffers rows until the write
completes. For multi-gigabyte datasets either enable ``inplace=True`` (the
default for the bundled CLIs) or switch to chunked writes by passing
``chunksize`` to :meth:`pandas.DataFrame.to_csv` when customising scripts.
