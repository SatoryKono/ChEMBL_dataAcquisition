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

*   Python >= 3.10
*   pandas >= 1.5
*   requests >= 2.31
*   PyYAML >= 6.0
*   jsonschema >= 4.17
*   tqdm >= 4.66
*   tenacity >= 8.2
*   pydantic >= 2.0

### Development Requirements

*   pytest >= 7.4
*   requests-mock >= 1.11
*   black >= 23.0
*   ruff >= 0.1
*   mypy >= 1.4

## Installation

1.  Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

2.  Install the project in editable mode with development dependencies:
    ```bash
    pip install -e .[dev]
    ```

3.  Install the pre-commit hooks to ensure consistent formatting, linting, type
    checking, and tests before each commit:
    ```bash
    pre-commit install
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
*   `get_target_data_main.py`: Download target metadata from ChEMBL.
*   `get_uniprot_target_data.py`: Retrieve and normalize detailed information about UniProt targets.
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

*   `black` for code formatting
*   `ruff` for linting
*   `mypy` for static type checking

You can run these checks with the following commands:

```bash
black --check .
ruff check .
mypy .
```

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

Validation errors are persisted to `<output>.errors.json` while dataset metadata
is written to `<output>.meta.yaml`. Quality and correlation reports are produced
alongside the main CSV file.
