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
*   `get_cell_line_main.py`: Download metadata for specific ChEMBL cell lines and
    serialise the records as JSON lines.
*   `get_uniprot_target_data.py`: Retrieve and normalize detailed information about UniProt targets.
*   `get_hgnc_by_uniprot.py`: Map UniProt accessions to HGNC identifiers.
*   `dump_gtop_target.py`: Download comprehensive GtoPdb target information.
*   `protein_classify_main.py`: Classify proteins based on UniProt data.
*   `uniprot_enrich_main.py`: Enrich a CSV file with additional UniProt annotations.

For detailed usage information for each script, run it with the `--help` flag.

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
