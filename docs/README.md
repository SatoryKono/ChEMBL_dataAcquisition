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
- pydantic >= 2.0

Development and testing additionally require:

- pytest >= 7.4
- requests-mock >= 1.11
- hypothesis
- black >= 23.0
- ruff >= 0.1
- mypy >= 1.4
- types-requests
- types-PyYAML

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
library/
    chembl2uniprot/      # Configuration and mapping utilities
        __init__.py
        config.py
    mapping.py
schemas/
    config.schema.json   # JSON schema for configuration validation
scripts/
    chembl2uniprot_main.py  # ChEMBL to UniProt mapper
    get_target_data_main.py # ChEMBL target downloader
tests/
    data/            # Sample config and CSV files used in tests
    test_mapping.py  # Unit tests
config.yaml         # Unified configuration
pyproject.toml       # Build system and dependency declaration
requirements.txt     # Dependency pinning
.gitignore
```

## Usage


1. Prepare a configuration file (see ``tests/data/config/valid.yaml`` for an example)
   or use the bundled ``config.yaml``.
2. Run the mapper:

```bash
python scripts/chembl2uniprot_main.py \
    --input input.csv \
    --output output.csv \
    --log-level INFO \
    --sep , \
    --encoding utf-8
```

To supply an alternative configuration file:

```bash
python scripts/chembl2uniprot_main.py \
    --input input.csv \
    --output output.csv \
    --config my_config.yaml
```


Flags ``--log-level``, ``--sep`` and ``--encoding`` are optional and default to
``INFO``, ``,``, and ``utf-8`` respectively.  When ``--output`` is omitted the
result is written next to ``input.csv`` with the suffix ``_with_uniprot.csv``.
The mapped UniProt identifiers are stored in a new column defined by the
configuration file.

The mapping function can also be used programmatically:

```python
from chembl2uniprot import map_chembl_to_uniprot

map_chembl_to_uniprot(
    "input.csv",
    "output.csv",
    "config.yaml",
    log_level="DEBUG",
    sep=";",
    encoding="utf-8",
)
```


## UniProt dump

The ``get_uniprot_target_data.py`` script retrieves detailed information about
UniProt targets.  Provide a CSV file with a column containing UniProt accession
IDs and obtain a normalised dump with deterministic column ordering.

```bash
python scripts/get_uniprot_target_data.py \
    --input ids.csv \
    --output targets.csv \
    --column uniprot_id \
    --include-sequence
```

The behaviour is configured via ``config.yaml``.  Lists are serialised either as
JSON (default) or as ``|``-delimited strings depending on the configuration.


### Including orthologs

Orthologous genes can be fetched via the Ensembl REST API and attached to the
output.  Enable this behaviour with ``--with-orthologs`` and optionally specify
an explicit path for the normalised ortholog table using ``--orthologs-output``.

### Isoforms

When the ``--with-isoforms`` flag is supplied the script queries the UniProt
REST API for the full set of isoforms for each accession.  A separate CSV file
is written via ``--isoforms-output`` containing one row per isoform with the
columns ``parent_uniprot_id``, ``isoform_uniprot_id``, ``isoform_name``,
``isoform_synonyms`` and ``is_canonical``.  The main output gains the aggregated
fields ``isoform_ids_all``, ``isoforms_json`` and ``isoforms_count``.


```bash
python scripts/get_uniprot_target_data.py \
    --input ids.csv \
    --output targets.csv \

    --with-orthologs \
    --orthologs-output orthologs.csv
```

Two additional columns are written to the main CSV: ``orthologs_json`` contains
an array of ortholog descriptors and ``orthologs_count`` records the number of
matches.  The secondary CSV lists one row per source/target pair with the
following columns:

``source_uniprot_id, source_ensembl_gene_id, source_species, target_species,
target_gene_symbol, target_ensembl_gene_id, target_uniprot_id, homology_type,
perc_id, perc_pos, dn, ds, is_high_confidence, source_db``.

Supported target species in the default configuration are ``human``, ``mouse``,
``rat``, ``zebrafish``, ``dog`` and ``macaque``.

    --with-isoforms \
    --isoforms-output isoforms.csv
```

Further details about alternative products are available in the
[UniProt documentation](https://www.uniprot.org/help/alternative_products).


### Downloading target metadata

Fetch basic information for targets listed in ``targets.csv`` and write the
result to ``targets_dump.csv``:

```bash
python scripts/get_target_data_main.py \
    --input targets.csv \
    --output targets_dump.csv \
    --column target_chembl_id \
    --log-level INFO
```

Nested fields in the output are encoded as JSON strings to ensure
deterministic, machine-readable results.

The set of columns retrieved from ChEMBL can be customised in
``config.yaml`` under the ``chembl.columns`` section.

### Unified pipeline

Combine ChEMBL, UniProt, HGNC and GtoP data into a single table:

```bash
python scripts/pipeline_targets_main.py \
    --input targets.csv \
    --output final.csv \
    --id-column target_chembl_id
```

The output contains one row per ``target_chembl_id`` with blocks of columns
covering identifiers, taxonomy, sequence features, cross-references and a brief
IUPHAR summary.  Lists are serialised as JSON arrays and sorted to guarantee
reproducible files.  See the source for the full column list.


## Testing and quality checks

After making changes, run the following tools:

```bash
ruff check .
black --check .
python -m mypy .
pytest
```

## License

This project is provided under the MIT license.
