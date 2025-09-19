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
    --log-format json \
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


Flags ``--log-level``, ``--log-format``, ``--sep`` and ``--encoding`` are optional
and default to ``INFO``, ``human``, ``,``, and ``utf-8`` respectively.  When
``--output`` is omitted the
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
The relevant configuration sections are:

- ``output`` – CSV separator, encoding, default list serialisation format and
  whether sequences are exported.
- ``uniprot`` – REST endpoint, retry budget, timeout, rate limit and optional
  request-level cache for UniProtKB.
- ``orthologs`` – toggle for ortholog enrichment, allowed species, retry and
  rate limit parameters plus an optional cache configuration.
- ``http_cache`` – global HTTP cache used as a fallback when a section does not
  define its own cache settings.

Each setting can be overridden via environment variables prefixed with
``CHEMBL_DA__``.  Components of the configuration path are separated with double
underscores, for example::

    export CHEMBL_DA__OUTPUT__SEP="\t"
    export CHEMBL_DA__UNIPROT__RPS=6
    export CHEMBL_DA__ORTHOLOGS__TARGET_SPECIES="[\"Human\", \"Mouse\"]"
    export CHEMBL_DA__HTTP_CACHE__ENABLED=true

These overrides are processed before validation, ensuring the resulting
configuration matches the constraints enforced by the loader.

Every run also emits a companion ``.meta.yaml`` file next to the main CSV.  The
metadata captures the executed command line, normalised CLI arguments and
summary statistics such as row and column counts.  Basic data quality metrics
are calculated via ``library.data_profiling.analyze_table_quality`` for
downstream validation.


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
    --sep , \
    --encoding utf-8-sig \
    --list-format json \
    --log-level INFO \
    --meta-output targets_dump.csv.meta.yaml
```

Nested fields in the output are encoded deterministically according to
``--list-format``. The CLI writes both ``targets_dump.csv`` and a metadata
sidecar capturing the command invocation, file checksum, and table dimensions.

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

The CLI honours defaults from ``config.yaml`` but command line switches always
take precedence. For example ``--list-format`` overrides ``pipeline.list_format``
and ``--species`` prepends values to ``pipeline.species_priority``. Ortholog
enrichment follows the ``orthologs.enabled`` setting unless either
``--with-orthologs`` or ``--no-with-orthologs`` is supplied, providing an easy
way to enable or disable the feature without editing YAML files.

Environment variables prefixed with ``CHEMBL_DA__PIPELINE__`` offer an
alternative override mechanism when editing configuration files is not
practical. For instance ``CHEMBL_DA__PIPELINE__LIST_FORMAT=pipe`` switches list
serialisation globally, while ``CHEMBL_DA__PIPELINE__IUPHAR__APPROVED_ONLY=true``
enables the corresponding IUPHAR filter.

Optional sections such as ``orthologs``, ``chembl`` or ``uniprot_enrich`` can be
set to ``null`` in the configuration file when they are not required. The
pipeline falls back to sensible defaults in this case while still allowing the
sections to be re-enabled later without removing placeholder keys.


## CSV output conventions

All command line utilities emit deterministic CSV files together with a
``.meta.yaml`` companion capturing the command invocation, configuration and a
SHA-256 digest of the output. The metadata file also records a determinism
summary in the ``determinism`` section. Each time the CLI is executed the
current hash is compared with the previous run and ``matches_previous`` reflects
whether the outputs are byte-for-byte identical.

### Column ordering

- Pipelines with an explicit schema (for example
  ``chembl_activities_main.py`` and ``chembl_testitems_main.py``) write schema
  columns first and append any extra fields sorted alphabetically. This keeps
  the core attributes stable while still surfacing optional enrichments in a
  predictable order.
- The unified target pipeline groups related columns so that identifiers and
  annotations appear first while IUPHAR classification columns are appended as a
  block at the end for clarity.

### Row ordering

- Pipelines that accept a list of identifiers preserve the order in which the
  identifiers are supplied after deduplication. When a natural sort order exists
  (e.g. assay or molecule identifiers) the data are sorted on those keys with
  ``NaN`` entries placed at the end of the table to avoid jitter between runs.

### CSV format and encoding

- CSV files use Unix line endings (``\n``) and default to UTF-8 encoding. The
  unified pipeline defaults to ``utf-8-sig`` for compatibility with downstream
  tools, but the encoding and delimiter can be customised via ``--encoding`` and
  ``--sep``.
- Lists and dictionaries are serialised as JSON strings by default. Supplying
  ``--list-format pipe`` switches to a pipe-delimited representation with
  deterministic escaping for embedded ``|`` characters.

### Missing values

- Normalisation layers convert missing collections into empty lists and map
  absent scalars to ``NaN``. When written to CSV these appear as empty cells
  rather than the literal string ``NA``. This ensures downstream consumers can
  distinguish between an empty value and a textual placeholder.

### Determinism tooling

The ``scripts/check_determinism.py`` utility exercises the low-level CSV writer
twice and verifies that the recorded hashes and metadata agree. Use it as a
sanity check when making changes to serialisation logic.


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
