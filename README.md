# ChEMBL data acquisition

Utilities for mapping ChEMBL identifiers to UniProt IDs.

## Installation

```bash
pip install -e .[dev]
```

## Usage

```bash
python -m chembl2uniprot \
    --input input.csv \
    --output output.csv \
    --config config.yaml \
    --log-level INFO \
    --sep ',' \
    --encoding utf-8
```

### Options

- `--log-level` – logging verbosity (`INFO` by default).
- `--sep` – CSV field separator (default `,`).
- `--encoding` – file encoding for both input and output (default `utf-8`).

## Dependencies

- pandas
- requests
- PyYAML
- jsonschema
- tqdm
- tenacity

## Testing

```bash
pytest
```
