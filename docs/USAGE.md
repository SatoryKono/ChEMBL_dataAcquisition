# Usage

## get_target_data_main.py

Download target metadata from ChEMBL for identifiers listed in a CSV file:
All scripts accept `--log-format` to switch between human-readable and JSON log
outputs in addition to the `--log-level` control shown in the examples below.

```bash
python scripts/get_target_data_main.py \
    --input data/targets.csv \
    --output out/targets_dump.csv \
    --column target_chembl_id \
    --log-level INFO \
    --log-format json
```

The input file must contain a column with ChEMBL target identifiers. Duplicate
and empty values are ignored. The resulting CSV contains one row per unique
identifier with nested fields serialised as JSON strings.

## dump_gtop_target.py

Resolve identifiers against the IUPHAR/BPS Guide to PHARMACOLOGY and download
target-related resources:

```bash
python scripts/dump_gtop_target.py \
    --input data/targets.csv \
    --output-dir out/gtop \
    --id-column uniprot_id \
    --affinity-parameter pKi \
    --affinity-ge 7 \
    --log-level INFO \
    --log-format json
```

This command creates ``targets.csv`` together with related tables such as
``targets_synonyms.csv`` and ``targets_interactions.csv`` in the specified output
directory. Only unique identifiers are queried and results are written in a
deterministic order.
