# Usage

## get_target_data_main.py

Download target metadata from ChEMBL for identifiers listed in a CSV file:

```bash
python scripts/get_target_data_main.py \
    --input data/targets.csv \
    --output out/targets_dump.csv \
    --column target_chembl_id \
    --log-level INFO
```

The input file must contain a column with ChEMBL target identifiers. Duplicate
and empty values are ignored. The resulting CSV contains one row per unique
identifier with nested fields serialised as JSON strings.
