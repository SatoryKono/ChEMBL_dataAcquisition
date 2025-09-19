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
    --sep , \
    --encoding utf-8-sig \
    --list-format json \
    --log-level INFO \
    --meta-output out/targets_dump.csv.meta.yaml
```

The input file must contain a column with ChEMBL target identifiers. Duplicate
and empty values are ignored. The resulting CSV contains one row per unique
identifier with nested fields serialised deterministically via
``--list-format``. A companion ``<output_filename>.meta.yaml`` file captures the
CLI invocation, row/column counts, and output checksum.  The suffix is appended
to the entire output filename to support multi-extension artefacts such as
``.tar.gz``.

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
    --sep , \
    --encoding utf-8-sig \
    --log-level INFO \
    --meta-output out/gtop/targets_overview.meta.yaml
```

This command creates ``targets.csv`` together with related tables such as
``targets_synonyms.csv`` and ``targets_interactions.csv`` in the specified output
directory. Only unique identifiers are queried and results are written in a
deterministic order. The main table is accompanied by metadata and validation
sidecars (configured above via ``--meta-output`` and the default
``<output_filename>.errors.json``) that record runtime parameters and checksums.

## Performance smoke testing

Use ``chembl_activities_main.py`` for quick performance smoke checks. The
following snippet measures how long the CLI takes to parse a realistic input
file without contacting external APIs:

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

This command is suitable for a CI smoke test step because it exercises argument
parsing and input validation without relying on the ChEMBL API. For a short
end-to-end measurement, remove ``--dry-run`` and keep a conservative limit:

```bash
python scripts/chembl_activities_main.py \
    --input tests/data/activities_input.csv \
    --output output/activities_smoke.csv \
    --column activity_chembl_id \
    --limit 5 \
    --chunk-size 5 \
    --log-level INFO
```
