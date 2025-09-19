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
``--list-format``. The CLI persists the :func:`library.cli_common.serialise_dataframe`
result directly through :meth:`pandas.DataFrame.to_csv`, which means metadata
and quality analysis routines observe the exact payload stored on disk. A
companion ``<output_filename>.meta.yaml`` file captures the CLI invocation,
row/column counts, and output checksum.  The suffix is appended to the entire
output filename to support multi-extension artefacts such as
``.tar.gz``.

Network and API failures are surfaced as ``requests`` exceptions. The downloader
raises :class:`requests.HTTPError` for non-success HTTP statuses (for example,
``404`` or ``500`` responses) and propagates other
:class:`requests.RequestException` instances so that automation can fail fast
instead of silently producing partial outputs.

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

Always provide a strictly positive integer to `--chunk-size`; the CLI rejects
zero or negative values during parsing to avoid invalid batching parameters.

## Semantic Scholar throughput

The `pubmed_main.py` workflow aggregates metadata from PubMed, Semantic Scholar,
OpenAlex, and Crossref. Semantic Scholar's public API throttles unauthenticated
clients to roughly **0.3 requests per second**, which is the default enforced by
`config/documents.yaml` and the script's built-in configuration. Attempting to
override the limit without explicit consent now falls back to the public rate
and emits a warning so that long-running exports do not trigger HTTP 429 errors.

Higher throughput requires an **official Semantic Scholar API key**. Provide the
credential either through the `SEMANTIC_SCHOLAR_API_KEY` environment variable or
the new `--semantic-scholar-api-key` option. Unlocking the faster rate additionally
demands the `--semantic-scholar-high-throughput` flag (or the matching
`semantic_scholar.high_throughput: true` configuration entry), which keeps
accidental overrides at bay. For example:

```bash
export SEMANTIC_SCHOLAR_API_KEY="sk_live_..."
python scripts/pubmed_main.py \
    --input data/input/document.csv \
    --output output/semantic_scholar.csv \
    --semantic-scholar-rps 1.0 \
    --semantic-scholar-high-throughput \
    scholar
```

When the API key is supplied on the command line, prefer a throwaway environment
variable instead of embedding secrets inside shell history:

```bash
python scripts/pubmed_main.py \
    --input data/input/document.csv \
    --output output/semantic_scholar.csv \
    --semantic-scholar-rps 1.0 \
    --semantic-scholar-high-throughput \
    --semantic-scholar-api-key "${SEMANTIC_SCHOLAR_API_KEY}" \
    scholar
```

Printing the effective configuration via `--print-config` redacts the API key to
reduce the risk of leaking credentials in logs.
