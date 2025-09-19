# Configuration overrides

The applications bundled with this repository load their defaults from
`config.yaml`.  Every configuration block can be tuned without editing the file
by setting environment variables before invoking a script or library function.
Variables prefixed with `CHEMBL_DA__` take precedence; the legacy `CHEMBL_`
prefix is still honoured for backwards compatibility.  Component names are
separated by double underscores and are case-insensitive.  For example, setting
`CHEMBL_DA__HGNC__NETWORK__TIMEOUT_SEC=45` overrides the timeout in the HGNC
section.

## HGNC mapping variables

The HGNC utilities consume the `hgnc` section of `config.yaml`.  The following
variables are recognised by :func:`library.hgnc_client.load_config` and can be
used to adjust runtime behaviour without editing configuration files.

| Variable | Description | Default |
| --- | --- | --- |
| `CHEMBL_DA__HGNC__HGNC__BASE_URL` | Base URL for the HGNC REST API endpoint. | `https://rest.genenames.org/fetch/uniprot_ids` |
| `CHEMBL_DA__HGNC__NETWORK__TIMEOUT_SEC` | Timeout for HTTP requests in seconds. | `30` |
| `CHEMBL_DA__HGNC__NETWORK__MAX_RETRIES` | Maximum number of retry attempts for HGNC calls. | `3` |
| `CHEMBL_DA__HGNC__RATE_LIMIT__RPS` | Allowed request rate in requests per second. | `3` |
| `CHEMBL_DA__HGNC__OUTPUT__SEP` | Delimiter used when writing output CSV files. | `,` |
| `CHEMBL_DA__HGNC__OUTPUT__ENCODING` | Text encoding for generated CSV files. | `utf-8` |
| `CHEMBL_DA__HGNC__CACHE__ENABLED` | Enable (`true`) or disable (`false`) the shared HTTP cache. | `false` |
| `CHEMBL_DA__HGNC__CACHE__PATH` | Filesystem location for the HTTP cache database. | `None` |
| `CHEMBL_DA__HGNC__CACHE__TTL_SEC` | Cache expiry time in seconds (set to `0` to disable persistence). | `0` |

Boolean values accept YAML semantics, so `true`/`false`, `yes`/`no` and `1`/`0`
are all valid inputs.  Numeric values may be provided as integers or floats.  The
values are parsed using `yaml.safe_load`, matching the behaviour of other
configuration loaders in the project.
