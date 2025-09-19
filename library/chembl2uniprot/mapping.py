"""Mapping utilities between ChEMBL identifiers and UniProt IDs.

The main entry point is :func:`map_chembl_to_uniprot` which performs the
mapping for a given input CSV file.

Algorithm Notes
---------------
1. Read and validate the YAML configuration file to obtain column names,
   network settings and batching parameters.
2. Stream the input CSV lazily, normalise and deduplicate the ChEMBL
   identifiers and split them into batches without materialising the full
   dataset in memory.
3. For each batch, submit an ID mapping job to the UniProt service, polling
   for completion when necessary and fetching the resulting mapping.
4. Combine all retrieved mappings and write the augmented CSV row by row while
   appending the UniProt identifiers as an additional column.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from urllib.parse import urljoin

from typing import Any, Dict, Iterable, Iterator, List, Set, cast, Literal, Sequence

import hashlib
import json
import logging
import time

import pandas as pd
import requests
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import (
    Config,
    ResolvedRuntimeOptions,
    RetryConfig,
    UniprotConfig,
    load_and_validate_config,
    resolve_runtime_options,
)
from .logging_utils import configure_logging

try:  # pragma: no cover - import resolution varies between runtime contexts
    from http_client import (  # type: ignore[import-not-found]
        DEFAULT_STATUS_FORCELIST,
        RateLimiter,
        RetryAfterWaitStrategy,
        retry_after_from_response,
    )
except ImportError:  # pragma: no cover - fallback when executed as a package
    from ..http_client import (
        DEFAULT_STATUS_FORCELIST,
        RateLimiter,
        RetryAfterWaitStrategy,
        retry_after_from_response,
    )

try:
    from data_profiling import analyze_table_quality
except ModuleNotFoundError:  # pragma: no cover
    from ..data_profiling import analyze_table_quality

LOGGER = logging.getLogger(__name__)


DEFAULT_USER_AGENT = "ChEMBLDataAcquisition/chembl2uniprot"
"""Descriptive User-Agent used for UniProt API requests."""


def _create_session() -> requests.Session:
    """Return a :class:`requests.Session` pre-configured for UniProt calls."""

    session = requests.Session()
    # Identify the integration clearly to aid UniProt troubleshooting.
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
    return session


_SESSION: requests.Session = _create_session()


FAILED_IDS_ERROR_THRESHOLD = 100
"""Maximum acceptable number of failed identifiers per UniProt job."""


# ---------------------------------------------------------------------------
# Helper classes and functions


def _chunked(seq: Iterable[str], size: int) -> Iterator[List[str]]:
    """Yield ``seq`` in lists containing at most ``size`` elements.

    Parameters
    ----------
    seq:
        The iterable to chunk. The iterable is consumed lazily which makes the
        helper suitable for generators.
    size:
        The maximum size of each chunk. ``size`` must be a positive integer.

    Returns
    -------
    Iterator[List[str]]
        An iterator over lists containing up to ``size`` elements from
        ``seq``.
    """

    if size <= 0:
        raise ValueError("size must be positive")

    chunk: List[str] = []
    for item in seq:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _stream_unique_ids(
    path: Path, column: str, *, sep: str, encoding: str
) -> Iterator[str]:
    """Yield trimmed, unique identifiers from a CSV column lazily.

    Parameters
    ----------
    path:
        CSV file that stores the raw identifiers.
    column:
        Name of the column containing the identifier values.
    sep:
        Column separator used in the CSV file.
    encoding:
        Text encoding used to decode the CSV file.

    Yields
    ------
    Iterator[str]
        Unique identifier strings in order of appearance with surrounding
        whitespace removed. Empty values and case-insensitive ``"nan"``
        literals are skipped.
    """

    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=sep)
        if reader.fieldnames is None or column not in reader.fieldnames:
            msg = f"Missing required column '{column}'"
            raise KeyError(msg)

        seen: Set[str] = set()
        for row in reader:
            raw_value = row.get(column)
            if raw_value is None:
                continue
            value = str(raw_value).strip()
            if not value or value.lower() == "nan" or value in seen:
                continue
            seen.add(value)
            yield value


@dataclass
class BatchMappingResult:
    """Container for the mapping output of a single batch.

    Attributes
    ----------
    mapping:
        Mapping from the original ChEMBL identifiers to the resolved UniProt
        accessions.
    failed_ids:
        Identifiers reported by UniProt in the ``failedIds`` field or inferred
        as unresolved when the job could not be completed.
    """

    mapping: Dict[str, List[str]]
    failed_ids: List[str]


def _normalise_failed_ids(raw_failed: Any) -> List[str]:
    """Normalise the content of a ``failedIds`` payload to a list of strings."""

    if not raw_failed:
        return []

    if isinstance(raw_failed, list):
        items = raw_failed
    else:
        items = [raw_failed]

    normalised: List[str] = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, dict):
            candidate = (
                item.get("from")
                or item.get("id")
                or item.get("identifier")
                or str(item)
            )
        else:
            candidate = str(item)
        text = str(candidate).strip()
        if text:
            normalised.append(text)
    return normalised


def _log_and_maybe_raise_failed_ids(job_id: str, failed_ids: Sequence[str]) -> None:
    """Emit diagnostics about failed identifiers and enforce the threshold."""

    if not failed_ids:
        return
    LOGGER.warning(
        "UniProt job %s reported %d failed identifiers: %s",
        job_id,
        len(failed_ids),
        list(failed_ids),
    )
    if len(failed_ids) > FAILED_IDS_ERROR_THRESHOLD:
        raise RuntimeError(
            "UniProt job %s reported %d failed identifiers (threshold %d)"
            % (job_id, len(failed_ids), FAILED_IDS_ERROR_THRESHOLD)
        )


def get_ids_from_dataframe(df: pd.DataFrame, column: str) -> List[str]:
    """Return normalised, unique identifiers from ``column``.

    Parameters
    ----------
    df:
        The input :class:`pandas.DataFrame` containing the identifier column.
    column:
        Name of the column that stores the ChEMBL identifiers.

    Returns
    -------
    List[str]
        A list of unique identifier strings in their original order.

    Notes
    -----
    Missing values are dropped prior to casting values to strings to avoid
    stringified ``"nan"`` entries. Literal ``"nan"`` values (case-insensitive)
    and empty strings are removed from the result as well.
    """

    ids = df[column].dropna()
    if ids.empty:
        return []

    normalised = ids.astype(str).str.strip()
    mask = (normalised != "") & (normalised.str.lower() != "nan")
    filtered = normalised[mask]
    if filtered.empty:
        return []

    return list(filtered.drop_duplicates())


def _request_with_retry(
    method: str,
    url: str,
    *,
    timeout: float,
    rate_limiter: RateLimiter,
    max_attempts: int,
    backoff: float,
    penalty_seconds: float | None = None,
    session: requests.Session | None = None,
    **kwargs: Any,
) -> requests.Response:
    """Perform an HTTP request with retry and rate limiting.

    This function wraps `requests.request` with `tenacity` for automatic
    retries on failures and a custom `RateLimiter` to avoid overwhelming
    the server.

    Parameters
    ----------
    method:
        The HTTP method to use (e.g., "GET", "POST").
    url:
        The URL to request.
    timeout:
        The request timeout in seconds.
    rate_limiter:
        An instance of `RateLimiter` to control the request rate.
    max_attempts:
        The maximum number of retry attempts.
    backoff:
        The backoff factor for exponential backoff between retries.
    penalty_seconds:
        Optional cooldown enforced when UniProt responds with a retryable
        status code without specifying an explicit wait duration.
        ``None`` falls back to ``backoff`` with a minimum of one second.
    session:
        Optional HTTP session. When omitted a module level session with a
        descriptive ``User-Agent`` header is reused across calls.
    **kwargs:
        Additional keyword arguments to pass to :meth:`requests.Session.request`.

    Returns
    -------
    requests.Response
        The HTTP response object.
    """

    wait_strategy = RetryAfterWaitStrategy(wait_exponential(multiplier=backoff))
    http_session = session or _SESSION

    def _log_retry(retry_state: RetryCallState) -> None:
        if retry_state.outcome is None:
            return
        sleep_seconds = 0.0
        if (
            retry_state.next_action is not None
            and retry_state.next_action.sleep is not None
        ):
            sleep_seconds = float(retry_state.next_action.sleep)
        exception = retry_state.outcome.exception()
        reason = "unknown"
        if isinstance(exception, requests.HTTPError) and exception.response is not None:
            status_code = exception.response.status_code
            reason = f"HTTP {status_code}"
            if status_code in {408, 429} and sleep_seconds > 0:
                rate_limiter.apply_penalty(sleep_seconds)
        elif exception is not None:
            reason = repr(exception)
        next_attempt = retry_state.attempt_number + 1
        LOGGER.warning(
            "Retrying %s %s (attempt %d/%d) after %.2f seconds due to %s",
            method.upper(),
            url,
            next_attempt,
            max_attempts,
            sleep_seconds,
            reason,
        )

    @retry(
        reraise=True,
        retry=retry_if_exception_type(requests.RequestException),
        stop=stop_after_attempt(max_attempts),
        wait=wait_strategy,
        before_sleep=_log_retry,
    )
    def _do_request() -> requests.Response:
        # Honour the rate limit before each network call, including retries.
        rate_limiter.wait()
        resp = http_session.request(method, url, timeout=timeout, **kwargs)
        if resp.status_code in DEFAULT_STATUS_FORCELIST:
            retry_after = retry_after_from_response(resp)
            if retry_after is not None and retry_after > 0:
                LOGGER.warning(
                    "Transient HTTP %s for %s %s; server requested %.2f seconds pause",
                    resp.status_code,
                    method.upper(),
                    url,
                    retry_after,
                )
                rate_limiter.apply_penalty(retry_after)
            else:
                fallback_penalty = (
                    penalty_seconds
                    if penalty_seconds is not None and penalty_seconds > 0
                    else (backoff if backoff > 0 else 1.0)
                )
                LOGGER.warning(
                    "Transient HTTP %s for %s %s; retrying with backoff and applying %.2f s penalty",
                    resp.status_code,
                    method.upper(),
                    url,
                    fallback_penalty,
                )
                rate_limiter.apply_penalty(fallback_penalty)
            resp.raise_for_status()
        return resp

    resp = _do_request()
    resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# Core logic


def _start_job(
    ids: List[str],
    cfg: UniprotConfig,
    rate_limiter: RateLimiter,
    timeout: float,
    retry_cfg: RetryConfig,
) -> Dict[str, Any]:
    """Start a new UniProt ID mapping job.

    Parameters
    ----------
    ids:
        A list of ChEMBL IDs to map.
    cfg:
        The UniProt configuration.
    rate_limiter:
        A `RateLimiter` instance.
    timeout:
        The request timeout in seconds.
    retry_cfg:
        The retry configuration.

    Returns
    -------
    Dict[str, Any]
        The JSON response from the UniProt API, which contains the job ID.
    """
    url = cfg.base_url.rstrip("/") + cfg.id_mapping.endpoint
    payload = {
        "from": cfg.id_mapping.from_db or "ChEMBL",
        "to": cfg.id_mapping.to_db or "UniProtKB",
        "ids": ",".join(ids),
    }
    resp = _request_with_retry(
        "post",
        url,
        timeout=timeout,
        rate_limiter=rate_limiter,
        max_attempts=retry_cfg.max_attempts,
        backoff=retry_cfg.backoff_sec,
        penalty_seconds=retry_cfg.penalty_sec,
        data=payload,
    )
    try:
        return cast(Dict[str, Any], resp.json())
    except json.JSONDecodeError:
        LOGGER.debug("Unparseable response from %s: %s", url, resp.text)
        raise


def _poll_job(
    job_id: str,
    cfg: UniprotConfig,
    rate_limiter: RateLimiter,
    timeout: float,
    retry_cfg: RetryConfig,
) -> None:
    """Poll the status of a UniProt ID mapping job until it completes.

    Parameters
    ----------
    job_id:
        The ID of the job to poll.
    cfg:
        The UniProt configuration.
    rate_limiter:
        A `RateLimiter` instance.
    timeout:
        The request timeout in seconds.
    retry_cfg:
        The retry configuration.
    """
    status_url = (
        cfg.base_url.rstrip("/") + (cfg.id_mapping.status_endpoint or "") + "/" + job_id
    )
    interval = cfg.polling.interval_sec
    while True:
        resp = _request_with_retry(
            "get",
            status_url,
            timeout=timeout,
            rate_limiter=rate_limiter,
            max_attempts=retry_cfg.max_attempts,
            backoff=retry_cfg.backoff_sec,
            penalty_seconds=retry_cfg.penalty_sec,
            allow_redirects=False,
        )
        if resp.status_code == 303:
            # UniProt signals completion via HTTP 303 and a "Location" header
            return
        try:
            payload = resp.json()
        except json.JSONDecodeError:
            LOGGER.debug("Unparseable status response: %s", resp.text)
            raise
        if payload.get("jobStatus") in {"FINISHED", "finished"}:
            return
        if payload.get("jobStatus") in {"ERROR", "failed"}:
            raise RuntimeError(f"UniProt job {job_id} failed: {payload}")
        time.sleep(interval)


def _fetch_results(
    job_id: str,
    cfg: UniprotConfig,
    rate_limiter: RateLimiter,
    timeout: float,
    retry_cfg: RetryConfig,
) -> BatchMappingResult:
    """Fetch the results of a completed UniProt ID mapping job.

    Parameters
    ----------
    job_id:
        The ID of the completed job.
    cfg:
        The UniProt configuration.
    rate_limiter:
        A `RateLimiter` instance.
    timeout:
        The request timeout in seconds.
    retry_cfg:
        The retry configuration.

    Returns
    -------
    BatchMappingResult
        Combined mapping information and failed identifiers for the job.
    """
    results_url = (
        cfg.base_url.rstrip("/")
        + (cfg.id_mapping.results_endpoint or "")
        + "/"
        + job_id
    )
    mapping: Dict[str, List[str]] = {}
    failed_ids: List[str] = []
    next_url: str | None = results_url

    while next_url:
        resp = _request_with_retry(
            "get",
            next_url,
            timeout=timeout,
            rate_limiter=rate_limiter,
            max_attempts=retry_cfg.max_attempts,
            backoff=retry_cfg.backoff_sec,
            penalty_seconds=retry_cfg.penalty_sec,
        )
        try:
            payload = resp.json()
        except json.JSONDecodeError:
            LOGGER.debug("Unparseable results response: %s", resp.text)
            raise

        for item in payload.get("results", []):
            frm = item.get("from")
            to = item.get("to")
            if frm and to:
                mapping.setdefault(frm, []).append(to)
        failed_ids.extend(_normalise_failed_ids(payload.get("failedIds")))

        next_link = payload.get("next")
        if next_link:
            next_url = urljoin(results_url, next_link)
        else:
            next_url = None

    _log_and_maybe_raise_failed_ids(job_id, failed_ids)
    return BatchMappingResult(mapping=mapping, failed_ids=failed_ids)


def _map_batch(
    ids: List[str],
    cfg: UniprotConfig,
    rate_limiter: RateLimiter,
    timeout: float,
    retry_cfg: RetryConfig,
) -> BatchMappingResult:
    """Map a batch of ChEMBL IDs to UniProt IDs.

    This function handles both synchronous and asynchronous UniProt API
    responses.

    Parameters
    ----------
    ids:
        A list of ChEMBL IDs to map.
    cfg:
        The UniProt configuration.
    rate_limiter:
        A `RateLimiter` instance.
    timeout:
        The request timeout in seconds.
    retry_cfg:
        The retry configuration.

    Returns
    -------
    BatchMappingResult
        Structured information about the resolved mappings and failed IDs.
    """
    try:
        job_payload = _start_job(ids, cfg, rate_limiter, timeout, retry_cfg)
    except Exception as exc:
        LOGGER.warning("Failed to start mapping job for batch %s: %s", ids, exc)
        return BatchMappingResult(mapping={}, failed_ids=list(ids))

    if "jobId" in job_payload:
        job_id = job_payload["jobId"]
        try:
            _poll_job(job_id, cfg, rate_limiter, timeout, retry_cfg)
            return _fetch_results(job_id, cfg, rate_limiter, timeout, retry_cfg)
        except Exception as exc:  # broad but logged
            LOGGER.warning("Job %s failed: %s", job_id, exc)
            return BatchMappingResult(mapping={}, failed_ids=list(ids))

    # Synchronous result
    if "results" in job_payload:
        mapping: Dict[str, List[str]] = {}
        for item in job_payload.get("results", []):
            frm = item.get("from")
            to = item.get("to")
            if frm and to:
                mapping.setdefault(frm, []).append(to)
        failed_ids = _normalise_failed_ids(job_payload.get("failedIds"))
        _log_and_maybe_raise_failed_ids("synchronous", failed_ids)
        return BatchMappingResult(mapping=mapping, failed_ids=failed_ids)

    LOGGER.warning("Unexpected response payload: %s", job_payload)
    return BatchMappingResult(mapping={}, failed_ids=list(ids))


def map_chembl_to_uniprot(
    input_csv_path: str | Path,
    output_csv_path: str | Path | None = None,
    config_path: str | Path = "config.yaml",
    schema_path: str | Path | None = None,
    *,
    config_section: str | None = None,
    log_level: str | None = None,
    log_format: Literal["human", "json"] | None = None,
    sep: str | None = None,
    encoding: str | None = None,
) -> Path:
    """Map ChEMBL identifiers in ``input_csv_path`` to UniProt IDs.

    Parameters
    ----------
    input_csv_path:
        Path to the input CSV file containing a column with ChEMBL identifiers.
    output_csv_path:
        Optional path for the output CSV file.  When ``None`` a file with the
        suffix ``"_with_uniprot.csv"`` is created next to ``input_csv_path``.
    config_path:
        Path to the YAML configuration file.
    schema_path:
        Optional path to the JSON schema used for validation.  When ``None``
        the schema is assumed to reside next to ``config_path`` under the name
        ``config.schema.json``.
    config_section:
        Optional top-level key within the YAML file containing the
        configuration relevant to this function.
    log_level:
        Logging verbosity (e.g. ``"INFO"`` or ``"DEBUG"``). When ``None`` the
        value from the configuration file is used.
    log_format:
        Logging format, ``"human"`` or ``"json"``. Defaults to the configuration
        value when ``None``.
    sep:
        CSV field separator. Falls back to the configuration when ``None``.
    encoding:
        Character encoding for both reading and writing CSV files. Falls back to
        the configuration when ``None``.

    Returns
    -------
    Path
        Path to the written CSV file containing an extra column with the mapped
        UniProt identifiers.

    Raises
    ------
    ValueError
        If the input CSV does not contain the required ChEMBL identifier column.
    """

    cfg: Config = load_and_validate_config(
        config_path, schema_path, section=config_section
    )
    runtime_options: ResolvedRuntimeOptions = resolve_runtime_options(
        cfg,
        cli_log_level=log_level,
        cli_log_format=log_format,
        cli_sep=sep,
        cli_encoding=encoding,
    )

    configure_logging(
        runtime_options.log_level,
        log_format=runtime_options.log_format,
    )
    logging.getLogger("urllib3").setLevel(logging.INFO)  # Reduce HTTP verbosity

    input_csv_path = Path(input_csv_path)
    if output_csv_path is None:
        output_csv_path = input_csv_path.with_name(
            input_csv_path.stem + "_with_uniprot.csv"
        )
    output_csv_path = Path(output_csv_path)

    chembl_col = cfg.columns.chembl_id
    out_col = cfg.columns.uniprot_out
    delimiter = cfg.io.csv.multivalue_delimiter
    separator = runtime_options.separator
    encoding_in = runtime_options.input_encoding
    encoding_out = runtime_options.output_encoding

    # Compute SHA256 of input file for logging purposes
    hasher = hashlib.sha256()
    with input_csv_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            if not chunk:
                break
            hasher.update(chunk)
    file_hash = hasher.hexdigest()
    LOGGER.info("Input file checksum (sha256): %s", file_hash)

    batch_size = cfg.batch.size
    timeout = cfg.network.timeout_sec
    retry_cfg = cfg.uniprot.retry
    rate_limiter = RateLimiter(cfg.uniprot.rate_limit.rps)

    mapping: Dict[str, List[str]] = {}

    failed_identifiers: List[str] = []
    unique_count = 0
    try:
        id_iter = _stream_unique_ids(
            input_csv_path, chembl_col, sep=separator, encoding=encoding_in
        )
        for batch in _chunked(id_iter, batch_size):
            unique_count += len(batch)
            batch_result = _map_batch(
                batch, cfg.uniprot, rate_limiter, timeout, retry_cfg
            )
            for key, value in batch_result.mapping.items():
                if not key:
                    continue
                normalised_key = str(key).strip()
                if not normalised_key:
                    continue
                mapping[normalised_key] = value
            failed_identifiers.extend(batch_result.failed_ids)
    except KeyError as exc:
        msg = f"Missing required column '{chembl_col}' in input CSV"
        raise ValueError(msg) from exc

    if failed_identifiers:
        LOGGER.warning(
            "UniProt mapping reported %d failed identifiers: %s",
            len(failed_identifiers),
            failed_identifiers,
        )

    LOGGER.info("Processing %d unique ChEMBL IDs", unique_count)

    mapped = sum(1 for v in mapping.values() if v)
    no_match = unique_count - mapped
    multi = sum(1 for v in mapping.values() if len(v) > 1)
    LOGGER.info(
        "Summary: unique=%d mapped=%d no_match=%d multi=%d",
        unique_count,
        mapped,
        no_match,
        multi,
    )

    with (
        input_csv_path.open("r", encoding=encoding_in, newline="") as src,
        output_csv_path.open("w", encoding=encoding_out, newline="") as dst,
    ):
        reader = csv.DictReader(src, delimiter=separator)
        if reader.fieldnames is None or chembl_col not in reader.fieldnames:
            msg = f"Missing required column '{chembl_col}' in input CSV"
            raise ValueError(msg)

        fieldnames = list(reader.fieldnames)
        if out_col not in fieldnames:
            fieldnames.append(out_col)

        writer = csv.DictWriter(dst, fieldnames=fieldnames, delimiter=separator)
        writer.writeheader()

        for row in reader:
            raw_value = row.get(chembl_col)
            if raw_value is None:
                row[out_col] = ""
            else:
                stripped = str(raw_value).strip()
                if not stripped or stripped.lower() == "nan":
                    row[out_col] = ""
                else:
                    ids = mapping.get(stripped)
                    row[out_col] = delimiter.join(ids) if ids else ""
            writer.writerow(row)

    analyze_table_quality(
        output_csv_path, table_name=str(Path(output_csv_path).with_suffix(""))
    )
    return output_csv_path
