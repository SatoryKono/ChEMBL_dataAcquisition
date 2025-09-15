"""Mapping utilities between ChEMBL identifiers and UniProt IDs.

The main entry point is :func:`map_chembl_to_uniprot` which performs the
mapping for a given input CSV file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
import hashlib
import json
import logging
import time

import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .config import load_and_validate_config

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper classes and functions


def _chunked(seq: Sequence[str], size: int) -> Iterable[List[str]]:
    """Yield ``seq`` in chunks of ``size``."""

    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


@dataclass
class RateLimiter:
    """Simple rate limiter based on sleep intervals."""

    rps: float
    last_call: float = 0.0

    def wait(self) -> None:
        if self.rps <= 0:
            return
        interval = 1.0 / self.rps
        now = time.monotonic()
        delta = now - self.last_call
        if delta < interval:
            time.sleep(interval - delta)
        self.last_call = time.monotonic()


def _request_with_retry(
    method: str,
    url: str,
    *,
    timeout: float,
    rate_limiter: RateLimiter,
    max_attempts: int,
    backoff: float,
    **kwargs,
) -> requests.Response:
    """Perform an HTTP request with retry and rate limiting."""

    rate_limiter.wait()

    @retry(
        reraise=True,
        retry=retry_if_exception_type(requests.RequestException),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=backoff, jitter=backoff),
    )
    def _do_request() -> requests.Response:
        resp = requests.request(method, url, timeout=timeout, **kwargs)
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    time.sleep(float(retry_after))
                except ValueError:
                    LOGGER.debug("Invalid Retry-After header: %s", retry_after)
            # Trigger retry for rate limiting
            resp.raise_for_status()
        if resp.status_code >= 500:
            # Trigger retry by raising for 5xx responses
            resp.raise_for_status()
        return resp

    return _do_request()


# ---------------------------------------------------------------------------
# Core logic


def _start_job(
    ids: List[str],
    cfg: Dict[str, Any],
    rate_limiter: RateLimiter,
    timeout: float,
    retry_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    url = cfg["base_url"].rstrip("/") + cfg["id_mapping"]["endpoint"]
    payload = {
        "from": cfg["id_mapping"].get("from_db", "ChEMBL"),
        "to": cfg["id_mapping"].get("to_db", "UniProtKB"),
        "ids": ",".join(ids),
    }
    resp = _request_with_retry(
        "post",
        url,
        timeout=timeout,
        rate_limiter=rate_limiter,
        max_attempts=retry_cfg["max_attempts"],
        backoff=retry_cfg["backoff_sec"],
        data=payload,
    )
    try:
        return resp.json()
    except json.JSONDecodeError:
        LOGGER.debug("Unparseable response from %s: %s", url, resp.text)
        raise


def _poll_job(
    job_id: str,
    cfg: Dict[str, Any],
    rate_limiter: RateLimiter,
    timeout: float,
    retry_cfg: Dict[str, Any],
) -> None:
    status_url = (
        cfg["base_url"].rstrip("/")
        + cfg["id_mapping"].get("status_endpoint", "")
        + "/"
        + job_id
    )
    interval = cfg["polling"]["interval_sec"]
    while True:
        resp = _request_with_retry(
            "get",
            status_url,
            timeout=timeout,
            rate_limiter=rate_limiter,
            max_attempts=retry_cfg["max_attempts"],
            backoff=retry_cfg["backoff_sec"],
        )
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
    cfg: Dict[str, Any],
    rate_limiter: RateLimiter,
    timeout: float,
    retry_cfg: Dict[str, Any],
) -> Dict[str, List[str]]:
    results_url = (
        cfg["base_url"].rstrip("/")
        + cfg["id_mapping"].get("results_endpoint", "")
        + "/"
        + job_id
    )
    resp = _request_with_retry(
        "get",
        results_url,
        timeout=timeout,
        rate_limiter=rate_limiter,
        max_attempts=retry_cfg["max_attempts"],
        backoff=retry_cfg["backoff_sec"],
    )
    try:
        payload = resp.json()
    except json.JSONDecodeError:
        LOGGER.debug("Unparseable results response: %s", resp.text)
        raise

    mapping: Dict[str, List[str]] = {}
    for item in payload.get("results", []):
        frm = item.get("from")
        to = item.get("to")
        if frm and to:
            mapping.setdefault(frm, []).append(to)
    return mapping


def _map_batch(
    ids: List[str],
    cfg: Dict[str, Any],
    rate_limiter: RateLimiter,
    timeout: float,
    retry_cfg: Dict[str, Any],
) -> Dict[str, List[str]]:
    try:
        job_payload = _start_job(ids, cfg, rate_limiter, timeout, retry_cfg)
    except Exception as exc:
        LOGGER.warning("Failed to start mapping job for batch %s: %s", ids, exc)
        return {}

    if "jobId" in job_payload:
        job_id = job_payload["jobId"]
        try:
            _poll_job(job_id, cfg, rate_limiter, timeout, retry_cfg)
            return _fetch_results(job_id, cfg, rate_limiter, timeout, retry_cfg)
        except Exception as exc:  # broad but logged
            LOGGER.warning("Job %s failed: %s", job_id, exc)
            return {}

    # Synchronous result
    if "results" in job_payload:
        mapping: Dict[str, List[str]] = {}
        for item in job_payload.get("results", []):
            frm = item.get("from")
            to = item.get("to")
            if frm and to:
                mapping.setdefault(frm, []).append(to)
        return mapping

    LOGGER.warning("Unexpected response payload: %s", job_payload)
    return {}


def map_chembl_to_uniprot(
    input_csv_path: str | Path,
    output_csv_path: str | Path | None = None,
    config_path: str | Path = "config.yaml",
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
        Path to the YAML configuration file which is validated against
        ``config.schema.json``.

    Returns
    -------
    Path
        Path to the written CSV file containing an extra column with the mapped
        UniProt identifiers.
    """

    cfg = load_and_validate_config(config_path).raw

    logging.basicConfig(level=getattr(logging, cfg["logging"]["level"].upper()))

    input_csv_path = Path(input_csv_path)
    if output_csv_path is None:
        output_csv_path = input_csv_path.with_name(
            input_csv_path.stem + "_with_uniprot.csv"
        )
    output_csv_path = Path(output_csv_path)

    sep = cfg["io"]["csv"]["separator"]
    encoding_in = cfg["io"]["input"]["encoding"]
    encoding_out = cfg["io"]["output"]["encoding"]
    chembl_col = cfg["columns"]["chembl_id"]
    out_col = cfg["columns"]["uniprot_out"]
    delimiter = cfg["io"]["csv"]["multivalue_delimiter"]

    # Compute SHA256 of input file for logging purposes
    with input_csv_path.open("rb") as fh:
        file_hash = hashlib.sha256(fh.read()).hexdigest()
    LOGGER.info("Input file checksum (sha256): %s", file_hash)

    df = pd.read_csv(input_csv_path, sep=sep, encoding=encoding_in)
    if chembl_col not in df.columns:
        raise ValueError(f"Missing required column '{chembl_col}' in input CSV")

    # Normalise and deduplicate identifiers
    ids_series = df[chembl_col].astype(str).map(lambda s: s.strip())
    ids_series = ids_series.replace({"": pd.NA}).dropna()
    unique_ids = list(ids_series.drop_duplicates())

    LOGGER.info("Processing %d unique ChEMBL IDs", len(unique_ids))

    batch_size = cfg["batch"]["size"]
    timeout = cfg["network"]["timeout_sec"]
    retry_cfg = cfg["uniprot"]["retry"]
    rate_limiter = RateLimiter(cfg["uniprot"]["rate_limit"]["rps"])

    mapping: Dict[str, List[str]] = {}
    for batch in _chunked(unique_ids, batch_size):
        batch_mapping = _map_batch(
            batch, cfg["uniprot"], rate_limiter, timeout, retry_cfg
        )
        mapping.update(batch_mapping)

    mapped = sum(1 for v in mapping.values() if v)
    no_match = len(unique_ids) - mapped
    multi = sum(1 for v in mapping.values() if len(v) > 1)
    LOGGER.info(
        "Summary: unique=%d mapped=%d no_match=%d multi=%d",
        len(unique_ids),
        mapped,
        no_match,
        multi,
    )

    def _join_ids(x: str | float | None) -> str | None:
        if x is None or x != x:  # NaN check
            return None
        ids = mapping.get(str(x).strip())
        if not ids:
            return None
        return delimiter.join(ids)

    df[out_col] = df[chembl_col].map(_join_ids)

    df.to_csv(output_csv_path, sep=sep, encoding=encoding_out, index=False)
    return output_csv_path
