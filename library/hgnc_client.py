"""HGNC lookup utilities.

Algorithm Notes
---------------
1. Load YAML configuration specifying HGNC endpoint, network and output options.
2. Read the input CSV containing UniProt accessions and validate required columns.
3. Deduplicate the accessions and query the HGNC API for each unique identifier.
4. For matched genes, request the recommended protein name from UniProt.
5. Emit a CSV mapping each input accession to gene and protein metadata.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

 
from types import TracebackType
from typing import Any,Dict, List
import logging


import pandas as pd
import requests
import yaml

try:
    from data_profiling import analyze_table_quality
except ModuleNotFoundError:  # pragma: no cover
    from .data_profiling import analyze_table_quality

try:  # pragma: no cover - support package and script imports
    from .http_client import CacheConfig, HttpClient
except ImportError:  # pragma: no cover
    from http_client import CacheConfig, HttpClient  # type: ignore[no-redef]

try:  # pragma: no cover - support package and script imports
    from .chembl2uniprot.config import _apply_env_overrides
except ImportError:  # pragma: no cover
    from chembl2uniprot.config import _apply_env_overrides  # type: ignore[no-redef]

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses


@dataclass
class HGNCServiceConfig:
    """Endpoint configuration for HGNC REST API."""

    base_url: str


@dataclass
class NetworkConfig:
    """Network behaviour settings."""

    timeout_sec: float
    max_retries: int


@dataclass
class RateLimitConfig:
    """Rate limiting parameters."""

    rps: float


@dataclass
class OutputConfig:
    """CSV output formatting options."""

    sep: str
    encoding: str


@dataclass
class Config:
    """Aggregated application configuration.

    Attributes
    ----------
    hgnc:
        Endpoint configuration for the HGNC REST API.
    network:
        Network behaviour settings (timeout and retries).
    rate_limit:
        Rate limiting parameters applied to outbound requests.
    output:
        CSV output options controlling encoding and separator.
    cache:
        Optional HTTP cache configuration shared by network requests.
    """

    hgnc: HGNCServiceConfig
    network: NetworkConfig
    rate_limit: RateLimitConfig
    output: OutputConfig
    cache: CacheConfig | None = None


def load_config(path: str | Path, *, section: str | None = None) -> Config:
    """Load configuration from ``path``.

    Parameters
    ----------
    path:
        Location of the YAML configuration file.
    section:
        Optional top-level key selecting a subsection of the configuration.
    """

    with Path(path).open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        msg = f"Root of {path} must be a mapping"
        raise TypeError(msg)
    data: Dict[str, Any]
    if section:
        try:
            section_payload = loaded[section]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Section '{section}' not found in {path}") from exc
        if not isinstance(section_payload, dict):
            msg = f"Section '{section}' in {path} must be a mapping"
            raise TypeError(msg)
        data = dict(section_payload)
    else:
        data = dict(loaded)
    _apply_env_overrides(data, section=section)
    return Config(
        hgnc=HGNCServiceConfig(**data["hgnc"]),
        network=NetworkConfig(**data["network"]),
        rate_limit=RateLimitConfig(**data["rate_limit"]),
        output=OutputConfig(**data["output"]),
        cache=CacheConfig.from_dict(data.get("cache")),
    )


# ---------------------------------------------------------------------------
# Rate limiting and HTTP helpers


# ---------------------------------------------------------------------------
# HGNC and UniProt clients


@dataclass
class HGNCRecord:
    """Represents mapping information for a single UniProt accession.

    Attributes:
        uniprot_id: The UniProt accession number.
        hgnc_id: The HGNC identifier.
        gene_symbol: The official gene symbol.
        gene_name: The official gene name.
        protein_name: The recommended protein name from UniProt.
    """

    uniprot_id: str
    hgnc_id: str
    gene_symbol: str
    gene_name: str
    protein_name: str


class HGNCClient:
    """A minimal client for querying the HGNC API.

    Args:
        cfg: The configuration object for the client.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.http = HttpClient(
            timeout=cfg.network.timeout_sec,
            max_retries=cfg.network.max_retries,
            rps=cfg.rate_limit.rps,
            cache_config=cfg.cache,
        )

    def close(self) -> None:
        """Release HTTP resources associated with the client."""

        self.http.close()

    def __enter__(self) -> "HGNCClient":
        """Enter the managed context, returning ``self``."""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Ensure resources are released when leaving a context manager block."""

        self.close()

    def _request(self, url: str) -> requests.Response:
        """Perform a GET request with retry and rate limiting.

        Parameters
        ----------
        url:
            The URL to request.

        Returns
        -------
        requests.Response
            The HTTP response object.

        Raises
        ------
        requests.HTTPError
            If the request fails after all retries.
        """

        return self.http.request(
            "get",
            url,
            headers={"Accept": "application/json"},
        )

    def _fetch_protein_name(self, uniprot_id: str) -> str:
        """Fetch the recommended protein name from UniProt for a given ID.

        Parameters
        ----------
        uniprot_id:
            The UniProt accession number.

        Returns
        -------
        str
            The recommended protein name, or an empty string if not found.
        """

        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        try:
            resp = self._request(url)
        except Exception:  # pragma: no cover - network errors handled silently
            return ""
        if resp.status_code != 200:
            return ""
        try:
            data = resp.json()
        except ValueError:
            return ""
        return (
            data.get("proteinDescription", {})
            .get("recommendedName", {})
            .get("fullName", {})
            .get("value", "")
        )

    def fetch(self, uniprot_id: str) -> HGNCRecord:
        """Fetch HGNC mapping data for a single UniProt accession.

        This method queries the HGNC API for the given UniProt ID and also
        fetches the corresponding protein name from UniProt.

        Parameters
        ----------
        uniprot_id:
            The UniProt accession number to look up.

        Returns
        -------
        HGNCRecord
            A record containing the mapping information. If the lookup fails,
            the record will have empty fields.
        """

        url = f"{self.cfg.hgnc.base_url.rstrip('/')}/{uniprot_id}"
        resp = self._request(url)
        if resp.status_code != 200:
            LOGGER.warning(
                "HGNC lookup for %s failed with %s", uniprot_id, resp.status_code
            )
            return HGNCRecord(uniprot_id, "", "", "", "")
        try:
            payload = resp.json()
        except ValueError:
            LOGGER.warning("Unparseable HGNC response for %s", uniprot_id)
            return HGNCRecord(uniprot_id, "", "", "", "")
        docs = payload.get("response", {}).get("docs", [])
        if not docs:
            LOGGER.warning("No HGNC entry for %s", uniprot_id)
            return HGNCRecord(uniprot_id, "", "", "", "")
        doc = docs[0]
        protein = self._fetch_protein_name(uniprot_id)
        return HGNCRecord(
            uniprot_id=uniprot_id,
            hgnc_id=doc.get("hgnc_id", ""),
            gene_symbol=doc.get("symbol", ""),
            gene_name=doc.get("name", ""),
            protein_name=protein,
        )


# ---------------------------------------------------------------------------
# Public API


OUTPUT_COLUMNS = ["uniprot_id", "hgnc_id", "gene_symbol", "gene_name", "protein_name"]


def map_uniprot_to_hgnc(
    input_csv_path: Path,
    output_csv_path: Path | None,
    config_path: Path,
    *,
    config_section: str | None = None,
    column: str = "uniprot_id",
    sep: str | None = None,
    encoding: str | None = None,
    log_level: str = "INFO",
) -> Path:
    """Map UniProt accessions to HGNC identifiers.

    Parameters
    ----------
    input_csv_path:
        Path to the CSV file containing UniProt accessions.
    output_csv_path:
        Desired path for the output CSV.  When ``None`` a path derived from the
        input name is used.
    config_path:
        Path to the YAML configuration file.
    config_section:
        Optional top-level key selecting the configuration block within
        ``config_path``.
    column:
        Name of the column in ``input_csv_path`` holding UniProt accessions.
    sep, encoding:
        Optional overrides for CSV formatting.  Defaults come from the
        configuration file.
    log_level:
        Logging verbosity level (e.g. ``INFO`` or ``DEBUG``).

    Returns
    -------
    Path
        Path to the written CSV file.
    """

    cfg = load_config(config_path, section=config_section)
    level_name = log_level.upper()
    level_value = logging.getLevelName(level_name)
    if isinstance(level_value, str):
        msg = f"Unknown log level: {log_level!r}"
        raise ValueError(msg)
    LOGGER.setLevel(int(level_value))
    sep = sep or cfg.output.sep
    encoding = encoding or cfg.output.encoding

    df = pd.read_csv(input_csv_path, sep=sep, encoding=encoding)
    if column not in df.columns:
        raise ValueError(f"Missing column '{column}' in input")

    # Preserve order while deduplicating.
    raw_ids: List[str] = [str(v) for v in df[column]]
    unique_ids = list(dict.fromkeys(filter(None, raw_ids)))

    max_workers = max(1, int(cfg.rate_limit.rps))
    with HGNCClient(cfg) as client:
        if not unique_ids:
            mapping: Dict[str, HGNCRecord] = {}
        elif max_workers == 1 or len(unique_ids) == 1:
            mapping = {uid: client.fetch(uid) for uid in unique_ids}
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = executor.map(client.fetch, unique_ids)
                mapping = dict(zip(unique_ids, results))

    rows = [
        asdict(mapping.get(uid, HGNCRecord(uid, "", "", "", ""))) for uid in raw_ids
    ]
    out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    if output_csv_path is None:
        output_csv_path = input_csv_path.with_name(f"hgnc_{input_csv_path.stem}.csv")
    out_df.to_csv(output_csv_path, sep=sep, encoding=encoding, index=False)
    analyze_table_quality(out_df, table_name=str(Path(output_csv_path).with_suffix("")))
    return output_csv_path
