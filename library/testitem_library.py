"""Utilities for enriching ChEMBL molecules with external annotations.

The :func:`add_pubchem_data` helper augments a molecule table with
properties sourced from the PubChem PUG REST API.  The function only
performs one HTTP request per unique SMILES string and caches responses to
minimise network traffic.

Algorithm Notes
---------------
1. Extract all unique, non-empty SMILES strings from the input DataFrame.
2. For each SMILES string query PubChem for the requested property fields.
3. Map the returned payload to deterministic ``pubchem_*`` columns.
4. Populate the original DataFrame with the enriched values, leaving
   missing entries untouched when the API does not return a match.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence
from urllib.parse import quote

import pandas as pd
import requests  # type: ignore[import-untyped]

try:  # pragma: no cover - импорт для тестовых окружений без пакета
    from .http_client import HttpClient
except ImportError:  # pragma: no cover
    from http_client import HttpClient  # type: ignore[no-redef]

LOGGER = logging.getLogger(__name__)

PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_PROPERTIES: tuple[str, ...] = (
    "CID",
    "MolecularFormula",
    "MolecularWeight",
    "TPSA",
    "XLogP",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
)
PUBCHEM_INT_FIELDS: frozenset[str] = frozenset(
    ["CID", "HBondDonorCount", "HBondAcceptorCount", "RotatableBondCount"]
)
PUBCHEM_FLOAT_FIELDS: frozenset[str] = frozenset(["MolecularWeight", "TPSA", "XLogP"])

PUBCHEM_DEFAULT_MAX_RETRIES = 3
PUBCHEM_DEFAULT_RPS = 5.0
PUBCHEM_DEFAULT_BACKOFF = 1.0
PUBCHEM_DEFAULT_RETRY_PENALTY = 5.0
PUBCHEM_STATUS_FORCELIST: tuple[int, ...] = (
    408,
    409,
    429,
    500,
    502,
    503,
    504,
)


def _to_snake_case(value: str) -> str:
    step1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    step2 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", step1)
    return step2.replace("-", "_").lower()


def _normalise_numeric(property_name: str, value: Any) -> Any:
    if value is None:
        return None
    if property_name in PUBCHEM_INT_FIELDS:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return int(float(stripped))
            except ValueError:
                return None
        if isinstance(value, (int, float)):
            return int(value)
        return None
    if property_name in PUBCHEM_FLOAT_FIELDS:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return float(stripped)
            except ValueError:
                return None
        if isinstance(value, (int, float)):
            return float(value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


def _prepare_columns(properties: Sequence[str]) -> Dict[str, str]:
    return {prop: f"pubchem_{_to_snake_case(prop)}" for prop in properties}


def _int_from_config(config: Mapping[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"{key} must be an integer, got {value!r}") from exc


def _float_from_config(config: Mapping[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"{key} must be a float, got {value!r}") from exc


def _status_forcelist_from_config(
    config: Mapping[str, Any], key: str, default: tuple[int, ...]
) -> tuple[int, ...]:
    value = config.get(key)
    if value is None:
        return default
    if isinstance(value, (str, bytes)):
        raise TypeError("status_forcelist must be an iterable of integers")
    try:
        iterator = iter(value)
    except TypeError as exc:  # pragma: no cover - defensive branch
        raise TypeError("status_forcelist must be an iterable of integers") from exc
    result: list[int] = []
    for item in iterator:
        try:
            result.append(int(item))
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive branch
            raise ValueError("status_forcelist entries must be integers") from exc
    return tuple(result)


def _build_http_client(
    *,
    timeout: float,
    http_client: HttpClient | None,
    http_client_config: Mapping[str, Any] | None,
    session: requests.Session | None,
) -> HttpClient:
    if http_client is not None:
        return http_client
    config = http_client_config or {}
    max_retries = _int_from_config(config, "max_retries", PUBCHEM_DEFAULT_MAX_RETRIES)
    rps = _float_from_config(config, "rps", PUBCHEM_DEFAULT_RPS)
    backoff = _float_from_config(config, "backoff_multiplier", PUBCHEM_DEFAULT_BACKOFF)
    penalty = _float_from_config(
        config, "retry_penalty_seconds", PUBCHEM_DEFAULT_RETRY_PENALTY
    )
    status_forcelist = _status_forcelist_from_config(
        config, "status_forcelist", PUBCHEM_STATUS_FORCELIST
    )

    return HttpClient(
        timeout=(timeout, timeout),
        max_retries=max_retries,
        rps=rps,
        backoff_multiplier=backoff,
        retry_penalty_seconds=penalty,
        status_forcelist=status_forcelist,
        session=session,
    )


@dataclass(slots=True)
class _PubChemRequest:
    base_url: str
    user_agent: str
    timeout: float
    properties: Sequence[str]
    http_client: HttpClient

    @property
    def session(self) -> requests.Session:
        """Expose the underlying :class:`requests.Session` used by the client."""

        return self.http_client.session

    def _get_json(
        self,
        url: str,
        *,
        smiles: str,
        context: str,
    ) -> Mapping[str, Any] | None:
        """Perform a GET request and decode the JSON payload.

        Parameters
        ----------
        url:
            Fully qualified PubChem endpoint.
        smiles:
            SMILES string used for logging.
        context:
            Short description of the request type (e.g. ``"properties"``).

        Returns
        -------
        Mapping[str, Any] | None
            Parsed JSON payload or ``None`` when the request fails.
        """

        headers = {"Accept": "application/json", "User-Agent": self.user_agent}
        try:

            response = self.session.get(url, timeout=self.timeout, headers=headers)
            if response.status_code == 404:
                LOGGER.debug(
                    "PubChem returned 404 for %s while fetching %s", smiles, context
                )
                return None
            response.raise_for_status()
        except requests.HTTPError:
            LOGGER.warning(
                "HTTP error when requesting PubChem %s for %s", context, smiles
            )
            return None
        except requests.RequestException:
            LOGGER.warning(
                "Network error when requesting PubChem %s for %s", context, smiles
            )
            return None

        try:
            return response.json()
        except ValueError:
            LOGGER.warning(
                "Failed to decode JSON response from PubChem %s for %s",
                context,
                smiles,
            )
            return None

    def fetch(self, smiles: str) -> Mapping[str, object]:
        encoded = quote(smiles, safe="")
        results: dict[str, object] = {}

        property_fields = list(dict.fromkeys(self.properties))
        if property_fields:
            url = (
                f"{self.base_url.rstrip('/')}/compound/smiles/{encoded}/property/"
                f"{','.join(property_fields)}/JSON"
            )
            payload = self._get_json(url, smiles=smiles, context="properties")
            if payload is not None:
                properties = payload.get("PropertyTable", {}).get("Properties", [])
                if properties:
                    record = properties[0]
                    for prop in property_fields:
                        results[prop] = _normalise_numeric(prop, record.get(prop))

        if "CID" in self.properties and results.get("CID") is None:
            cid_url = f"{self.base_url.rstrip('/')}/compound/smiles/{encoded}/cids/JSON"
            payload = self._get_json(cid_url, smiles=smiles, context="CID list")
            if payload is not None:
                identifiers = payload.get("IdentifierList", {}).get("CID", [])
                if identifiers:
                    results["CID"] = _normalise_numeric("CID", identifiers[0])

        return results


def _unique_smiles(values: Iterable[object]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for raw in values:
        if raw is None:
            continue
        candidate = str(raw).strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    unique.sort()
    return unique


def add_pubchem_data(
    df: pd.DataFrame,
    *,
    smiles_column: str = "canonical_smiles",
    properties: Sequence[str] = PUBCHEM_PROPERTIES,
    timeout: float = 10.0,
    base_url: str = PUBCHEM_BASE_URL,
    user_agent: str = "ChEMBLDataAcquisition/1.0",
    http_client: HttpClient | None = None,
    http_client_config: Mapping[str, Any] | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Augment ``df`` with PubChem descriptors based on SMILES strings.

    Parameters
    ----------
    df:
        Input molecule table.  The function returns ``df`` unchanged when the
        frame is empty or does not contain ``smiles_column``.
    smiles_column:
        Name of the column containing SMILES representations used to query the
        PubChem service.
    properties:
        Sequence of property names requested from the API.  Defaults to the
        ``PUBCHEM_PROPERTIES`` tuple defined in this module.
    timeout:
        Socket timeout for PubChem HTTP requests in seconds.
    base_url:
        Base URL of the PubChem PUG REST API.
    user_agent:
        Custom ``User-Agent`` header sent with each HTTP request.
    http_client:
        Optional :class:`~library.http_client.HttpClient` used for outbound
        requests. When omitted a new client is created with sensible defaults
        for retry and rate limiting.
    http_client_config:
        Optional mapping overriding the HTTP client defaults. Supported keys are
        ``max_retries``, ``rps``, ``backoff_multiplier``,
        ``retry_penalty_seconds`` and ``status_forcelist``. Ignored when
        ``http_client`` is provided.
    session:
        Optional :class:`requests.Session` reused when a new
        :class:`HttpClient` is instantiated internally. Retained for backwards
        compatibility with older code paths.

    Returns
    -------
    pandas.DataFrame
        A copy of the input frame enriched with ``pubchem_*`` columns.  Missing
        annotations remain represented as ``pd.NA`` values.
    """

    if df.empty:
        LOGGER.debug("Skipping PubChem enrichment for empty DataFrame")
        return df
    if smiles_column not in df.columns:
        LOGGER.warning(
            "SMILES column '%s' not found; skipping PubChem enrichment", smiles_column
        )
        return df
    if not properties:
        LOGGER.debug("No PubChem properties requested; returning original DataFrame")
        return df

    http = _build_http_client(
        timeout=timeout,
        http_client=http_client,
        http_client_config=http_client_config,
        session=session,
    )
    request = _PubChemRequest(
        base_url=base_url,
        user_agent=user_agent,
        timeout=timeout,
        properties=properties,
        http_client=http,
    )

    columns_map = _prepare_columns(properties)
    enriched = df.copy()
    for column in columns_map.values():
        if column not in enriched.columns:
            enriched[column] = pd.NA

    cache: MutableMapping[str, Mapping[str, object]] = {}
    for smiles in _unique_smiles(enriched[smiles_column].tolist()):
        cache[smiles] = request.fetch(smiles)

    for idx, raw_smiles in enriched[smiles_column].items():
        smiles_value = None if raw_smiles is None else str(raw_smiles).strip()
        if not smiles_value:
            continue
        data = cache.get(smiles_value, {})
        for prop, column in columns_map.items():
            value = data.get(prop)
            if value is not None:
                enriched.at[idx, column] = value

    return enriched


__all__ = ["add_pubchem_data", "PUBCHEM_PROPERTIES"]
