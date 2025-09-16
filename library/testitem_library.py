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

from dataclasses import dataclass
import logging
import re
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence
from urllib.parse import quote

import pandas as pd
import requests  # type: ignore[import-untyped]

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


@dataclass(slots=True)
class _PubChemRequest:
    base_url: str
    user_agent: str
    timeout: float
    session: requests.Session
    properties: Sequence[str]

    def fetch(self, smiles: str) -> Mapping[str, object]:
        encoded = quote(smiles, safe="")
        url = (
            f"{self.base_url.rstrip('/')}/compound/smiles/{encoded}/property/"
            f"{','.join(self.properties)}/JSON"
        )
        headers = {"Accept": "application/json", "User-Agent": self.user_agent}
        try:
            response = self.session.get(url, timeout=self.timeout, headers=headers)
            if response.status_code == 404:
                LOGGER.debug("PubChem returned 404 for SMILES %s", smiles)
                return {}
            response.raise_for_status()
        except requests.HTTPError:
            LOGGER.warning(
                "HTTP error when requesting PubChem properties for %s", smiles
            )
            return {}
        except requests.RequestException:
            LOGGER.warning(
                "Network error when requesting PubChem properties for %s", smiles
            )
            return {}

        payload = response.json()
        properties = payload.get("PropertyTable", {}).get("Properties", [])
        if not properties:
            return {}
        record = properties[0]
        return {
            prop: _normalise_numeric(prop, record.get(prop)) for prop in self.properties
        }


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
    session: requests.Session | None = None,
    base_url: str = PUBCHEM_BASE_URL,
    user_agent: str = "ChEMBLDataAcquisition/1.0",
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
    session:
        Optional :class:`requests.Session` to reuse connections during testing.
        When omitted a temporary session is created and closed automatically.
    base_url:
        Base URL of the PubChem PUG REST API.
    user_agent:
        Custom ``User-Agent`` header sent with each HTTP request.

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

    owns_session = session is None
    http = session or requests.Session()
    request = _PubChemRequest(
        base_url=base_url,
        user_agent=user_agent,
        timeout=timeout,
        session=http,
        properties=properties,
    )

    columns_map = _prepare_columns(properties)
    enriched = df.copy()
    for column in columns_map.values():
        if column not in enriched.columns:
            enriched[column] = pd.NA

    cache: MutableMapping[str, Mapping[str, object]] = {}
    try:
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
    finally:
        if owns_session:
            http.close()

    return enriched


__all__ = ["add_pubchem_data", "PUBCHEM_PROPERTIES"]
