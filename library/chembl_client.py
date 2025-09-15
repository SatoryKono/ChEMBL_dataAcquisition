"""Client utilities for retrieving ChEMBL document records."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Sequence
from typing import Any, Dict, List

import pandas as pd

from .http_client import HttpClient


@dataclass
class ApiCfg:
    """Minimal configuration for the ChEMBL API."""

    chembl_base: str = "https://www.ebi.ac.uk/chembl/api/data"
    timeout_read: float = 30.0


class ChemblClient:
    """Thin wrapper around :class:`HttpClient` for JSON requests."""

    def __init__(self, http: HttpClient) -> None:
        self.http = http

    def request_json(self, url: str, *, cfg: ApiCfg, timeout: float) -> Dict[str, Any]:
        """Return JSON payload for ``url`` using ``http``."""

        resp = self.http.request("get", url)
        resp.raise_for_status()
        return resp.json()


def _chunked(seq: Sequence[str], size: int) -> Iterable[List[str]]:
    """Yield ``seq`` in chunks of ``size``."""

    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


DOCUMENT_COLUMNS = [
    "document_chembl_id",
    "title",
    "abstract",
    "doi",
    "year",
    "journal",
    "journal_abbrev",
    "volume",
    "issue",
    "first_page",
    "last_page",
    "pubmed_id",
    "authors",
    "source",
]


def get_documents(
    ids: Iterable[str],
    *,
    cfg: ApiCfg,
    client: ChemblClient,
    chunk_size: int = 5,
    timeout: float | None = None,
) -> pd.DataFrame:
    """Fetch document records for ``ids`` from the ChEMBL API."""

    valid = [i for i in ids if i not in {"", "#N/A"}]
    unique_ids = list(dict.fromkeys(valid))
    if not unique_ids:
        return pd.DataFrame(columns=DOCUMENT_COLUMNS)

    base = f"{cfg.chembl_base.rstrip('/')}/document.json?format=json"
    effective_timeout = timeout if timeout is not None else cfg.timeout_read
    records: list[dict[str, Any]] = []

    for chunk in _chunked(unique_ids, chunk_size):
        url = f"{base}&document_chembl_id__in={','.join(chunk)}"
        data = client.request_json(url, cfg=cfg, timeout=effective_timeout)
        items = data.get("documents") or data.get("document") or []
        for item in items:
            record = {
                "document_chembl_id": item.get("document_chembl_id"),
                "title": item.get("title"),
                "abstract": item.get("abstract"),
                "doi": item.get("doi"),
                "year": item.get("year"),
                "journal": item.get("journal_full_title"),
                "journal_abbrev": item.get("journal"),
                "volume": item.get("volume"),
                "issue": item.get("issue"),
                "first_page": item.get("first_page"),
                "last_page": item.get("last_page"),
                "pubmed_id": item.get("pubmed_id"),
                "authors": item.get("authors"),
                "source": "ChEMBL",
            }
            records.append(record)

    if not records:
        return pd.DataFrame(columns=DOCUMENT_COLUMNS)

    df = pd.DataFrame(records)
    return df.reindex(columns=DOCUMENT_COLUMNS)


__all__ = ["ApiCfg", "ChemblClient", "get_documents"]
