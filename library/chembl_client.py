"""HTTP client for retrieving ChEMBL records."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Sequence
from typing import Any, Callable, Dict, List

import pandas as pd
import requests  # type: ignore[import-untyped]

try:  # pragma: no cover - поддержка импорта без контекста пакета
    from .http_client import CacheConfig, HttpClient
except ImportError:  # pragma: no cover
    from http_client import CacheConfig, HttpClient  # type: ignore[no-redef]

import logging
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


@dataclass
class ApiCfg:
    """Минимальная конфигурация для ChEMBL API."""

    chembl_base: str = "https://www.ebi.ac.uk/chembl/api/data"
    timeout_read: float = 30.0


@dataclass
class ChemblClient:
    """Клиент для получения данных из ChEMBL API."""

    base_url: str = "https://www.ebi.ac.uk/chembl/api/data"
    timeout: float = 30.0
    max_retries: int = 3
    rps: float = 2.0
    user_agent: str = "ChEMBLDataAcquisition/1.0"
    cache_config: CacheConfig | None = None
    http_client: HttpClient | None = None

    def __post_init__(self) -> None:
        self._http = self.http_client or HttpClient(
            timeout=self.timeout,
            max_retries=self.max_retries,
            rps=self.rps,
            cache_config=self.cache_config,
        )

    def _fetch_resource(
        self, resource: str, identifier: str, *, id_field: str
    ) -> Dict[str, Any] | None:
        url = f"{self.base_url.rstrip('/')}/{resource}/{identifier}.json"
        headers = {
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }
        try:
            response = self._http.request("get", url, headers=headers)
            if response.status_code == 404:
                LOGGER.warning(
                    "%s %s не найден (404)", resource.capitalize(), identifier
                )
                return None
            response.raise_for_status()
        except requests.HTTPError:
            LOGGER.exception("Не удалось получить %s %s", resource, identifier)
            raise
        except requests.RequestException:
            LOGGER.exception("Сетевая ошибка при получении %s %s", resource, identifier)
            raise

        payload: Dict[str, Any] = response.json()
        payload.setdefault(id_field, identifier)
        return payload

    def fetch_assay(self, assay_id: str) -> Dict[str, Any] | None:
        """Вернуть JSON для ``assay_id``."""

        return self._fetch_resource("assay", assay_id, id_field="assay_chembl_id")

    def fetch_activity(self, activity_id: str) -> Dict[str, Any] | None:
        """Вернуть JSON для ``activity_id``."""

        return self._fetch_resource(
            "activity", activity_id, id_field="activity_chembl_id"
        )

    def _fetch_many(
        self,
        identifiers: Iterable[str],
        fetcher: Callable[[str], Dict[str, Any] | None],
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for identifier in identifiers:
            payload = fetcher(identifier)
            if payload is not None:
                records.append(payload)
        return records

    def fetch_many(self, assay_ids: Iterable[str]) -> List[Dict[str, Any]]:
        """Получить несколько payloads для анализов."""

        return self._fetch_many(assay_ids, self.fetch_assay)

    def fetch_many_activities(
        self, activity_ids: Iterable[str]
    ) -> List[Dict[str, Any]]:
        """Получить несколько payloads для активностей."""

        return self._fetch_many(activity_ids, self.fetch_activity)

    def request_json(self, url: str, *, cfg: ApiCfg, timeout: float) -> Dict[str, Any]:
        """Вернуть JSON payload для ``url`` используя http."""

        resp = self._http.request("get", url, timeout=(timeout, timeout))
        resp.raise_for_status()
        return resp.json()


def _chunked(seq: Sequence[str], size: int) -> Iterable[List[str]]:
    """Вернуть последовательность по частям размера ``size``."""

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
    """Получить записи документов для ``ids`` из ChEMBL API."""

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
