from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Any

from library.hgnc_client import (
    Config,
    HGNCClient,
    HGNCServiceConfig,
    NetworkConfig,
    OutputConfig,
    RateLimitConfig,
)


class FakeResponse:
    """Minimal stub implementing the subset of ``requests.Response`` used in tests."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.status_code = 200
        self.headers: dict[str, str] = {}

    def json(self) -> dict[str, Any]:
        accession = self.url.rsplit("/", 1)[-1]
        if accession.endswith(".json"):
            accession = accession[:-5]
        if "rest.uniprot.org" in self.url:
            return {
                "proteinDescription": {
                    "recommendedName": {"fullName": {"value": f"Protein {accession}"}}
                }
            }
        return {
            "response": {
                "docs": [
                    {
                        "symbol": f"SYM_{accession}",
                        "name": f"Gene {accession}",
                        "hgnc_id": f"HGNC:{accession}",
                    }
                ]
            }
        }

    def raise_for_status(self) -> None:  # pragma: no cover - compatibility shim
        return None


class FakeSession:
    """Thread-bound HTTP session tracking the worker that created it."""

    def __init__(self) -> None:
        self.creator = threading.get_ident()
        self.closed = False
        self.request_threads: list[int] = []

    def request(
        self,
        method: str,
        url: str,
        *,
        timeout: float | tuple[float, float] | None = None,
        **_: Any,
    ) -> FakeResponse:
        thread_id = threading.get_ident()
        self.request_threads.append(thread_id)
        if thread_id != self.creator:
            raise AssertionError("Session used across threads")
        return FakeResponse(url)

    def close(self) -> None:
        self.closed = True


def test_parallel_fetch_uses_thread_local_sessions(monkeypatch) -> None:
    """Ensure HGNCClient creates one HTTP session per worker thread."""

    created_sessions: list[FakeSession] = []

    def fake_create_http_session(_cache_config):
        session = FakeSession()
        created_sessions.append(session)
        return session

    monkeypatch.setattr(
        "library.http_client.create_http_session", fake_create_http_session
    )

    cfg = Config(
        hgnc=HGNCServiceConfig(base_url="https://example.org/hgnc"),
        network=NetworkConfig(timeout_sec=1.0, max_retries=1),
        rate_limit=RateLimitConfig(rps=3.0),
        output=OutputConfig(sep=",", encoding="utf-8"),
        cache=None,
    )

    accessions = ["P1", "P2", "P3", "P4"]

    with HGNCClient(cfg) as client:
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(client.fetch, accessions))

    assert {record.uniprot_id for record in results} == set(accessions)
    assert all(session.closed for session in created_sessions)
    assert len(created_sessions) >= 2
    for session in created_sessions:
        assert all(
            thread_id == session.creator for thread_id in session.request_threads
        )
        assert session.request_threads, "Session was created but never used"
