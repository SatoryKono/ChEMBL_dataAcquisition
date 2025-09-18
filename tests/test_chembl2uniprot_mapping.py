"""Unit tests for :mod:`chembl2uniprot.mapping`."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List
from unittest import mock

import pytest
import requests

from chembl2uniprot.config import (
    IdMappingConfig,
    PollingConfig,
    RateLimitConfig,
    RetryConfig,
    UniprotConfig,
)
from chembl2uniprot.mapping import (
    RateLimiter,
    _fetch_results,
    _map_batch,
    _request_with_retry,
)


@dataclass
class _DummyResponse:
    """Simple stand-in for :class:`requests.Response` objects."""

    payload: Dict[str, Any]

    def __post_init__(self) -> None:
        self.status_code = 200
        self.text = json.dumps(self.payload)

    def json(self) -> Dict[str, Any]:
        return self.payload


@pytest.fixture
def uniprot_cfg() -> UniprotConfig:
    """Return a minimal UniProt configuration for tests."""

    return UniprotConfig(
        base_url="https://rest.uniprot.org",
        id_mapping=IdMappingConfig(
            endpoint="/idmapping/run",
            status_endpoint="/idmapping/status",
            results_endpoint="/idmapping/results",
        ),
        polling=PollingConfig(interval_sec=0.0),
        rate_limit=RateLimitConfig(rps=10.0),
        retry=RetryConfig(max_attempts=1, backoff_sec=0.0),
    )


def _make_request_stub(responses: Iterator[_DummyResponse]):
    """Create a stub replacing :func:`_request_with_retry`."""

    def _stub(
        method: str,
        url: str,
        *,
        timeout: float,
        rate_limiter: RateLimiter,
        max_attempts: int,
        backoff: float,
        **_: Any,
    ) -> _DummyResponse:
        try:
            response = next(responses)
        except StopIteration:  # pragma: no cover - defensive guard
            raise AssertionError("Unexpected extra request for %s" % url) from None
        return response

    return _stub


def test_fetch_results_handles_pagination(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    uniprot_cfg: UniprotConfig,
) -> None:
    """All result pages should be consumed and combined."""

    payloads = iter(
        [
            _DummyResponse(
                {
                    "results": [{"from": "CHEMBL1", "to": "P1"}],
                    "next": "/idmapping/results/123?cursor=abc",
                }
            ),
            _DummyResponse(
                {
                    "results": [{"from": "CHEMBL2", "to": "P2"}],
                    "failedIds": ["CHEMBL3"],
                }
            ),
        ]
    )
    requested_urls: List[str] = []

    def _tracking_stub(
        method: str,
        url: str,
        *,
        timeout: float,
        rate_limiter: RateLimiter,
        max_attempts: int,
        backoff: float,
        **kwargs: Any,
    ) -> _DummyResponse:
        requested_urls.append(url)
        return _make_request_stub(payloads)(
            method,
            url,
            timeout=timeout,
            rate_limiter=rate_limiter,
            max_attempts=max_attempts,
            backoff=backoff,
            **kwargs,
        )

    monkeypatch.setattr("chembl2uniprot.mapping._request_with_retry", _tracking_stub)

    with caplog.at_level(logging.WARNING):
        result = _fetch_results(
            "123",
            uniprot_cfg,
            RateLimiter(0),
            timeout=1.0,
            retry_cfg=RetryConfig(max_attempts=1, backoff_sec=0.0),
        )

    assert result.mapping == {"CHEMBL1": ["P1"], "CHEMBL2": ["P2"]}
    assert result.failed_ids == ["CHEMBL3"]
    assert requested_urls == [
        "https://rest.uniprot.org/idmapping/results/123",
        "https://rest.uniprot.org/idmapping/results/123?cursor=abc",
    ]
    assert "CHEMBL3" in caplog.text


def test_fetch_results_raises_when_failed_threshold_exceeded(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    uniprot_cfg: UniprotConfig,
) -> None:
    """A RuntimeError should be raised when too many IDs fail."""

    responses = iter(
        [
            _DummyResponse({"results": [], "failedIds": ["CHEMBL1", "CHEMBL2"]}),
        ]
    )
    monkeypatch.setattr(
        "chembl2uniprot.mapping._request_with_retry",
        _make_request_stub(responses),
    )
    monkeypatch.setattr("chembl2uniprot.mapping.FAILED_IDS_ERROR_THRESHOLD", 1)

    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError):
        _fetch_results(
            "job-5",
            uniprot_cfg,
            RateLimiter(0),
            timeout=1.0,
            retry_cfg=RetryConfig(max_attempts=1, backoff_sec=0.0),
        )

    assert "CHEMBL1" in caplog.text
    assert "job-5" in caplog.text


def test_map_batch_returns_failed_ids(
    monkeypatch: pytest.MonkeyPatch, uniprot_cfg: UniprotConfig
) -> None:
    """Synchronous responses should expose failed identifiers in the result."""

    monkeypatch.setattr(
        "chembl2uniprot.mapping._start_job",
        lambda *_, **__: {  # type: ignore[misc]
            "results": [{"from": "CHEMBL1", "to": "P1"}],
            "failedIds": ["CHEMBL2"],
        },
    )

    result = _map_batch(
        ["CHEMBL1", "CHEMBL2"],
        uniprot_cfg,
        RateLimiter(0),
        timeout=1.0,
        retry_cfg=RetryConfig(max_attempts=1, backoff_sec=0.0),
    )

    assert result.mapping == {"CHEMBL1": ["P1"]}
    assert result.failed_ids == ["CHEMBL2"]


def test_request_with_retry_honours_retry_after(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """429 responses should trigger retries and rate limiter penalties."""

    monkeypatch.setattr("tenacity.nap.sleep", lambda _: None)
    monkeypatch.setattr("library.http_client.time.sleep", lambda _: None)

    class _StubResponse:
        def __init__(
            self,
            status_code: int,
            *,
            headers: Dict[str, str] | None = None,
            payload: Dict[str, Any] | None = None,
        ) -> None:
            self.status_code = status_code
            self.headers = headers or {}
            self._payload = payload or {}
            self.text = json.dumps(self._payload)

        def json(self) -> Dict[str, Any]:
            return self._payload

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                err = requests.HTTPError(f"HTTP {self.status_code}")
                err.response = self
                raise err

    responses = iter(
        [
            _StubResponse(429, headers={"Retry-After": "0.2"}),
            _StubResponse(200, payload={"ok": True}),
        ]
    )
    call_log: List[str] = []

    def _request_stub(
        method: str, url: str, *, timeout: float, **kwargs: Any
    ) -> _StubResponse:
        call_log.append(f"{method}:{url}:{timeout}")
        try:
            return next(responses)
        except StopIteration:  # pragma: no cover - defensive guard
            raise AssertionError("Unexpected extra request") from None

    monkeypatch.setattr("chembl2uniprot.mapping.requests.request", _request_stub)

    rate_limiter = RateLimiter(1000.0)

    with (
        mock.patch.object(
            rate_limiter,
            "apply_penalty",
            wraps=rate_limiter.apply_penalty,
        ) as penalty_mock,
        caplog.at_level(logging.WARNING),
    ):
        response = _request_with_retry(
            "get",
            "https://rest.uniprot.org/test",
            timeout=1.0,
            rate_limiter=rate_limiter,
            max_attempts=3,
            backoff=0.1,
        )

    assert response.json() == {"ok": True}
    assert len(call_log) == 2
    assert penalty_mock.call_count >= 1
    assert any(
        call.args and pytest.approx(0.2, rel=0.05) == call.args[0]
        for call in penalty_mock.call_args_list
    )
    assert any("attempt" in record.message for record in caplog.records)
