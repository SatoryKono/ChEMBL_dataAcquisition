"""Unit tests for :mod:`chembl2uniprot.mapping`."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List

import pytest

from chembl2uniprot.config import (
    IdMappingConfig,
    PollingConfig,
    RateLimitConfig,
    RetryConfig,
    UniprotConfig,
)
from chembl2uniprot.mapping import RateLimiter, _fetch_results, _map_batch


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
