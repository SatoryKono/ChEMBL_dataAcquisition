import logging
import sys
from pathlib import Path
from typing import Dict
from unittest import mock

import pytest
import requests
import requests_mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from library.uniprot_client import NetworkConfig, RateLimitConfig, UniProtClient


def test_fetch_entries_json_batches(requests_mock: requests_mock.Mocker) -> None:
    client = UniProtClient(
        base_url="https://rest.uniprot.org/uniprotkb",
        fields="",
        network=NetworkConfig(timeout_sec=1, max_retries=1),
        rate_limit=RateLimitConfig(rps=1000),
    )
    url = "https://rest.uniprot.org/uniprotkb/stream"
    requests_mock.get(
        url,
        json={
            "results": [
                {"primaryAccession": "P1"},
                {"primaryAccession": "P2"},
            ]
        },
    )
    res: Dict[str, Dict[str, str]] = client.fetch_entries_json(["P1", "P2"])
    assert requests_mock.call_count == 1
    assert set(res.keys()) == {"P1", "P2"}


def test_fetch_respects_retry_after(
    requests_mock: requests_mock.Mocker,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """HTTP 429 responses should trigger retries honouring Retry-After."""

    monkeypatch.setattr("tenacity.nap.sleep", lambda _: None)
    monkeypatch.setattr("library.http_client.time.sleep", lambda _: None)

    client = UniProtClient(
        base_url="https://rest.uniprot.org/uniprotkb",
        fields="",
        network=NetworkConfig(timeout_sec=1, max_retries=3, backoff_sec=0.1),
        rate_limit=RateLimitConfig(rps=1000),
    )
    url = "https://rest.uniprot.org/uniprotkb/search"
    requests_mock.get(
        url,
        [
            {
                "status_code": 429,
                "headers": {"Retry-After": "0.2"},
                "json": {"error": "rate limit"},
            },
            {
                "status_code": 200,
                "json": {"results": [{"primaryAccession": "P12345"}]},
            },
        ],
    )

    with (
        mock.patch.object(
            client._rate_limiter,  # type: ignore[attr-defined]
            "apply_penalty",
            wraps=client._rate_limiter.apply_penalty,  # type: ignore[attr-defined]
        ) as penalty_mock,
        caplog.at_level(logging.WARNING),
    ):
        result = client.fetch("P12345")

    assert result == {"primaryAccession": "P12345"}
    assert requests_mock.call_count == 2
    assert penalty_mock.call_count >= 1
    assert any(
        call.args and pytest.approx(0.2, rel=0.05) == call.args[0]
        for call in penalty_mock.call_args_list
    )
    assert any("attempt" in record.message for record in caplog.records)


def test_uniprot_client_retries_configured_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client performs ``max_retries`` attempts after the initial request."""

    monkeypatch.setattr("tenacity.nap.sleep", lambda *_: None)
    monkeypatch.setattr("library.http_client.time.sleep", lambda *_: None)
    session = mock.create_autospec(requests.Session, instance=True)
    session.get.side_effect = requests.exceptions.ConnectTimeout("boom")
    client = UniProtClient(
        base_url="https://rest.uniprot.org/uniprotkb",
        fields="",
        network=NetworkConfig(timeout_sec=1, max_retries=2, backoff_sec=0.1),
        rate_limit=RateLimitConfig(rps=1000),
        session=session,
    )

    response = client._request(
        "https://rest.uniprot.org/uniprotkb/search",
        {"query": "accession:P12345"},
    )

    assert response is None
    assert session.get.call_count == 3
