from __future__ import annotations

import importlib
from typing import Any, cast
from unittest.mock import Mock

import requests

testitem_library_module = importlib.import_module("testitem_library")

_PubChemRequest = testitem_library_module._PubChemRequest


def _mock_response(payload: dict[str, Any]) -> requests.Response:
    response = Mock(spec=requests.Response)
    response.status_code = 200
    response.json.return_value = payload
    return cast(requests.Response, response)


def test_pubchem_request_uses_http_client_and_handles_errors() -> None:
    http_client_mock = Mock()

    success_payload = {"foo": "bar"}
    success_response = _mock_response(success_payload)

    not_found_response = Mock(spec=requests.Response)
    not_found_response.status_code = 404
    not_found_error = requests.HTTPError("Not Found", response=not_found_response)

    invalid_json_response = Mock(spec=requests.Response)
    invalid_json_response.status_code = 200
    invalid_json_response.json.side_effect = ValueError("invalid json")

    http_client_mock.request.side_effect = [
        success_response,
        not_found_error,
        cast(requests.Response, invalid_json_response),
    ]

    request = _PubChemRequest(
        base_url="https://example.test",
        user_agent="agent",
        timeout=1.0,
        properties=(),
        http_client=http_client_mock,
    )

    url = "https://example.test/resource"
    result = request._get_json(url, smiles="C", context="properties")
    assert result == success_payload

    call = http_client_mock.request.call_args_list[0]
    assert call.args == ("get", url)
    assert call.kwargs == {"headers": {"Accept": "application/json", "User-Agent": "agent"}}

    not_found = request._get_json(url, smiles="C", context="properties")
    assert not_found is None

    invalid_result = request._get_json(
        "https://example.test/invalid", smiles="C", context="properties"
    )
    assert invalid_result is None

    assert http_client_mock.request.call_count == 3
