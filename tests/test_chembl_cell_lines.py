from __future__ import annotations

from pathlib import Path
import sys

import pytest
import requests_mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from library.chembl_cell_lines import (  # noqa: E402
    CellLineClient,
    CellLineConfig,
    CellLineNotFoundError,
    CellLineServiceError,
    fetch_cell_line,
)


def test_fetch_cell_line_returns_payload(requests_mock: requests_mock.Mocker) -> None:
    cfg = CellLineConfig(base_url="https://example.org")
    url = "https://example.org/cell_line/CHEMBL1234.json"
    payload = {"cell_chembl_id": "CHEMBL1234", "cell_name": "Example"}
    requests_mock.get(url, json=payload)
    record = fetch_cell_line("  chembl1234  ", cfg)
    assert record == payload


def test_fetch_cell_line_404(requests_mock: requests_mock.Mocker) -> None:
    cfg = CellLineConfig(base_url="https://example.org")
    url = "https://example.org/cell_line/CHEMBL9999.json"
    requests_mock.get(url, status_code=404)
    with pytest.raises(CellLineNotFoundError):
        fetch_cell_line("CHEMBL9999", cfg)


def test_fetch_cell_line_invalid_payload(requests_mock: requests_mock.Mocker) -> None:
    cfg = CellLineConfig(base_url="https://example.org")
    url = "https://example.org/cell_line/CHEMBL0001.json"
    requests_mock.get(url, json=[{"unexpected": "structure"}])
    client = CellLineClient(cfg)
    with pytest.raises(CellLineServiceError):
        client.fetch_cell_line("CHEMBL0001")


def test_fetch_cell_line_empty_identifier() -> None:
    client = CellLineClient()
    with pytest.raises(ValueError):
        client.fetch_cell_line("   ")


def test_fetch_cell_line_none_identifier() -> None:
    client = CellLineClient()
    with pytest.raises(ValueError):
        client.fetch_cell_line(None)  # type: ignore[arg-type]
