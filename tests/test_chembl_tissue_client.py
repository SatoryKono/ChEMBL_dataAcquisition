from __future__ import annotations

import pytest

from chembl_tissue_client import (
    TissueConfig,
    TissueNotFoundError,
    fetch_tissue_record,
    fetch_tissues,
    normalise_tissue_id,
)


def test_normalise_tissue_id_strips_and_upper_cases() -> None:
    assert normalise_tissue_id("  chembl123 ") == "CHEMBL123"


@pytest.mark.parametrize("value", [None, "   "])
def test_normalise_tissue_id_rejects_empty(value: str | None) -> None:
    with pytest.raises(ValueError):
        normalise_tissue_id(value)  # type: ignore[arg-type]


def test_normalise_tissue_id_rejects_incorrect_pattern() -> None:
    with pytest.raises(ValueError):
        normalise_tissue_id("CHEMBLXYZ")


def test_fetch_tissue_record_success(requests_mock) -> None:
    config = TissueConfig(base_url="https://example.org")
    requests_mock.get(
        "https://example.org/tissue/CHEMBL42.json",
        json={"tissue_chembl_id": "CHEMBL42", "pref_name": "Example"},
    )
    payload = fetch_tissue_record("chembl42", config=config)
    assert payload["pref_name"] == "Example"


def test_fetch_tissue_record_not_found(requests_mock) -> None:
    config = TissueConfig(base_url="https://example.org")
    requests_mock.get("https://example.org/tissue/CHEMBL42.json", status_code=404)
    with pytest.raises(TissueNotFoundError):
        fetch_tissue_record("CHEMBL42", config=config)


def test_fetch_tissue_record_requires_fields(requests_mock) -> None:
    config = TissueConfig(base_url="https://example.org")
    requests_mock.get(
        "https://example.org/tissue/CHEMBL42.json",
        json={"pref_name": "Example"},
    )
    with pytest.raises(ValueError):
        fetch_tissue_record("CHEMBL42", config=config)


def test_fetch_tissues_continue_on_error(requests_mock) -> None:
    config = TissueConfig(base_url="https://example.org")
    requests_mock.get(
        "https://example.org/tissue/CHEMBL1.json",
        json={"tissue_chembl_id": "CHEMBL1", "pref_name": "One"},
    )
    requests_mock.get("https://example.org/tissue/CHEMBL2.json", status_code=404)
    results = fetch_tissues(
        ["CHEMBL1", "CHEMBL2", "CHEMBL1"],
        config=config,
        continue_on_error=True,
    )
    assert [record["tissue_chembl_id"] for record in results] == ["CHEMBL1"]
