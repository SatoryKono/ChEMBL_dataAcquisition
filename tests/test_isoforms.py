import json
import logging
from pathlib import Path
import pytest

from uniprot_normalize import extract_isoforms

ROOT = Path(__file__).resolve().parents[1]


def _load_sample() -> dict:
    return json.loads((ROOT / "tests/data/uniprot_entry.json").read_text())


def test_extract_isoforms_no_isoforms() -> None:
    entry = _load_sample()
    entry["comments"] = [
        c
        for c in entry.get("comments", [])
        if c.get("commentType") != "ALTERNATIVE_PRODUCTS"
    ]
    result = extract_isoforms(entry, [">sp|P12345|"])
    assert result == []


def test_extract_isoforms_mismatch_warn(caplog: pytest.LogCaptureFixture) -> None:
    entry = _load_sample()
    with caplog.at_level(logging.WARNING):
        result = extract_isoforms(entry, [">sp|P12345-1|"])
    assert any("Isoform mismatch" in r.message for r in caplog.records)
    assert len(result) == 2
    assert result[1]["isoform_uniprot_id"] == ""
