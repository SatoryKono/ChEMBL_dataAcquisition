"""Unit tests for protein classification."""

from __future__ import annotations

import json
from pathlib import Path

from protein_classifier import classify_protein

SAMPLES = json.loads(
    (Path(__file__).parent / "data" / "protein_samples.json").read_text()
)


def test_gpc_receptor() -> None:
    result = classify_protein(SAMPLES["gpcr"])
    assert result["protein_class_L1"] == "Receptor: GPCR"
    assert result["confidence"] == "high"


def test_ion_channel_voltage_gated() -> None:
    result = classify_protein(SAMPLES["ion_channel"])
    assert result["protein_class_L1"] == "Ion channel"
    assert result["protein_class_L2"] == "Voltage-gated"


def test_transporter_abc() -> None:
    result = classify_protein(SAMPLES["transporter"])
    assert result["protein_class_L1"] == "Transporter"
    assert result["protein_class_L2"] == "ABC"


def test_nuclear_receptor() -> None:
    result = classify_protein(SAMPLES["nuclear_receptor"])
    assert result["protein_class_L1"] == "Receptor: Nuclear"


def test_kinase() -> None:
    result = classify_protein(SAMPLES["kinase"])
    assert result["protein_class_L1"] == "Enzyme"
    assert result["protein_class_L2"] == "Kinase"


def test_protease() -> None:
    result = classify_protein(SAMPLES["protease"])
    assert result["protein_class_L2"] == "Protease"


def test_catalytic_generic() -> None:
    result = classify_protein(SAMPLES["catalytic"])
    assert result["protein_class_L1"] == "Enzyme"
    assert result["protein_class_L2"] == "Enzyme: Other"
    assert result["confidence"] == "medium"


def test_receptor_hint_only() -> None:
    result = classify_protein(SAMPLES["receptor_text"])
    assert result["protein_class_L1"] == "Receptor"


def test_transmembrane_only() -> None:
    result = classify_protein(SAMPLES["transmembrane"])
    assert result["protein_class_L1"] == "Other/Unknown"
    assert result["confidence"] == "low"
