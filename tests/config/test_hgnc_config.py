"""Tests for environment overrides in the HGNC configuration loader."""

from __future__ import annotations

from pathlib import Path

from hgnc_client import load_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yaml"


def test_hgnc_config_env_override(monkeypatch) -> None:
    """Environment variables should override network settings."""

    monkeypatch.setenv("CHEMBL_DA__HGNC__NETWORK__TIMEOUT_SEC", "45")

    cfg = load_config(CONFIG_PATH, section="hgnc")

    assert cfg.network.timeout_sec == 45
    assert cfg.rate_limit.rps == 3
