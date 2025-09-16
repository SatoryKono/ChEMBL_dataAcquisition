"""Validate the bundled configuration against the JSON schema."""

from __future__ import annotations

from pathlib import Path

from chembl2uniprot.config import load_and_validate_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "config.schema.json"


def test_bundled_config_matches_schema() -> None:
    """Ensure the repository configuration conforms to the schema."""
    config = load_and_validate_config(
        CONFIG_PATH,
        SCHEMA_PATH,
        section="chembl2uniprot",
    )
    assert config.uniprot.rate_limit.rps == 5
    assert config.uniprot.polling.interval_sec == 2
    assert config.uniprot.retry.max_attempts == 5
    assert config.uniprot.retry.backoff_sec == 1
