from __future__ import annotations

from pathlib import Path
import json
import logging
import pytest

from chembl2uniprot.config import load_and_validate_config

DATA_DIR = Path(__file__).parent / "data"
CONFIG_DIR = DATA_DIR / "config"
SCHEMA = CONFIG_DIR / "config.schema.json"
CONFIG = CONFIG_DIR / "valid.yaml"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_CONFIG = PROJECT_ROOT / "config.yaml"
PROJECT_SCHEMA = PROJECT_ROOT / "schemas" / "config.schema.json"


def _write_config(tmp_path: Path, text: str) -> Path:
    cfg = tmp_path / "config.yaml"
    schema = tmp_path / "config.schema.json"
    cfg.write_text(text)
    schema.write_text(SCHEMA.read_text())
    return cfg


def test_contact_section_parsed() -> None:
    """The ``contact`` section is required and parsed via Pydantic."""

    loaded = load_and_validate_config(CONFIG)
    assert loaded.contact.name == "Example Maintainer"
    assert (
        loaded.contact.user_agent == "example-tool/1.0 (mailto:maintainer@example.org)"
    )


def test_invalid_type(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    cfg_text = CONFIG.read_text().replace("rps: 1000", 'rps: "fast"')
    cfg = _write_config(tmp_path, cfg_text)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError):
            load_and_validate_config(cfg)
    assert "uniprot.rate_limit.rps" in caplog.text
    assert "'fast'" in caplog.text
    assert "is not of type 'number'" in caplog.text


def test_unknown_key(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    cfg_text = CONFIG.read_text() + "\nunknown: 42\n"
    cfg = _write_config(tmp_path, cfg_text)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError):
            load_and_validate_config(cfg)
    assert "unknown" in caplog.text
    assert "42" in caplog.text
    assert "Additional properties are not allowed" in caplog.text


def test_invalid_value(tmp_path: Path) -> None:
    cfg_text = CONFIG.read_text().replace("timeout_sec: 30", "timeout_sec: 0")
    cfg = _write_config(tmp_path, cfg_text)
    with pytest.raises(ValueError):
        load_and_validate_config(cfg)


def test_target_chembl_id_alias() -> None:
    """Configuration accepts ``target_chembl_id`` as an alias."""
    cfg = CONFIG_DIR / "alias.yaml"
    loaded = load_and_validate_config(cfg)
    assert loaded.columns.chembl_id == "chembl_id"


def test_chembl_id_with_legacy_schema(tmp_path: Path) -> None:
    """Config using ``chembl_id`` passes against a schema requiring ``target_chembl_id``."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(CONFIG.read_text())
    legacy_schema = json.loads(SCHEMA.read_text())
    columns_schema = legacy_schema["properties"]["columns"]
    columns_schema["properties"].pop("chembl_id")
    columns_schema["required"] = ["target_chembl_id", "uniprot_out"]
    columns_schema.pop("anyOf", None)
    schema_path = tmp_path / "config.schema.json"
    schema_path.write_text(json.dumps(legacy_schema))
    loaded = load_and_validate_config(cfg_path, schema_path)
    assert loaded.columns.chembl_id == "target_chembl_id"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables override YAML configuration values."""
    monkeypatch.setenv("CHEMBL_BATCH__SIZE", "5")
    loaded = load_and_validate_config(CONFIG)
    assert loaded.batch.size == 5
    assert loaded.logging.format == "human"


def test_env_override_invalid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Schema validation rejects invalid values injected via environment."""
    monkeypatch.setenv("CHEMBL_BATCH__SIZE", "0")
    with pytest.raises(ValueError, match="batch.size"):
        load_and_validate_config(CONFIG)


def test_env_override_project_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Project-scoped environment variables target combined configurations."""
    monkeypatch.setenv("CHEMBL_DA__CHEMBL2UNIPROT__BATCH__SIZE", "6")
    loaded = load_and_validate_config(
        PROJECT_CONFIG,
        PROJECT_SCHEMA,
        section="chembl2uniprot",
    )
    assert loaded.batch.size == 6
