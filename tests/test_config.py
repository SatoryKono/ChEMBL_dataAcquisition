from __future__ import annotations

from pathlib import Path
import pytest

from chembl2uniprot.config import load_and_validate_config

DATA_DIR = Path(__file__).parent / "data" / "config"
SCHEMA = DATA_DIR / "config.schema.json"
CONFIG = DATA_DIR / "valid.yaml"


def _write_config(tmp_path: Path, text: str) -> Path:
    cfg = tmp_path / "config.yaml"
    schema = tmp_path / "config.schema.json"
    cfg.write_text(text)
    schema.write_text(SCHEMA.read_text())
    return cfg


def test_invalid_type(tmp_path: Path) -> None:
    cfg_text = CONFIG.read_text().replace("rps: 1000", 'rps: "fast"')
    cfg = _write_config(tmp_path, cfg_text)
    with pytest.raises(ValueError):
        load_and_validate_config(cfg)


def test_invalid_value(tmp_path: Path) -> None:
    cfg_text = CONFIG.read_text().replace("timeout_sec: 30", "timeout_sec: 0")
    cfg = _write_config(tmp_path, cfg_text)
    with pytest.raises(ValueError):
        load_and_validate_config(cfg)
