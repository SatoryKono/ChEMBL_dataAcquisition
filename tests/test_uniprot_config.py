from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

from library.config.uniprot import ConfigError, load_uniprot_target_config


def _write_config(tmp_path: Path, text: str) -> Path:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(textwrap.dedent(text))
    return cfg_path


def test_load_valid_configuration(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path,
        """
        output:
          sep: ";"
          encoding: latin-1
          list_format: json
          include_sequence: true
        uniprot:
          base_url: https://example.org/uniprot
          include_isoforms: true
          use_fasta_stream_for_isoform_ids: false
          timeout_sec: 45
          retries: 5
          rps: 4
          columns:
            - accession
            - gene
        orthologs:
          enabled: false
          target_species: [human, mouse]
        http_cache:
          enabled: true
          path: .cache/http
          ttl_sec: 3600
        """,
    )
    cfg = load_uniprot_target_config(cfg_path)
    assert cfg.output.sep == ";"
    assert cfg.output.encoding == "latin-1"
    assert cfg.output.list_format == "json"
    assert cfg.output.include_sequence is True
    assert cfg.uniprot.base_url == "https://example.org/uniprot"
    assert cfg.uniprot.include_isoforms is True
    assert cfg.uniprot.use_fasta_stream_for_isoform_ids is False
    assert cfg.uniprot.timeout_sec == 45
    assert cfg.uniprot.retries == 5
    assert cfg.uniprot.rps == 4
    assert cfg.uniprot.columns == ["accession", "gene"]
    assert cfg.orthologs.enabled is False
    assert cfg.orthologs.target_species == ["human", "mouse"]
    assert cfg.http_cache is not None
    assert cfg.http_cache.enabled is True
    assert cfg.http_cache.path == ".cache/http"
    assert cfg.http_cache.ttl_sec == 3600


def test_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = _write_config(
        tmp_path,
        """
        output: {}
        """,
    )
    monkeypatch.setenv("CHEMBL_DA__OUTPUT__SEP", "\t")
    cfg = load_uniprot_target_config(cfg_path)
    assert cfg.output.sep == "\t"


def test_invalid_list_format(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path,
        """
        output:
          list_format: invalid
        """,
    )
    with pytest.raises(ConfigError, match="output.list_format"):
        load_uniprot_target_config(cfg_path)
