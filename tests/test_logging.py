from __future__ import annotations

import io
import json
import logging
import sys

from chembl2uniprot.logging_utils import configure_logging
import pytest


def test_json_logging_with_redaction(monkeypatch: pytest.MonkeyPatch) -> None:
    """configure_logging emits JSON and redacts secrets."""
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    configure_logging("INFO", json_logs=True)
    logger = logging.getLogger("test")
    logger.info("token=SECRET")
    output = stream.getvalue().strip()
    data = json.loads(output)
    assert data["message"] == "token=***"
    assert data["level"] == "INFO"
