from __future__ import annotations

import io
import json
import logging
import sys

from typing import Any, cast

import pytest

from library.logging_utils import configure_logging


def test_json_logging_with_redaction(monkeypatch: pytest.MonkeyPatch) -> None:
    """``configure_logging`` emits JSON output and redacts secrets."""

    stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)

    configure_logging("INFO", log_format="json")

    logger = logging.getLogger("test")
    logger.info(
        "token=SECRET",
        extra={"api_key": "dont_show", "nested": {"password": "value"}},
    )

    output = stream.getvalue().strip()
    data = json.loads(output)

    assert data["message"] == "token=***"
    assert data["level"] == "INFO"
    assert data["extra"]["api_key"] == "***"
    assert data["extra"]["nested"]["password"] == "***"


def test_configure_logging_rejects_unknown_format() -> None:
    """Unsupported log formats raise a :class:`ValueError`."""

    with pytest.raises(ValueError):
        configure_logging("INFO", log_format=cast(Any, "xml"))
