"""Structured logging helpers with redaction support."""

from __future__ import annotations

import json
import logging
import logging.config
import re
from typing import Any

__all__ = ["configure_logging"]


class RedactFilter(logging.Filter):
    """Filter that redacts obvious secret values.

    Any log message containing substrings like ``token=`` or ``key=`` will have
    the value replaced with ``***`` before being emitted.  The implementation is
    intentionally simple and meant only as a safeguard against accidental leaks.
    """

    _pattern = re.compile(r"(?i)(token|key)=([^\s]+)")

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        message = record.getMessage()
        message = self._pattern.sub(lambda m: f"{m.group(1)}=***", message)
        record.msg = message
        record.args = {}
        return True


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        data: dict[str, Any] = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(data, ensure_ascii=False)


def configure_logging(level: str, json_logs: bool = False) -> None:
    """Configure application logging.

    Parameters
    ----------
    level:
        Logging level (e.g. ``"INFO"`` or ``"DEBUG"``).
    json_logs:
        When ``True`` emit JSON-formatted logs instead of human-readable ones.
    """

    formatter = "json" if json_logs else "human"
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {"redact": {"()": RedactFilter}},
            "formatters": {
                "human": {"format": "%(levelname)s %(name)s %(message)s"},
                "json": {"()": JsonFormatter},
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "level": level.upper(),
                    "filters": ["redact"],
                    "formatter": formatter,
                    "stream": "ext://sys.stderr",
                }
            },
            "root": {"level": level.upper(), "handlers": ["default"]},
        }
    )
