"""Project-wide logging helpers with JSON output and secret redaction.

The module provides a configurable logging setup that can emit either
human-readable or JSON-formatted records.  A :class:`SecretRedactingFilter`
scans messages and extra context for common secret patterns and replaces the
corresponding values with ``"***"`` to reduce the risk of leaking credentials.
"""

from __future__ import annotations

import json
import logging
import logging.config
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Literal

__all__ = [
    "JsonFormatter",
    "SecretRedactingFilter",
    "configure_logging",
]

LogFormat = Literal["human", "json"]

_SECRET_KEY_PATTERN = (
    r"(?:token|secret|password|api[_-]?key|access[_-]?token|auth[_-]?token)"
)
_SECRET_REGEX = re.compile(
    rf"(?i)(?P<prefix>\b{_SECRET_KEY_PATTERN}\b(?:\"|')?\s*[:=]\s*(?:\"|')?)(?P<secret>[^\"',;\s]+)"
)
_SECRET_NAME_REGEX = re.compile(rf"(?i)^{_SECRET_KEY_PATTERN}$")

_LOG_RECORD_SKIP_SANITISE = {"msg", "args", "exc_info", "stack_info"}
_LOG_RECORD_RESERVED_KEYS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "message",
    "module",
    "msecs",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


class SecretRedactingFilter(logging.Filter):
    """Filter log records to mask common secret tokens.

    The filter replaces obvious secrets (``token=``, ``password=``, etc.) with
    the placeholder ``"***"``.  Redaction is applied to the formatted log
    message as well as to values supplied through ``extra`` keyword arguments.
    The implementation intentionally favours simplicity over completeness and is
    meant to prevent accidental leaks in command line tools.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        record.msg = self._sanitize(record.getMessage())
        # ``logging.Filter`` runs before the formatter.  By emptying ``args`` we
        # prevent the logging framework from attempting to perform %-formatting
        # again after the message has been sanitised.
        record.args = ()

        for attr, value in list(record.__dict__.items()):
            if attr in _LOG_RECORD_SKIP_SANITISE:
                continue
            new_value = self._sanitize(value)
            if _SECRET_NAME_REGEX.search(attr):
                new_value = "***"
            record.__dict__[attr] = new_value
        return True

    @classmethod
    def _sanitize(cls, value: Any) -> Any:
        """Recursively redact secrets from ``value``."""

        if isinstance(value, str):
            return _SECRET_REGEX.sub(lambda match: f"{match.group('prefix')}***", value)
        if isinstance(value, Mapping):
            sanitized: dict[Any, Any] = {}
            for key, val in value.items():
                new_val = cls._sanitize(val)
                if isinstance(key, str) and _SECRET_NAME_REGEX.search(key):
                    new_val = "***"
                sanitized[key] = new_val
            return sanitized
        if isinstance(value, list):
            return [cls._sanitize(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._sanitize(item) for item in value)
        if isinstance(value, set):
            return {cls._sanitize(item) for item in value}
        return value


class JsonFormatter(logging.Formatter):
    """Format :class:`logging.LogRecord` instances as JSON strings.

    The formatter emits ISO-8601 timestamps, log levels, logger names and the
    message.  Extra context attached to the log record via ``extra`` is emitted
    under the ``"extra"`` key once converted into JSON-compatible data
    structures.
    """

    def __init__(self, *, ensure_ascii: bool = False) -> None:
        super().__init__()
        self.ensure_ascii = ensure_ascii

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc)
        payload: dict[str, Any] = {
            "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _LOG_RECORD_RESERVED_KEYS and not key.startswith("_")
        }
        if extras:
            payload["extra"] = self._json_compatible(extras)

        return json.dumps(payload, ensure_ascii=self.ensure_ascii)

    def _json_compatible(self, value: Any) -> Any:
        """Convert ``value`` into JSON-serialisable data."""

        if isinstance(value, Mapping):
            return {str(key): self._json_compatible(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._json_compatible(item) for item in value]
        if isinstance(value, tuple):
            return [self._json_compatible(item) for item in value]
        if isinstance(value, set):
            return [self._json_compatible(item) for item in value]
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return [self._json_compatible(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)


def _resolve_level(log_level: str) -> int:
    """Resolve ``log_level`` to a numeric logging level."""

    level = logging.getLevelName(log_level.upper())
    if isinstance(level, str):
        msg = f"Unknown log level: {log_level!r}"
        raise ValueError(msg)
    return int(level)


def configure_logging(
    log_level: str = "INFO", *, log_format: LogFormat = "human"
) -> None:
    """Configure application logging for CLI entry points.

    Args:
        log_level: Verbosity level name, for example ``"INFO"`` or ``"DEBUG"``.
        log_format: Output format for log records.  ``"human"`` emits formatted
            text while ``"json"`` writes structured JSON lines.

    Raises:
        ValueError: If ``log_format`` or ``log_level`` is not recognised.
    """

    if log_format not in ("human", "json"):
        msg = f"Unsupported log format: {log_format!r}"
        raise ValueError(msg)

    level_value = _resolve_level(log_level)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {"redact": {"()": SecretRedactingFilter}},
            "formatters": {
                "human": {
                    "format": "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                    "datefmt": "%Y-%m-%dT%H:%M:%S%z",
                },
                "json": {"()": JsonFormatter},
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "level": level_value,
                    "filters": ["redact"],
                    "formatter": log_format,
                    "stream": "ext://sys.stderr",
                }
            },
            "root": {"level": level_value, "handlers": ["default"]},
        }
    )
