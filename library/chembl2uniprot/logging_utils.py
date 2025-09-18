"""Compatibility wrapper around :mod:`library.logging_utils`.

The project's historical entry points import :func:`configure_logging` from
``library.chembl2uniprot.logging_utils``.  This module now simply re-exports the
centralised utilities located in :mod:`library.logging_utils`.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:  # pragma: no cover - imported for static type checking only
    from library.logging_utils import (
        JsonFormatter as _JsonFormatter,
        SecretRedactingFilter as _SecretRedactingFilter,
        configure_logging as _ConfigureLogging,
    )

__all__ = ["JsonFormatter", "SecretRedactingFilter", "configure_logging"]

_LOGGING_MODULE_CANDIDATES = ("library.logging_utils", "logging_utils")


def _import_logging_module() -> Any:
    """Return the first available logging module implementation."""

    for module_name in _LOGGING_MODULE_CANDIDATES:
        try:
            return import_module(module_name)
        except ModuleNotFoundError:
            continue
    msg = (
        "Unable to import either 'library.logging_utils' or 'logging_utils'. "
        "Ensure the project package is available on PYTHONPATH."
    )
    raise ModuleNotFoundError(msg)


_module = _import_logging_module()

JsonFormatter = cast("type[_JsonFormatter]", getattr(_module, "JsonFormatter"))
SecretRedactingFilter = cast(
    "type[_SecretRedactingFilter]", getattr(_module, "SecretRedactingFilter")
)
configure_logging = cast("_ConfigureLogging", getattr(_module, "configure_logging"))
