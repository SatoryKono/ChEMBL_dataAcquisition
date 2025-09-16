"""Compatibility wrapper around :mod:`library.logging_utils`.

The project's historical entry points import :func:`configure_logging` from
``library.chembl2uniprot.logging_utils``.  This module now simply re-exports the
centralised utilities located in :mod:`library.logging_utils`.
"""

from __future__ import annotations

from library.logging_utils import (
    JsonFormatter,
    SecretRedactingFilter,
    configure_logging,
)

__all__ = ["JsonFormatter", "SecretRedactingFilter", "configure_logging"]
