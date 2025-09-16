"""Tests for the CLI path helper utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _path_utils  # noqa: E402


def test_ensure_project_root_inserts_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ensure_project_root`` should add the root and library directories."""

    monkeypatch.setattr(sys, "path", ["/tmp"])
    project_root = _path_utils.ensure_project_root()
    library_path = project_root / "library"

    assert sys.path[0] == str(library_path)
    assert sys.path[1] == str(project_root)
    assert library_path.exists()


def test_ensure_project_root_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running the helper twice must not duplicate ``sys.path`` entries."""

    monkeypatch.setattr(sys, "path", ["/tmp"])
    first_root = _path_utils.ensure_project_root()
    second_root = _path_utils.ensure_project_root()

    assert first_root == second_root
    occurrences_root = [entry for entry in sys.path if entry == str(first_root)]
    occurrences_library = [
        entry for entry in sys.path if entry == str(first_root / "library")
    ]
    assert len(occurrences_root) == 1
    assert len(occurrences_library) == 1
