"""Tests for dependency synchronisation utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from library import dependency_tools


def test_extract_constraints_filters_lines() -> None:
    """Editable requirements and comments are excluded from the output."""

    raw_lines = [
        "# A comment",
        "annotated-types==0.7.0\n",
        "\n",
        "-e .\n",
        "requests==2.32.5",
    ]
    assert dependency_tools.extract_constraints(raw_lines) == [
        "annotated-types==0.7.0",
        "requests==2.32.5",
    ]


def test_synchronize_constraints_writes_expected(tmp_path: Path) -> None:
    """The constraints file mirrors the lock file contents without editable installs."""

    lock_path = tmp_path / "requirements.lock"
    lock_path.write_text(
        "# comment\n-e .\nattrs==25.3.0\nrequests==2.32.5\n",
        encoding="utf-8",
    )
    constraints_path = tmp_path / "constraints.txt"

    dependency_tools.synchronize_constraints(lock_path, constraints_path)

    assert constraints_path.read_text(encoding="utf-8") == "attrs==25.3.0\nrequests==2.32.5\n"


def test_synchronize_constraints_missing_file(tmp_path: Path) -> None:
    """A missing lock file raises ``FileNotFoundError``."""

    with pytest.raises(FileNotFoundError):
        dependency_tools.synchronize_constraints(
            tmp_path / "missing.lock", tmp_path / "constraints.txt"
        )
