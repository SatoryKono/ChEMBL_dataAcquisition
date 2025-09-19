"""Tests for dependency synchronisation utilities."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

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

    assert (
        constraints_path.read_text(encoding="utf-8")
        == "attrs==25.3.0\nrequests==2.32.5\n"
    )


def test_synchronize_constraints_missing_file(tmp_path: Path) -> None:
    """A missing lock file raises ``FileNotFoundError``."""

    with pytest.raises(FileNotFoundError):
        dependency_tools.synchronize_constraints(
            tmp_path / "missing.lock", tmp_path / "constraints.txt"
        )


def test_validate_pyproject_constraints_accepts_matching_versions(
    tmp_path: Path,
) -> None:
    """Validation succeeds when the lower bounds mirror the pinned versions."""

    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(
        dedent(
            """
            [project]
            dependencies = [
                "requests>=2.32.5",
            ]

            [project.optional-dependencies]
            dev = [
                "pytest>=8.4.2,<9",
            ]
            """
        ),
        encoding="utf-8",
    )
    constraints_path = tmp_path / "constraints.txt"
    constraints_path.write_text("requests==2.32.5\npytest==8.4.2\n", encoding="utf-8")

    dependency_tools.validate_pyproject_constraints(
        pyproject_path, constraints_path, extras=("dev",)
    )


def test_validate_pyproject_constraints_requires_lower_bounds(tmp_path: Path) -> None:
    """Missing lower bounds are reported as configuration errors."""

    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(
        dedent(
            """
            [project]
            dependencies = [
                "requests",
            ]
            """
        ),
        encoding="utf-8",
    )
    constraints_path = tmp_path / "constraints.txt"
    constraints_path.write_text("requests==2.32.5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must declare an explicit version bound"):
        dependency_tools.validate_pyproject_constraints(
            pyproject_path, constraints_path
        )


def test_validate_pyproject_constraints_detects_mismatched_bounds(
    tmp_path: Path,
) -> None:
    """A different minimum version triggers a validation failure."""

    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(
        dedent(
            """
            [project]
            dependencies = [
                "requests>=2.0",
            ]
            """
        ),
        encoding="utf-8",
    )
    constraints_path = tmp_path / "constraints.txt"
    constraints_path.write_text("requests==2.32.5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Minimum version 2.0"):
        dependency_tools.validate_pyproject_constraints(
            pyproject_path, constraints_path
        )
