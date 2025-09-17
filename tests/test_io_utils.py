"""Tests for :mod:`library.io_utils`."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

from hypothesis import given, settings, strategies as st

from library.io_utils import CsvConfig, _serialise_list, serialise_cell, write_rows


# ---------------------------------------------------------------------------
# Hypothesis-based tests for _serialise_list
# ---------------------------------------------------------------------------


def _parse_pipe(value: str) -> Iterable[str]:
    r"""Parse a pipe-delimited string produced by ``_serialise_list``.

    This function splits on unescaped pipes and unescapes ``\|`` sequences.
    """

    result = []
    current = []
    escape = False
    for ch in value:
        if escape:
            if ch not in ("|", "\\"):
                current.append("\\")
            current.append(ch)
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == "|":
            result.append("".join(current))
            current = []
            continue
        current.append(ch)
    if escape:
        current.append("\\")
    result.append("".join(current))
    return result


pipe_strategy = st.lists(
    st.one_of(
        st.text(),
        st.integers(),
        st.tuples(st.text(), st.text()),
    )
)


@settings(max_examples=25)
@given(pipe_strategy)
def test_serialise_list_pipe_roundtrip(values: list[Any]) -> None:
    """Serialisation to ``pipe`` is stable under a parse/serialise round-trip."""

    serialised = _serialise_list(values, "pipe")
    parsed = list(_parse_pipe(serialised))
    assert _serialise_list(parsed, "pipe") == serialised


@settings(max_examples=25)
@given(pipe_strategy)
def test_serialise_list_json_roundtrip(values: list[Any]) -> None:
    """Serialisation to ``json`` round-trips via ``json.loads``."""

    serialised = _serialise_list(values, "json")
    parsed = json.loads(serialised)
    expected = [
        {"id": v[0], "name": v[1]} if isinstance(v, tuple) else v for v in values
    ]
    assert parsed == expected


def test_serialise_cell_pipe_escapes_strings() -> None:
    """Ensure ``serialise_cell`` escapes pipe characters in scalar strings."""

    assert serialise_cell("alpha|beta", "pipe") == "alpha\\|beta"


def test_serialise_cell_pipe_handles_lists() -> None:
    """Ensure ``serialise_cell`` escapes list items containing pipes."""

    value = ["foo|bar", "baz"]
    assert serialise_cell(value, "pipe") == "foo\\|bar|baz"


def test_serialise_cell_pipe_handles_dicts() -> None:
    """Ensure dictionaries are serialised deterministically with escaping."""

    result = serialise_cell({"name": "alpha|beta", "id": 1}, "pipe")
    assert result == '{"id": 1, "name": "alpha\\|beta"}'


# ---------------------------------------------------------------------------
# Golden file test for write_rows
# ---------------------------------------------------------------------------


def test_write_rows_golden(tmp_path: Path) -> None:
    """Ensure :func:`write_rows` produces deterministic output."""

    cfg = CsvConfig(sep=",", encoding="utf-8", list_format="json")
    rows = [
        {"id": 1, "names": ["alpha", "beta"], "meta": [("x", "y")]},
        {
            "id": 2,
            "names": ["gamma", "delta"],
            "meta": [("u", "v"), ("w", "z")],
        },
    ]
    columns = ["id", "names", "meta"]

    out = tmp_path / "rows.csv"
    write_rows(out, rows, columns, cfg)

    golden = Path(__file__).parent / "data" / "golden_write_rows.csv"
    assert out.read_bytes() == golden.read_bytes()

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    assert digest == "f3acc2865cc6a94ad18b62dfac787ba4cb991e95b815a1cb5558bdacd071e285"


def test_write_rows_pipe_golden(tmp_path: Path) -> None:
    """Golden test covering pipe serialisation with escaping."""

    cfg = CsvConfig(sep=",", encoding="utf-8", list_format="pipe")
    rows = [
        {
            "id": 1,
            "names": ["alpha|beta", "gamma"],
            "meta": [{"id": "x|y", "name": "value|with|pipes"}],
        },
        {
            "id": 2,
            "names": ["delta", "epsilon|zeta"],
            "meta": [],
        },
    ]
    columns = ["id", "names", "meta"]

    out = tmp_path / "rows_pipe.csv"
    write_rows(out, rows, columns, cfg)

    golden = Path(__file__).parent / "data" / "golden_write_rows_pipe.csv"
    assert out.read_bytes() == golden.read_bytes()

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    assert digest == "bf34f72b1240346d4ed9c1e79c4e1a3cbb13666bd1e20e0df5ba2f310dc8a7e3"


# ---------------------------------------------------------------------------
# Integration of determinism script
# ---------------------------------------------------------------------------


def test_check_determinism_script(tmp_path: Path) -> None:
    """Run the determinism script twice and inspect metadata output."""

    output_csv = tmp_path / "determinism.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/check_determinism.py",
            "--output",
            str(output_csv),
            "--keep-artifacts",
        ],
        check=True,
    )

    meta_path = output_csv.with_suffix(".csv.meta.yaml")
    metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    determinism = metadata["determinism"]
    assert determinism["matches_previous"] is True
    assert determinism["previous_sha256"] == determinism["current_sha256"]
    assert determinism["check_count"] >= 2
