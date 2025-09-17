"""Tests for ``scripts.chembl_testitems_main`` helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from library.io_utils import serialise_cell
from scripts.chembl_testitems_main import _serialise_complex_columns


def test_serialise_complex_columns_serialises_lists_with_json_format() -> None:
    """Ensure list values are serialised via ``serialise_cell`` when using JSON."""

    df = pd.DataFrame({
        "list_column": [[1, 2], ["foo", "bar"]],
    })

    result = _serialise_complex_columns(df, "json")

    assert result.loc[0, "list_column"] == serialise_cell([1, 2], "json")
    assert result.loc[1, "list_column"] == serialise_cell(["foo", "bar"], "json")
