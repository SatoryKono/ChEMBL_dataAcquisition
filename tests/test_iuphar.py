from __future__ import annotations

from pathlib import Path

import pandas as pd

from iuphar import IUPHARData, load_families, load_targets


def test_load_functions() -> None:
    target_df = load_targets(Path("tests/data/iuphar_target.csv"))
    family_df = load_families(Path("tests/data/iuphar_family.csv"))
    assert "target_id" in target_df.columns
    assert "family_id" in family_df.columns


def test_map_uniprot_file(tmp_path) -> None:
    data = IUPHARData.from_files(
        Path("tests/data/iuphar_target.csv"), Path("tests/data/iuphar_family.csv")
    )
    output = tmp_path / "mapped.csv"
    df = data.map_uniprot_file(Path("tests/data/iuphar_input.csv"), output)
    assert output.exists()
    assert isinstance(df, pd.DataFrame)
    assert df.loc[0, "target_id"] == "0001"
    assert df.loc[0, "IUPHAR_class"] == "Enzyme"
