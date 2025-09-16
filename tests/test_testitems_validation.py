from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd

validate_testitems = importlib.import_module(
    "library.testitem_validation"
).validate_testitems


def test_validate_testitems_writes_errors(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "molecule_chembl_id": "CHEMBL1",
                "canonical_smiles": "C",
                "standard_inchi_key": "AAAA",
            },
            {
                "molecule_chembl_id": "CHEMBL2",
                "canonical_smiles": "",
                "standard_inchi_key": None,
            },
        ]
    )

    errors_path = tmp_path / "errors.json"
    validated = validate_testitems(df, errors_path=errors_path)

    assert len(validated) == 1
    assert not validated["molecule_chembl_id"].isna().any()
    assert errors_path.exists()


def test_validate_testitems_handles_numpy_payloads(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "molecule_chembl_id": "CHEMBL1",
                "canonical_smiles": np.array(["C"], dtype=object),
                "standard_inchi_key": np.array(["KEY"], dtype=object),
                "synonyms": np.array([["one", "two"]], dtype=object),
            }
        ]
    )

    errors_path = tmp_path / "errors.json"
    validated = validate_testitems(df, errors_path=errors_path)

    assert not errors_path.exists()
    assert validated.loc[0, "canonical_smiles"] == "C"
    assert validated.loc[0, "standard_inchi_key"] == "KEY"
    assert validated.loc[0, "synonyms"] == ["one", "two"]
