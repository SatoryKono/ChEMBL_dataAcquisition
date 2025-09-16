from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chembl2uniprot.mapping import get_ids_from_dataframe  # noqa: E402


def test_get_ids_from_dataframe_filters_nan_and_empty_strings() -> None:
    df = pd.DataFrame(
        {
            "chembl_id": [
                "CHEMBL1",
                pd.NA,
                float("nan"),
                "nan",
                " NaN ",
                "",
                "CHEMBL2",
                "CHEMBL1",
            ]
        }
    )

    assert get_ids_from_dataframe(df, "chembl_id") == ["CHEMBL1", "CHEMBL2"]
