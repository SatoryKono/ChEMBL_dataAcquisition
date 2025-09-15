import pandas as pd

from scripts.pipeline_targets_main import merge_chembl_fields


def test_merge_chembl_fields_adds_columns():
    pipeline_df = pd.DataFrame(
        {
            "target_chembl_id": ["CHEMBL1"],
            "hgnc_id": ["HGNC:1"],
        }
    )
    chembl_df = pd.DataFrame(
        {
            "target_chembl_id": ["CHEMBL1"],
            "species_group_flag": [1],
            "hgnc_name": ["ABC"],
            "hgnc_id": ["HGNC:1"],
        }
    )
    merged = merge_chembl_fields(pipeline_df, chembl_df)
    assert "species_group_flag" in merged.columns
    assert "hgnc_name" in merged.columns
    assert merged.loc[0, "hgnc_id"] == "HGNC:1"
