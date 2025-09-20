"""Property-based tests comparing vectorised validators to legacy Pydantic models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from hypothesis import HealthCheck, given, settings, strategies as st

from library import legacy_validation
from library.activity_validation import ActivitiesSchema, validate_activities
from library.assay_validation import AssaysSchema, validate_assays
from library.testitem_validation import TestitemsSchema, validate_testitems
from library.validation_core import coerce_value


def _row_strategy(
    columns: Iterable[str], value_strategies: dict[str, st.SearchStrategy[Any]]
) -> st.SearchStrategy[dict[str, Any]]:
    column_list = list(columns)
    flag_strategy = st.lists(
        st.booleans(), min_size=len(column_list), max_size=len(column_list)
    )
    value_tuple = st.tuples(*(value_strategies[name] for name in column_list))

    def _build_row(flags: list[bool], values: tuple[Any, ...]) -> dict[str, Any]:
        return {
            name: values[index]
            for index, (name, include) in enumerate(
                zip(column_list, flags, strict=False)
            )
            if include
        }

    return st.builds(_build_row, flag_strategy, value_tuple)


def _dataframe_strategy(
    columns: Iterable[str], value_strategies: dict[str, st.SearchStrategy[Any]]
) -> st.SearchStrategy[pd.DataFrame]:
    row_strategy = _row_strategy(columns, value_strategies)
    return st.lists(row_strategy, min_size=0, max_size=5).map(pd.DataFrame)


def _ordered_view(df: pd.DataFrame, ordered_columns: list[str]) -> pd.DataFrame:
    known = [column for column in ordered_columns if column in df.columns]
    extras = sorted(column for column in df.columns if column not in known)
    if not known and not extras:
        return df.copy().reset_index(drop=True)
    return df.reindex(columns=ordered_columns + extras).reset_index(drop=True)


def _legacy_error_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(entry["index"]) for entry in payload}


def _new_error_indices(errors: pd.DataFrame) -> set[int]:
    if errors.empty:
        return set()
    return {int(index) for index in errors["index"].tolist()}


def _canonicalise_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df.apply(lambda column: column.map(coerce_value))


_testitems_columns = [
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_inchi_key",
    "max_phase",
    "chembl_full_mwt",
    "pubchem_cid",
]
_testitems_value_strategies = {
    "molecule_chembl_id": st.one_of(
        st.none(), st.text(min_size=0, max_size=5), st.integers(-5, 5)
    ),
    "canonical_smiles": st.one_of(st.none(), st.text(min_size=0, max_size=5)),
    "standard_inchi_key": st.one_of(st.none(), st.text(min_size=0, max_size=5)),
    "max_phase": st.one_of(
        st.none(),
        st.integers(-3, 5),
        st.floats(min_value=-3, max_value=5, allow_nan=False, allow_infinity=False),
        st.just(float("nan")),
        st.text(min_size=0, max_size=5),
    ),
    "chembl_full_mwt": st.one_of(
        st.none(),
        st.floats(
            min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        st.just(float("nan")),
        st.text(min_size=0, max_size=5),
    ),
    "pubchem_cid": st.one_of(
        st.none(),
        st.integers(-10, 10),
        st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        st.just(float("nan")),
        st.text(min_size=0, max_size=5),
    ),
}

_assay_columns = [
    "assay_chembl_id",
    "document_chembl_id",
    "assay_with_same_target",
    "confidence_score",
]
_assay_value_strategies = {
    "assay_chembl_id": st.one_of(st.none(), st.text(min_size=0, max_size=5)),
    "document_chembl_id": st.one_of(st.none(), st.text(min_size=0, max_size=5)),
    "assay_with_same_target": st.one_of(
        st.none(),
        st.integers(-2, 5),
        st.floats(min_value=-2, max_value=5, allow_nan=False, allow_infinity=False),
        st.just(float("nan")),
        st.text(min_size=0, max_size=5),
    ),
    "confidence_score": st.one_of(
        st.none(),
        st.integers(-2, 5),
        st.floats(min_value=-2, max_value=5, allow_nan=False, allow_infinity=False),
        st.just(float("nan")),
        st.text(min_size=0, max_size=5),
    ),
}

_activity_columns = [
    "activity_chembl_id",
    "assay_chembl_id",
    "record_id",
    "activity_id",
    "standard_value",
]
_activity_value_strategies = {
    "activity_chembl_id": st.one_of(st.none(), st.text(min_size=0, max_size=5)),
    "assay_chembl_id": st.one_of(st.none(), st.text(min_size=0, max_size=5)),
    "record_id": st.one_of(
        st.none(),
        st.integers(-5, 5),
        st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False),
        st.just(float("nan")),
        st.text(min_size=0, max_size=5),
    ),
    "activity_id": st.one_of(
        st.none(),
        st.integers(-5, 5),
        st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False),
        st.just(float("nan")),
        st.text(min_size=0, max_size=5),
    ),
    "standard_value": st.one_of(
        st.none(),
        st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        st.just(float("nan")),
        st.text(min_size=0, max_size=5),
    ),
}


@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(_dataframe_strategy(_testitems_columns, _testitems_value_strategies))
def test_validate_testitems_matches_legacy(tmp_path: Path, df: pd.DataFrame) -> None:
    new_errors = tmp_path / "new_testitems.json"
    legacy_errors = tmp_path / "legacy_testitems.json"
    new_errors.unlink(missing_ok=True)
    legacy_errors.unlink(missing_ok=True)

    result = validate_testitems(df.copy(deep=True), errors_path=new_errors)
    legacy = legacy_validation.legacy_validate_testitems(
        df.copy(deep=True), errors_path=legacy_errors
    )

    new_valid = _canonicalise_frame(
        _ordered_view(result.valid, TestitemsSchema.ordered_columns())
    )
    legacy_valid = _canonicalise_frame(
        _ordered_view(legacy, TestitemsSchema.ordered_columns())
    )
    pd.testing.assert_frame_equal(new_valid, legacy_valid, check_dtype=False)

    assert new_errors.exists() == legacy_errors.exists()
    assert _new_error_indices(result.errors) == _legacy_error_indices(legacy_errors)


@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(_dataframe_strategy(_assay_columns, _assay_value_strategies))
def test_validate_assays_matches_legacy(tmp_path: Path, df: pd.DataFrame) -> None:
    new_errors = tmp_path / "new_assays.json"
    legacy_errors = tmp_path / "legacy_assays.json"
    new_errors.unlink(missing_ok=True)
    legacy_errors.unlink(missing_ok=True)

    result = validate_assays(df.copy(deep=True), errors_path=new_errors)
    legacy = legacy_validation.legacy_validate_assays(
        df.copy(deep=True), errors_path=legacy_errors
    )

    new_valid = _canonicalise_frame(
        _ordered_view(result.valid, AssaysSchema.ordered_columns())
    )
    legacy_valid = _canonicalise_frame(
        _ordered_view(legacy, AssaysSchema.ordered_columns())
    )
    pd.testing.assert_frame_equal(new_valid, legacy_valid, check_dtype=False)

    assert new_errors.exists() == legacy_errors.exists()
    assert _new_error_indices(result.errors) == _legacy_error_indices(legacy_errors)


@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(_dataframe_strategy(_activity_columns, _activity_value_strategies))
def test_validate_activities_matches_legacy(tmp_path: Path, df: pd.DataFrame) -> None:
    new_errors = tmp_path / "new_activities.json"
    legacy_errors = tmp_path / "legacy_activities.json"
    new_errors.unlink(missing_ok=True)
    legacy_errors.unlink(missing_ok=True)

    result = validate_activities(df.copy(deep=True), errors_path=new_errors)
    legacy = legacy_validation.legacy_validate_activities(
        df.copy(deep=True), errors_path=legacy_errors
    )

    new_valid = _canonicalise_frame(
        _ordered_view(result.valid, ActivitiesSchema.ordered_columns())
    )
    legacy_valid = _canonicalise_frame(
        _ordered_view(legacy, ActivitiesSchema.ordered_columns())
    )
    pd.testing.assert_frame_equal(new_valid, legacy_valid, check_dtype=False)

    assert new_errors.exists() == legacy_errors.exists()
    assert _new_error_indices(result.errors) == _legacy_error_indices(legacy_errors)
