"""Unit tests for the shared validation utilities."""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from library.validation_core import ValidationResult


def test_validation_result_unwraps_nested() -> None:
    """Nested :class:`ValidationResult` instances should be unwrapped automatically."""

    valid_frame = pd.DataFrame({"activity_chembl_id": ["CHEMBL1"]})
    errors_frame = pd.DataFrame(
        {
            "index": [0],
            "column": ["activity_chembl_id"],
            "message": ["value must not be empty"],
            "value": ["CHEMBL1"],
            "error_type": ["value_error"],
        }
    )

    inner = ValidationResult(valid=valid_frame, errors=errors_frame)
    outer = ValidationResult(
        valid=cast(pd.DataFrame, inner),
        errors=cast(pd.DataFrame, inner),
    )

    pd.testing.assert_frame_equal(outer.valid, valid_frame)
    pd.testing.assert_frame_equal(outer.errors, errors_frame)


def test_validation_result_empty_property() -> None:
    """The ``empty`` property should mirror the state of the validated frame."""

    result = ValidationResult(valid=pd.DataFrame(), errors=pd.DataFrame())

    assert result.empty is True
    assert bool(result) is False


def test_validation_result_rejects_non_dataframe() -> None:
    """Initialising with non-DataFrame payloads should raise ``TypeError``."""

    with pytest.raises(TypeError, match="ValidationResult\\.valid"):
        ValidationResult(
            valid=cast(pd.DataFrame, [{"activity_chembl_id": "CHEMBL1"}]),
            errors=pd.DataFrame(),
        )
