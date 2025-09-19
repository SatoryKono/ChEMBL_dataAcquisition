from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import pandas as pd

from iuphar import IUPHARData, load_families, load_targets


def _iuphar_test_data_path(filename: str) -> Path:
    """Return the path to a fixture file in ``tests/data``."""

    return Path("tests/data") / filename


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


def test_cli_initialises_session_with_contact_user_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CLI initialises the HTTP session with the configured user agent."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "contact:\n"
        "  name: Test Runner\n"
        "  email: runner@example.org\n"
        "  user_agent: test-runner/1.0 (mailto:runner@example.org)\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "output.csv"

    import scripts.iuphar_main as module
    import library.iuphar as iuphar_module

    args = SimpleNamespace(
        target=str(_iuphar_test_data_path("iuphar_target.csv")),
        family=str(_iuphar_test_data_path("iuphar_family.csv")),
        input=str(_iuphar_test_data_path("iuphar_input.csv")),
        output=str(output_path),
        sep=",",
        encoding="utf-8",
        log_level="INFO",
        log_format="human",
        config=str(config_path),
    )
    monkeypatch.setattr(module, "parse_args", lambda: args)

    created_sessions: list[Any] = []

    class DummySession:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def get(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - safety
            raise AssertionError("Network access not expected during CLI test")

    def fake_session() -> DummySession:
        session = DummySession()
        created_sessions.append(session)
        return session

    monkeypatch.setattr(iuphar_module.requests, "Session", fake_session)
    monkeypatch.setattr(iuphar_module, "_session", iuphar_module._session)

    module.main()

    assert created_sessions, "Expected init_session to create a requests.Session"
    assert (
        created_sessions[0].headers.get("User-Agent")
        == "test-runner/1.0 (mailto:runner@example.org)"
    )
    assert output_path.exists()
