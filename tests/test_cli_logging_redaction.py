"""Smoke tests ensuring CLI logging redacts obvious secrets."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List
from importlib import util

import pandas as pd
import pytest


SECRET_MESSAGE = "token=SHOULD_NOT_APPEAR"
MASKED_FRAGMENT = "token=***"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _module_logger(module: object) -> logging.Logger:
    """Return the module-specific logger if available."""

    return getattr(module, "LOGGER", logging.getLogger(getattr(module, "__name__", "cli")))


def _load_module(module_path: str) -> object:
    """Load ``module_path`` from the ``scripts`` directory."""

    module_file = PROJECT_ROOT / (module_path.replace(".", "/") + ".py")
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    if "scripts" not in sys.modules:
        package = type(sys)("scripts")
        package.__path__ = [str(PROJECT_ROOT / "scripts")]
        sys.modules["scripts"] = package
    spec = util.spec_from_file_location(module_path, module_file)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Cannot load module {module_path}")
    module = util.module_from_spec(spec)
    sys.modules[module_path] = module
    spec.loader.exec_module(module)
    return module


def _make_stub(
    module: object, *, return_value: object | None = None, exit_code: int | None = None
) -> Callable[..., object]:
    """Create a callable emitting a masked log message before returning."""

    logger = _module_logger(module)

    def _stub(*args: object, **kwargs: object) -> object:
        logger.info(SECRET_MESSAGE)
        if exit_code is not None:
            raise SystemExit(exit_code)
        return return_value

    return _stub


def _ensure_mapping_stub(module: object) -> None:
    """Log the secret marker and terminate execution."""

    _module_logger(module).info(SECRET_MESSAGE)
    raise SystemExit(0)


@dataclass
class CliSpec:
    """Configuration for driving a CLI entry point in tests."""

    module: str
    args: Callable[[Path], List[str]]
    prepare: Callable[[object, pytest.MonkeyPatch, Path], None]
    uses_sys_argv: bool = False
    expect_exit: bool = False
    exit_code: int = 0


def _noop(*args: object, **kwargs: object) -> None:  # pragma: no cover - helper
    return None


CLI_SPECS: list[CliSpec] = [
    CliSpec(
        module="scripts.chembl_assays_main",
        args=lambda _: [],
        prepare=lambda module, monkeypatch, tmp_path: monkeypatch.setattr(
            module, "run_pipeline", _make_stub(module, return_value=0)
        ),
    ),
    CliSpec(
        module="scripts.chembl_activities_main",
        args=lambda _: [],
        prepare=lambda module, monkeypatch, tmp_path: monkeypatch.setattr(
            module, "run_pipeline", _make_stub(module, return_value=0)
        ),
    ),
    CliSpec(
        module="scripts.chembl_testitems_main",
        args=lambda _: [],
        prepare=lambda module, monkeypatch, tmp_path: monkeypatch.setattr(
            module, "run_pipeline", _make_stub(module, return_value=0)
        ),
    ),
    CliSpec(
        module="scripts.data_profiling_main",
        args=lambda _: ["--input", "input.csv"],
        prepare=lambda module, monkeypatch, tmp_path: monkeypatch.setattr(
            module, "analyze_table_quality", _make_stub(module)
        ),
    ),
    CliSpec(
        module="scripts.chembl2uniprot_main",
        args=lambda tmp: [
            "--input",
            str(_write_csv(tmp / "chembl.csv", ["chembl_id"], [["CHEMBL1"]])),
            "--output",
            str(tmp / "out.csv"),
        ],
        prepare=lambda module, monkeypatch, tmp_path: monkeypatch.setattr(
            module,
            "map_chembl_to_uniprot",
            _make_stub(module, return_value=tmp_path / "mapped.csv"),
        ),
    ),
    CliSpec(
        module="scripts.uniprot_enrich_main",
        args=lambda tmp: [
            "--input",
            str(_write_csv(tmp / "input.csv", ["uniprot_id"], [["P12345"]])),
        ],
        uses_sys_argv=True,
        prepare=lambda module, monkeypatch, tmp_path: monkeypatch.setattr(
            module, "enrich_uniprot", _make_stub(module)
        ),
    ),
    CliSpec(
        module="scripts.pubmed_main",
        args=lambda tmp: ["pubmed", "--input", str(tmp / "ids.csv")],
        uses_sys_argv=True,
        prepare=lambda module, monkeypatch, tmp_path: (
            monkeypatch.setattr(
                module,
                "load_config",
                _make_stub(module, return_value=module.DEFAULT_CONFIG),
            ),
            monkeypatch.setattr(module, "run_pubmed_command", _noop),
            monkeypatch.setattr(module, "run_chembl_command", _noop),
            monkeypatch.setattr(module, "run_all_command", _noop),
        ),
    ),
    CliSpec(
        module="scripts.pipeline_targets_main",
        args=lambda tmp: [
            "--input",
            str(tmp / "targets.csv"),
            "--output",
            str(tmp / "out.csv"),
        ],
        uses_sys_argv=True,
        expect_exit=True,
        exit_code=0,
        prepare=lambda module, monkeypatch, tmp_path: monkeypatch.setattr(
            module, "load_pipeline_config", _make_stub(module, exit_code=0)
        ),
    ),
    CliSpec(
        module="scripts.get_target_data_main",
        args=lambda tmp: [
            "--input",
            str(_write_csv(tmp / "targets.csv", ["target_chembl_id"], [["CHEMBL1"]])),
            "--output",
            str(tmp / "out.csv"),
        ],
        prepare=lambda module, monkeypatch, tmp_path: (
            monkeypatch.setattr(
                module,
                "fetch_targets",
                _make_stub(
                    module,
                    return_value=pd.DataFrame({"target_chembl_id": ["CHEMBL1"]}),
                ),
            ),
            monkeypatch.setattr(module, "analyze_table_quality", _noop),
        ),
    ),
    CliSpec(
        module="scripts.dump_gtop_target",
        args=lambda tmp: [
            "--input",
            str(tmp / "targets.csv"),
            "--output-dir",
            str(tmp / "out"),
            "--id-column",
            "uniprot_id",
        ],
        uses_sys_argv=True,
        expect_exit=True,
        exit_code=0,
        prepare=lambda module, monkeypatch, tmp_path: monkeypatch.setattr(
            module, "_load_config", _make_stub(module, exit_code=0)
        ),
    ),
    CliSpec(
        module="scripts.protein_classify_main",
        args=lambda tmp: [
            "--input",
            str(
                _write_csv(
                    tmp / "proteins.csv",
                    ["uniprot_json"],
                    [["{}"], ["{}"]],
                )
            ),
            "--output",
            str(tmp / "out.csv"),
        ],
        prepare=lambda module, monkeypatch, tmp_path: (
            monkeypatch.setattr(
                module,
                "classify_protein",
                _make_stub(module, return_value={}),
            ),
            monkeypatch.setattr(module, "analyze_table_quality", _noop),
        ),
    ),
    CliSpec(
        module="scripts.iuphar_main",
        args=lambda tmp: [
            "--target",
            str(tmp / "target.csv"),
            "--family",
            str(tmp / "family.csv"),
            "--input",
            str(tmp / "input.csv"),
            "--output",
            str(tmp / "out.csv"),
        ],
        uses_sys_argv=True,
        prepare=lambda module, monkeypatch, tmp_path: monkeypatch.setattr(
            module,
            "IUPHARData",
            type(
                "_StubData",
                (),
                {
                    "from_files": staticmethod(
                        lambda *a, **k: type(
                            "_StubMapper",
                            (),
                            {
                                "map_uniprot_file": _make_stub(module),
                            },
                        )()
                    )
                },
            ),
        ),
    ),
    CliSpec(
        module="scripts.get_hgnc_by_uniprot",
        args=lambda tmp: [
            "--input",
            str(_write_csv(tmp / "input.csv", ["uniprot_id"], [["P12345"]])),
            "--output",
            str(tmp / "out.csv"),
        ],
        prepare=lambda module, monkeypatch, tmp_path: monkeypatch.setattr(
            module,
            "map_uniprot_to_hgnc",
            _make_stub(module, return_value=tmp_path / "out.csv"),
        ),
    ),
    CliSpec(
        module="scripts.get_uniprot_target_data",
        args=lambda tmp: ["--input", str(tmp / "input.csv")],
        prepare=lambda module, monkeypatch, tmp_path: monkeypatch.setattr(
            module,
            "_ensure_mapping",
            lambda *a, **k: _ensure_mapping_stub(module),
        ),
        expect_exit=True,
        exit_code=0,
    ),
]


def _write_csv(path: Path, columns: list[str], rows: list[list[str]]) -> Path:
    """Write ``rows`` to ``path`` with ``columns`` as headers."""

    df = pd.DataFrame(rows, columns=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("spec", CLI_SPECS, ids=lambda spec: spec.module.split(".")[-1])
@pytest.mark.parametrize("log_format", ["human", "json"])
def test_cli_logs_mask_secrets(
    spec: CliSpec,
    log_format: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """All CLI entry points must mask secrets in log messages."""

    module = _load_module(spec.module)
    spec.prepare(module, monkeypatch, tmp_path)

    argv = spec.args(tmp_path) + ["--log-format", log_format]
    if spec.uses_sys_argv:
        monkeypatch.setattr(sys, "argv", [spec.module] + argv)
        entry = getattr(module, "main")
        if spec.expect_exit:
            with pytest.raises(SystemExit) as exc_info:
                entry()
            assert exc_info.value.code == spec.exit_code
        else:
            entry()
    else:
        entry = getattr(module, "main")
        if spec.expect_exit:
            with pytest.raises(SystemExit) as exc_info:
                entry(argv)
            assert exc_info.value.code == spec.exit_code
        else:
            entry(argv)

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert SECRET_MESSAGE not in combined
    assert MASKED_FRAGMENT in combined
