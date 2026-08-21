"""Shared test fixtures.

``tests/data/`` holds real upstream rows - header plus three matches, copied byte-for-byte from
Football-Data - one file per era boundary in docs/DECISIONS.md. Hand-written samples would not
reproduce the BOM, the CRLF line endings, the cp1252 bytes or the unnamed trailing columns, which
are the parts most likely to break.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from epl.clubs import ClubResolver

DATA_DIR = Path(__file__).parent / "data"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests that hit football-data.co.uk",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-network"):
        return
    skip = pytest.mark.skip(reason="needs --run-network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def data_dir() -> Path:
    """Directory of committed sample CSVs."""
    return DATA_DIR


@pytest.fixture
def resolver() -> ClubResolver:
    """The Club resolver built from the tables shipped with the package."""
    return ClubResolver.load()


@pytest.fixture
def project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the whole on-disk layout at a temporary directory."""
    monkeypatch.setenv("EPL_PROJECT_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture
def write_csv(tmp_path: Path):
    """Write a CSV with upstream's CRLF line endings and return its path."""

    def _write(name: str, header: str, *rows: str, bom: bool = False) -> Path:
        path = tmp_path / name
        text = "\r\n".join([header, *rows]) + "\r\n"
        path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))
        return path

    return _write
