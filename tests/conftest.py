"""Shared test fixtures.

``tests/data/`` holds real upstream rows - header plus three matches, copied byte-for-byte from
Football-Data - one file per era boundary in docs/DECISIONS.md. Hand-written samples would not
reproduce the BOM, the CRLF line endings, the cp1252 bytes or the unnamed trailing columns, which
are the parts most likely to break.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from epl import predictors
from epl.clubs import ClubResolver
from epl.predictors import Evidence
from the_schedule import CronLine, crontab_lines

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
def schedule() -> tuple[CronLine, ...]:
    """What `deploy/crontab` actually schedules, parsed from the committed file."""
    return crontab_lines()


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


#: A match row shaped like ``epl.ingest.load_matches`` output. Tests override only the fields they
#: are about, so a test reads as the thing it is checking rather than as a table of filler.
MATCH_DEFAULTS: dict[str, object] = {
    "season": 2024,
    "division": "E0",
    "date": "2024-08-17",
    "time": pd.NA,
    "home_club": "arsenal",
    "away_club": "chelsea",
    "home_goals": 1,
    "away_goals": 0,
    "outcome": "H",
}


@pytest.fixture
def make_matches() -> Callable[..., pd.DataFrame]:
    """Build a frame of matches, naming only the fields the test is about."""

    def _make(*rows: dict[str, object]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=list(MATCH_DEFAULTS))
        frame = pd.DataFrame([{**MATCH_DEFAULTS, **row} for row in rows])
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        return frame

    return _make


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """An empty Predictor registry, so one test's Predictors never reach another's scoreboard.

    The registry is process-wide by design — importing ``epl.benchmarks`` is what puts the Naive
    Baseline on the scoreboard — so a test that registers anything has to put it back.
    """
    empty: dict[str, object] = {}
    monkeypatch.setattr(predictors, "_REGISTRY", empty)
    return empty


class FixedPredictor:
    """A Predictor that always says the same thing, for testing everything downstream of one.

    It reads its Evidence, because what the ledger records about a Prediction depends on what the
    Predictor actually asked for.
    """

    def __init__(
        self,
        name: str = "fixed",
        probabilities: tuple[float, float, float] = (0.5, 0.3, 0.2),
        divisions: tuple[str, ...] | None = None,
    ) -> None:
        self.name = name
        self.probabilities = probabilities
        self.divisions = divisions

    def predict(self, fixtures: pd.DataFrame, evidence: Evidence) -> npt.NDArray[np.float64]:
        evidence.matches(divisions=self.divisions)
        return np.tile(self.probabilities, (len(fixtures), 1))


@pytest.fixture
def make_predictor() -> Callable[..., FixedPredictor]:
    """A Predictor that always says the same thing. Tests that need a stranger one write it."""

    def _make(
        name: str = "fixed",
        probabilities: tuple[float, float, float] = (0.5, 0.3, 0.2),
        divisions: tuple[str, ...] | None = None,
    ) -> FixedPredictor:
        return FixedPredictor(name, probabilities, divisions)

    return _make
