"""The live loop's command line — the whole weekly path, end to end.

`round`, then `seal`, then `score` once the results are in. These run against a small corpus in a
temporary project root, with the real Predictor registry, so what is exercised is the wiring rather
than any model's arithmetic.

The clock is stopped through :func:`epl.live.__main__.clock`, which exists for exactly that and is
deliberately not a command-line option — see its docstring.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from epl.ingest.fixtures import fixtures_dir
from epl.ledger import live as store
from epl.ledger import schema
from epl.live import __main__ as cli
from epl.paths import processed_dir
from epl.windows import LIVE_SEASON

#: Inside the window of the round anchored to Friday 2026-08-28, whose Fixtures are the Saturday.
INSIDE_THE_WINDOW = pd.Timestamp("2026-08-28 14:00:00")

#: The rolling file, in Football-Data's own shape: upstream spellings, `DD/MM/YYYY`, `Avg*` odds.
ROLLING_CSV = "\r\n".join(
    [
        "Div,Date,Time,HomeTeam,AwayTeam,AvgH,AvgD,AvgA",
        "E0,29/08/2026,15:00,Arsenal,Everton,1.55,4.20,5.50",
        "E0,29/08/2026,17:30,Man City,Chelsea,1.85,3.70,4.10",
        "E2,29/08/2026,15:00,Barnsley,Blackpool,2.10,3.30,3.60",
        "",
    ]
)


@pytest.fixture
def stopped_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "clock", lambda: INSIDE_THE_WINDOW)


@pytest.fixture
def corpus(project_root: Path, make_matches: Callable[..., pd.DataFrame]) -> Path:
    """A match table with the live Season under way and nowhere near finished."""
    matches = make_matches(
        {"season": LIVE_SEASON, "date": "2026-08-21", "home_club": "arsenal",
         "away_club": "coventry", "outcome": "H"},
        {"season": LIVE_SEASON, "date": "2026-08-22", "home_club": "everton",
         "away_club": "chelsea", "outcome": "A"},
        {"season": LIVE_SEASON, "date": "2026-08-24", "home_club": "man_city",
         "away_club": "coventry", "outcome": "H"},
    )
    processed_dir().mkdir(parents=True, exist_ok=True)
    path = processed_dir() / "matches.csv"
    matches.to_csv(path, index=False)
    return path


@pytest.fixture
def rolling(project_root: Path) -> Path:
    """A cached rolling fixtures file, as `python -m epl.ingest fixtures` would have left one."""
    fixtures_dir().mkdir(parents=True, exist_ok=True)
    path = fixtures_dir() / "fixtures_20260828T120000Z.csv"
    path.write_bytes(ROLLING_CSV.encode("utf-8"))
    return path


@pytest.fixture
def repo(project_root: Path) -> Path:
    """A git repository at the project root, so a sealed round can be committed."""
    for command in (
        ("init", "-q"),
        ("config", "user.email", "test@example.com"),
        ("config", "user.name", "Test"),
    ):
        subprocess.run(["git", "-C", str(project_root), *command], check=True)
    return project_root


def _played(path: Path, make_matches: Callable[..., pd.DataFrame]) -> None:
    """What `score` would find after the results were ingested: the round, now with Outcomes."""
    results = make_matches(
        {"season": LIVE_SEASON, "date": "2026-08-29", "time": "15:00",
         "home_club": "arsenal", "away_club": "everton", "outcome": "H"},
        {"season": LIVE_SEASON, "date": "2026-08-29", "time": "17:30",
         "home_club": "man_city", "away_club": "chelsea", "outcome": "D"},
    )
    pd.concat([pd.read_csv(path), results], ignore_index=True).to_csv(path, index=False)


class TestUpcoming:
    def test_it_reports_the_round_and_writes_nothing(
        self, corpus: Path, rolling: Path, stopped_clock: None,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert cli.main(["upcoming", "--cached"]) == 0

        printed = capsys.readouterr().out
        assert "2026-08-28" in printed
        assert "sealable now: 2026-08-28" in printed
        assert store.sealed_rounds() == []

    def test_other_tiers_are_not_reported_as_upcoming(
        self, corpus: Path, rolling: Path, stopped_clock: None,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        cli.main(["upcoming", "--cached"])

        printed = capsys.readouterr().out
        assert "2 upcoming Premier League Fixtures" in printed
        assert "barnsley" not in printed

    def test_a_cache_that_was_never_filled_says_so(
        self, corpus: Path, stopped_clock: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["upcoming", "--cached"]) == 1

        assert "has ever been fetched" in capsys.readouterr().out


class TestSeal:
    def test_it_seals_the_round_and_commits_it(
        self, corpus: Path, rolling: Path, repo: Path, stopped_clock: None,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert cli.main(["seal", "--cached"]) == 0

        assert store.path("2026-08-28").exists()
        assert "committed" in capsys.readouterr().out
        assert schema.audit(store.read()) == []

    def test_a_second_run_in_the_same_round_is_a_no_op_rather_than_a_failure(
        self, corpus: Path, rolling: Path, repo: Path, stopped_clock: None,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Acceptance criterion 3, at the level a schedule meets it: a loop that runs hourly must
        not fail every hour after the first, and must not touch the file either."""
        cli.main(["seal", "--cached"])
        before = store.path("2026-08-28").read_bytes()
        capsys.readouterr()

        assert cli.main(["seal", "--cached"]) == 0

        assert "already sealed" in capsys.readouterr().out
        assert store.path("2026-08-28").read_bytes() == before
        assert len(store.sealed_rounds()) == 1

    def test_a_correction_is_a_new_revision_at_a_new_instant(
        self, corpus: Path, rolling: Path, repo: Path, stopped_clock: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cli.main(["seal", "--cached"])
        monkeypatch.setattr(cli, "clock", lambda: pd.Timestamp("2026-08-28 18:00:00"))

        assert cli.main(["seal", "--cached", "--supersede"]) == 0

        assert store.path("2026-08-28", revision=1).exists()
        assert set(store.read()["as_of_instant"]) == {
            pd.Timestamp("2026-08-28"),
            pd.Timestamp("2026-08-28 18:00:00"),
        }

    def test_an_uncommitted_seal_is_reported_as_unproven(
        self, corpus: Path, rolling: Path, stopped_clock: None,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """No repository here, so the round cannot be committed. It was still written — and the
        run has to say the file proves nothing yet rather than report success."""
        assert cli.main(["seal", "--cached"]) == 1

        assert "NOT COMMITTED" in capsys.readouterr().out

    def test_the_pundits_are_named_as_silent_rather_than_missing(
        self, corpus: Path, rolling: Path, repo: Path, stopped_clock: None,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        cli.main(["seal", "--cached"])

        printed = capsys.readouterr().out
        assert "silent (cover none of this round)" in printed
        assert "lawrenson" in printed
        assert "ceiling_line" in printed


class TestScore:
    def test_nothing_sealed_yet_is_said_rather_than_scored(
        self, corpus: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["score", "--no-ingest"]) == 0

        assert "nothing has been sealed yet" in capsys.readouterr().out

    def test_a_sealed_round_whose_fixtures_are_unplayed_scores_nothing_yet(
        self, corpus: Path, rolling: Path, repo: Path, stopped_clock: None,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        cli.main(["seal", "--cached"])
        capsys.readouterr()

        assert cli.main(["score", "--no-ingest"]) == 0

        assert "none of whose Fixtures has been played" in capsys.readouterr().out

    def test_results_arriving_turn_a_sealed_round_into_a_scored_one(
        self, corpus: Path, rolling: Path, repo: Path, stopped_clock: None,
        make_matches: Callable[..., pd.DataFrame], capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Acceptance criterion 5: sealed Predictions are scored retrospectively, with no step in
        between beyond the results being ingested."""
        cli.main(["seal", "--cached"])
        _played(corpus, make_matches)
        capsys.readouterr()

        assert cli.main(["score", "--no-ingest"]) == 0

        printed = capsys.readouterr().out
        assert "2026/27, sealed and scored" in printed
        assert "market_line" in printed
        assert cli.live_scoreboard_path().exists()

    def test_the_live_board_is_its_own_file(
        self, corpus: Path, rolling: Path, repo: Path, stopped_clock: None,
        make_matches: Callable[..., pd.DataFrame],
    ) -> None:
        """The Evaluation Window's board is over closed Seasons and must go on meaning the same
        thing week to week, so the live Season is scored beside it and never into it."""
        from epl.ledger import scoreboard

        cli.main(["seal", "--cached"])
        _played(corpus, make_matches)
        cli.main(["score", "--no-ingest"])

        board = pd.read_csv(cli.live_scoreboard_path())
        assert not scoreboard.path().exists()
        assert set(board["fixtures"]) == {2}
