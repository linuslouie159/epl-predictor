"""The Pundit command line, end to end.

``build`` is the one that has to be believed. It prints its own evidence — how many published
results agreed with Football-Data, which ones did not, and which Fixtures nobody called — because a
backfill whose only output is a row count is a backfill nobody can check. A report nobody can run
is not a report, so the path from cached HTML to a printed cross-check is tested here rather than
assumed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from epl.ingest import cache
from epl.paths import processed_dir
from epl.pundits import __main__ as cli
from epl.pundits import dataset, myfootballfacts

PAGE = myfootballfacts.PAGES[0]

CALLS = [
    ("Arsenal 2-0 Chelsea", "1-0"),
    ("Everton 1-1 Burnley", "0-2"),
    ("Watford 0-2 Liverpool", "0-3"),
]


@pytest.fixture
def corpus(project_root: Path, make_matches: Callable[..., pd.DataFrame]) -> Path:
    """A processed match table on disk, which is what the command line reads."""
    matches = make_matches(
        *[
            {
                "season": PAGE.season,
                "date": "2017-08-12",
                "home_club": home,
                "away_club": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "outcome": outcome,
            }
            # The page publishes 1-0, 0-2 and 0-3 beside these three calls. Football-Data agrees
            # about Arsenal and contradicts the other two, which is what the report has to show.
            for home, away, home_goals, away_goals, outcome in [
                ("arsenal", "chelsea", 1, 0, "H"),
                ("everton", "burnley", 1, 1, "D"),
                ("watford", "liverpool", 2, 0, "H"),
                ("fulham", "tottenham", 1, 1, "D"),
            ]
        ]
    )
    processed_dir().mkdir(parents=True, exist_ok=True)
    path = processed_dir() / "matches.csv"
    matches.to_csv(path, index=False)
    return path


@pytest.fixture
def cached_page(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One page in the raw cache, with the size guard relaxed to fit three calls."""
    monkeypatch.setattr(myfootballfacts, "MIN_CALLS", 1)
    monkeypatch.setattr(myfootballfacts, "PAGES", (PAGE,))
    body = "".join(f"<tr><td>{listing}</td><td>{result}</td></tr>" for listing, result in CALLS)
    cache.store(
        myfootballfacts.raw_page_path(PAGE),
        f"<html><body><table>{body}</table></body></html>".encode(),
    )


@pytest.fixture
def frozen(project_root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the frozen dataset at a temporary file, so a test never rewrites the committed one."""
    destination = tmp_path / "predictions.csv"
    monkeypatch.setattr(dataset, "path", lambda: destination)
    return destination


class TestBuild:
    def test_it_writes_the_frozen_dataset(
        self, corpus: Path, cached_page: None, frozen: Path
    ) -> None:
        assert cli.main(["build"]) == 0

        assert list(pd.read_csv(frozen)["home_club"]) == ["arsenal", "everton", "watford"]

    def test_it_prints_the_cross_check_against_football_data(
        self, corpus: Path, cached_page: None, frozen: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Two of the three published results here are wrong on purpose, and the point of the
        command is that a reader is shown them rather than told a count was fine."""
        cli.main(["build"])

        printed = capsys.readouterr().out
        assert "1 of 3 published results match Football-Data" in printed
        assert "everton" in printed and "watford" in printed

    def test_a_call_with_no_published_result_is_not_counted_as_a_check(
        self, corpus: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch, frozen: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A postponed listing carries a call and no score. In the denominator it would report a
        check that was never made as one that passed."""
        monkeypatch.setattr(myfootballfacts, "MIN_CALLS", 1)
        monkeypatch.setattr(myfootballfacts, "PAGES", (PAGE,))
        body = "".join(
            f"<tr><td>{listing}</td><td>{result}</td></tr>"
            for listing, result in [CALLS[0], (CALLS[1][0], "PP")]
        )
        cache.store(
            myfootballfacts.raw_page_path(PAGE),
            f"<html><body><table>{body}</table></body></html>".encode(),
        )

        cli.main(["build"])

        assert "1 of 1 published results match Football-Data" in capsys.readouterr().out

    def test_it_names_the_fixtures_nobody_called(
        self, corpus: Path, cached_page: None, frozen: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An unremarked gap is how twelve quietly becomes a hundred."""
        cli.main(["build"])

        printed = capsys.readouterr().out
        assert "1 of 4 Fixtures have no call" in printed
        assert "fulham v tottenham" in printed

    def test_it_attributes_the_origin_rather_than_only_the_archive(
        self, corpus: Path, cached_page: None, frozen: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli.main(["build"])

        assert "BBC by way of MyFootballFacts" in capsys.readouterr().out


class TestGrades:
    def test_it_prints_both_readings_per_season_and_per_career(
        self, corpus: Path, cached_page: None, frozen: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli.main(["build"])
        capsys.readouterr()

        assert cli.main(["grades"]) == 0

        printed = capsys.readouterr().out
        assert "exact_rate" in printed and "outcome_rate" in printed
        assert "2017/18" in printed


class TestFetch:
    def test_it_reports_where_every_page_landed(
        self, project_root: Path, cached_page: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["fetch"]) == 0

        assert PAGE.path in capsys.readouterr().out
