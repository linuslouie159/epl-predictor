"""The benchmarks reports — where the vig removal shows its working.

Issue #8 asks that the overround be "reported alongside every Market Line ... so the vig removal
can be sanity-checked rather than trusted". A report nobody can run is not a report, so the path
from a match table to a printed margin is tested end to end here.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from epl.benchmarks import __main__ as cli
from epl.paths import processed_dir

TYPICAL = {"prematch_odds_home": 1.80, "prematch_odds_draw": 3.60, "prematch_odds_away": 4.50}
TYPICAL_CLOSE = {
    "closing_odds_home": 1.74,
    "closing_odds_draw": 3.70,
    "closing_odds_away": 4.75,
}


@pytest.fixture
def corpus(project_root: Path, make_matches: Callable[..., pd.DataFrame]) -> Path:
    """A processed match table on disk, which is what the command line reads."""
    matches = make_matches(
        {"season": 2005, "date": "2005-08-13", "home_club": "arsenal", **TYPICAL},
        {"season": 2005, "date": "2005-08-13", "home_club": "everton", **TYPICAL},
        {"season": 2019, "date": "2019-08-17", "home_club": "fulham", **TYPICAL, **TYPICAL_CLOSE},
        {"season": 2019, "date": "2019-08-17", "division": "E1", "home_club": "wolves", **TYPICAL},
    )
    processed_dir().mkdir(parents=True, exist_ok=True)
    path = processed_dir() / "matches.csv"
    matches.to_csv(path, index=False)
    return path


class TestTheOverroundReport:
    def test_it_writes_a_row_per_season_and_tier(self, corpus: Path) -> None:
        assert cli.main(["overround"]) == 0

        written = pd.read_csv(cli.path())
        assert set(written["predictor"]) == {"market_line", "ceiling_line"}
        assert set(zip(written["season"], written["division"], strict=True)) == {
            (2005, "E0"),
            (2019, "E0"),
            (2019, "E1"),
        }

    def test_it_reports_the_margin_that_is_actually_in_the_book(self, corpus: Path) -> None:
        """1/1.80 + 1/3.60 + 1/4.50 = 1.055556. The number is the point of the report."""
        cli.main(["overround"])

        written = pd.read_csv(cli.path())
        market = written.loc[written["predictor"] == "market_line"]
        assert market["mean_overround"].tolist() == pytest.approx([1.05556] * 3, abs=1e-5)

    def test_the_ceiling_line_is_reported_only_where_it_has_a_book(self, corpus: Path) -> None:
        cli.main(["overround"])

        written = pd.read_csv(cli.path())
        ceiling = written.loc[written["predictor"] == "ceiling_line"]
        assert ceiling["season"].tolist() == [2019]

    def test_it_prints_the_premier_league_by_default(
        self, corpus: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Every tier is written to the file; printing all four would bury the one that matters."""
        cli.main(["overround"])

        printed = capsys.readouterr().out
        assert "market_line: 3 Fixtures" in printed

    def test_the_printed_tiers_can_be_widened(
        self, corpus: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli.main(["overround", "--divisions", "E0", "E1"])

        assert "market_line: 4 Fixtures" in capsys.readouterr().out

    def test_it_says_how_to_build_the_corpus_if_there_is_none(self, project_root: Path) -> None:
        with pytest.raises(SystemExit, match=r"epl.ingest build"):
            cli.main(["overround"])


class TestComparingTheMethods:
    def test_it_prints_all_three_and_names_the_default(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["methods"]) == 0

        printed = capsys.readouterr().out
        assert all(name in printed for name in ("normalise", "power", "shin"))
        assert "shin" in printed.split("(default)")[0].splitlines()[-1]

    def test_it_shows_the_book_it_is_working_from(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Otherwise three columns of six-figure numbers say nothing a reader can check."""
        cli.main(["methods"])

        assert "1.05556" in capsys.readouterr().out
