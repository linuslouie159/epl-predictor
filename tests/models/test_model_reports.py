"""The model reports — where Elo shows its working.

Issue #9's acceptance is mostly about things that are true over a whole corpus: that the draw band
narrows, that promoted Clubs arrive with earned ratings, that the frozen hyperparameters are still
what the Burn-In fit produces. ``tests/models/test_elo_over_the_corpus.py`` asserts those against
the real cache. What is tested here is that a reader without the cache can still run the commands,
and that they say what they claim to say.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from epl.ledger import backtest
from epl.models import ELO
from epl.models import __main__ as cli
from epl.paths import processed_dir
from epl.windows import BURN_IN_WINDOW

#: A Burn-In Championship Season, the Premier League Season above it, and the Premier League
#: Season that two of the Championship's Clubs are promoted into. Small enough to read; shaped
#: like the thing the commands are really run against.
CORPUS: tuple[tuple[int, str, str, str, str, str], ...] = (
    (2004, "E1", "2004-09-04", "leeds", "burnley", "H"),
    (2004, "E1", "2004-09-11", "sunderland", "burnley", "H"),
    (2004, "E1", "2004-09-18", "leeds", "sunderland", "D"),
    (2004, "E1", "2004-09-25", "burnley", "leeds", "A"),
    (2004, "E0", "2004-09-04", "arsenal", "chelsea", "H"),
    (2004, "E0", "2004-09-11", "chelsea", "everton", "H"),
    (2004, "E0", "2004-09-18", "everton", "arsenal", "A"),
    (2004, "E0", "2004-09-25", "arsenal", "everton", "D"),
    (2005, "E0", "2005-08-13", "arsenal", "leeds", "H"),
    (2005, "E0", "2005-08-20", "sunderland", "arsenal", "A"),
    (2005, "E0", "2005-08-27", "leeds", "sunderland", "D"),
)


@pytest.fixture
def corpus(project_root: Path, make_matches: Callable[..., pd.DataFrame]) -> pd.DataFrame:
    """A processed match table on disk, which is what the command line reads."""
    matches = make_matches(
        *[
            {"season": season, "division": division, "date": date,
             "home_club": home, "away_club": away, "outcome": outcome}
            for season, division, date, home, away, outcome in CORPUS
        ]
    )
    processed_dir().mkdir(parents=True, exist_ok=True)
    matches.to_csv(processed_dir() / "matches.csv", index=False)
    return matches


class TestTheRatingsReport:
    def test_it_names_the_clubs_promoted_into_the_season(
        self, corpus: pd.DataFrame, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """ADR 0004's question, asked of a corpus: Leeds and Sunderland are in the Premier League
        in 2005/06 and were not in 2004/05."""
        assert cli.main(["ratings", "--season", "2005"]) == 0

        printed = capsys.readouterr().out
        assert "promoted into the Premier League (2)" in printed
        assert "leeds" in printed
        assert "sunderland" in printed

    def test_promoted_clubs_are_shown_with_the_ratings_they_earned(
        self, corpus: pd.DataFrame, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Not a shared starting value, and not from nowhere — the report says how many matches
        each rating rests on, because a distinct rating from one match is barely a rating."""
        cli.main(["ratings", "--season", "2005"])

        printed = capsys.readouterr().out
        assert "1500.0" not in printed
        assert "from 3 matches" in printed

    def test_a_season_the_corpus_does_not_hold_is_refused(self, corpus: pd.DataFrame) -> None:
        with pytest.raises(SystemExit, match=r"2019/20"):
            cli.main(["ratings", "--season", "2019"])


class TestTheDrawsReport:
    def test_it_says_so_when_there_is_nothing_stored(
        self, corpus: pd.DataFrame, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The curve is measured over stored Predictions, so an empty store is an instruction
        rather than an empty table."""
        assert cli.main(["draws"]) == 0

        assert "run `python -m epl.ledger backfill`" in capsys.readouterr().out

    def test_it_curves_what_has_been_stored(
        self, corpus: pd.DataFrame, capsys: pytest.CaptureFixture[str]
    ) -> None:
        backtest.write(backtest.backfill(ELO, corpus, seasons=[2005]))

        assert cli.main(["draws"]) == 0

        printed = capsys.readouterr().out
        assert "elo:" in printed
        assert "predicted_draw" in printed
        assert "observed_draw" in printed


class TestTheFitReport:
    def test_it_prints_what_it_found_beside_what_is_frozen(
        self, corpus: pd.DataFrame, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """ADR 0008 freezes the hyperparameters as literals, so the two can drift apart in
        silence. Printing both is what lets a reader see which way."""
        assert cli.main(["fit"]) == 0

        printed = capsys.readouterr().out
        assert "found:" in printed
        assert "frozen:" in printed
        assert "differ" in printed, "this toy corpus cannot reproduce the real fit"

    def test_it_names_the_seasons_it_fitted_on(
        self, corpus: pd.DataFrame, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The Burn-In Window's later Seasons, warmed by its first (ADR 0008)."""
        cli.main(["fit"])

        printed = capsys.readouterr().out
        assert "2001/02-2004/05" in printed
        assert "warmed from 2000/01" in printed
        assert set(BURN_IN_WINDOW) == set(range(2000, 2005))


class TestItReadsTheProcessedTable:
    def test_a_missing_match_table_is_an_instruction(self, project_root: Path) -> None:
        with pytest.raises(SystemExit, match=r"python -m epl.ingest"):
            cli.main(["fit"])
