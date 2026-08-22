"""Scoring the ledger: one line per registered Predictor over the Evaluation Window.

The scoreboard is where issue #7's first acceptance criterion is cashed in — scoring code written
once against the Predictor contract that never special-cases a Predictor. It reads ledger rows and
the match table and nothing else, so it cannot tell which store a Prediction came from or which
Predictor made it beyond the name it prints.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from epl import predictors
from epl.ledger import schema, scoreboard
from epl.predictors import Evidence

CERTAIN_HOME = (1.0, 0.0, 0.0)
CERTAIN_AWAY = (0.0, 0.0, 1.0)


@pytest.fixture
def played(make_matches: Callable[..., pd.DataFrame]) -> pd.DataFrame:
    """Two Premier League Fixtures of 2005/06, both Home wins."""
    return make_matches(
        {"season": 2005, "date": "2005-08-13", "home_club": "arsenal", "outcome": "H"},
        {"season": 2005, "date": "2005-08-13", "home_club": "everton", "outcome": "H"},
    )


def _rows(
    predictor: object, fixtures: pd.DataFrame, as_of: str = "2005-08-12"
) -> pd.DataFrame:
    return schema.predictions_for(predictor, fixtures, Evidence.before(fixtures.head(0), as_of))


class TestTheScoreboard:
    def test_it_scores_each_predictor_on_the_outcomes_that_happened(
        self, played: pd.DataFrame, make_predictor: Callable[..., object]
    ) -> None:
        """Both Fixtures were Home wins: certainty on Home scores 0.00 RPS, certainty on Away
        scores 1.00. Worked by hand in tests/metrics/test_scores.py."""
        rows = pd.concat(
            [
                _rows(make_predictor("oracle", CERTAIN_HOME), played),
                _rows(make_predictor("fool", CERTAIN_AWAY), played),
            ],
            ignore_index=True,
        )

        board = scoreboard.build(rows, played)

        assert list(board["predictor"]) == ["oracle", "fool"]
        assert list(board["rps"]) == [0.0, 1.0]
        assert list(board["fixtures"]) == [2, 2]

    def test_the_best_predictor_is_listed_first(
        self, played: pd.DataFrame, make_predictor: Callable[..., object]
    ) -> None:
        """RPS is primary and lower is better, so the scoreboard is sorted by it (CLAUDE.md)."""
        rows = pd.concat(
            [
                _rows(make_predictor("fool", CERTAIN_AWAY), played),
                _rows(make_predictor("oracle", CERTAIN_HOME), played),
            ],
            ignore_index=True,
        )

        assert list(scoreboard.build(rows, played)["predictor"]) == ["oracle", "fool"]

    def test_it_reports_every_metric_the_module_produces(
        self, played: pd.DataFrame, make_predictor: Callable[..., object]
    ) -> None:
        board = scoreboard.build(_rows(make_predictor("oracle", CERTAIN_HOME), played), played)

        assert list(board.columns) == list(scoreboard.SCOREBOARD_COLUMNS)
        assert board.iloc[0]["accuracy"] == 1.0
        assert board.iloc[0]["brier"] == 0.0

    def test_a_fixture_with_no_outcome_yet_is_not_scored(
        self, make_matches: Callable[..., pd.DataFrame], make_predictor: Callable[..., object]
    ) -> None:
        """Sealed Predictions exist before their Fixtures are played. An unplayed Fixture is absent
        from the scoreboard, not scored as a miss."""
        fixtures = make_matches(
            {"season": 2005, "date": "2005-08-13", "home_club": "arsenal", "outcome": "H"},
            {"season": 2005, "date": "2005-08-13", "home_club": "everton", "outcome": pd.NA},
        )

        board = scoreboard.build(_rows(make_predictor(), fixtures), fixtures)

        assert board.iloc[0]["fixtures"] == 1

    def test_a_postponed_fixture_is_still_the_fixture_that_was_predicted(
        self, played: pd.DataFrame, make_predictor: Callable[..., object]
    ) -> None:
        """The join is on the Club pairing, not the date. A Fixture moved after its Prediction was
        sealed would otherwise silently drop off the scoreboard."""
        rows = _rows(make_predictor("oracle", CERTAIN_HOME), played)
        rearranged = played.assign(date=pd.to_datetime("2005-09-20").date())

        assert scoreboard.build(rows, rearranged).iloc[0]["fixtures"] == 2

    def test_only_the_latest_prediction_for_a_fixture_counts(
        self, played: pd.DataFrame, make_predictor: Callable[..., object]
    ) -> None:
        """Superseding is how a sealed mistake is corrected (ADR 0005): a new row at a new As-Of
        Instant. Scoring both would count the Fixture twice and average in the mistake."""
        wrong = _rows(make_predictor("oracle", CERTAIN_AWAY), played)
        corrected = _rows(make_predictor("oracle", CERTAIN_HOME), played)
        corrected["as_of_instant"] += pd.Timedelta(hours=6)

        board = scoreboard.build(pd.concat([wrong, corrected], ignore_index=True), played)

        assert board.iloc[0]["fixtures"] == 2
        assert board.iloc[0]["rps"] == 0.0

    def test_seasons_outside_the_evaluation_window_are_not_scored(
        self, make_matches: Callable[..., pd.DataFrame], make_predictor: Callable[..., object]
    ) -> None:
        """Nothing in the Burn-In Window is ever scored — that is what keeps the Evaluation Window
        uncontaminated (ADR 0008)."""
        burn_in = make_matches(
            {"season": 2002, "date": "2002-08-17", "home_club": "arsenal", "outcome": "H"}
        )

        board = scoreboard.build(_rows(make_predictor(), burn_in, "2002-08-16"), burn_in)

        assert board.empty

    def test_an_empty_ledger_gives_an_empty_scoreboard(self, played: pd.DataFrame) -> None:
        board = scoreboard.build(schema.empty(), played)

        assert board.empty
        assert list(board.columns) == list(scoreboard.SCOREBOARD_COLUMNS)


class TestTheScoreboardFile:
    def test_it_is_written_where_regenerable_output_belongs(
        self, project_root: Path, played: pd.DataFrame, make_predictor: Callable[..., object]
    ) -> None:
        """A scoreboard is derived from the stores, so it is regenerable and gitignored — the same
        reasoning ADR 0005 applies to outputs/backtest/ itself. It sits beside the two stores
        rather than inside one, so that reading a store never picks up a report."""
        board = scoreboard.build(_rows(make_predictor(), played), played)

        written = scoreboard.write(board)

        assert written.name == "scoreboard.csv"
        assert written.parent.name == "outputs"
        assert pd.read_csv(written)["predictor"].tolist() == ["fixed"]


class TestTheCaveatsTravelWithTheScores:
    """A Predictor may carry a note, and it has to reach every place its score is reported.

    The Ceiling Line is why (issue #8): a line that knows team news the model cannot have, scored
    over a shorter span than everything else on the board, is a misleading number rather than an
    incomplete one if it appears bare. The mechanism is generic — the scoreboard reads a note off
    whatever Predictor the row names, and has no idea which one that is.
    """

    def test_a_note_reaches_the_board(
        self, played: pd.DataFrame, make_predictor: Callable[..., object], registry: dict
    ) -> None:
        caveated = make_predictor("caveated", CERTAIN_HOME)
        caveated.note = "knows something the others do not"
        predictors.register(caveated)

        board = scoreboard.build(_rows(caveated, played), played, seasons=[2005])

        assert list(board["note"]) == ["knows something the others do not"]

    def test_a_predictor_without_one_gets_a_blank(
        self, played: pd.DataFrame, make_predictor: Callable[..., object], registry: dict
    ) -> None:
        plain = predictors.register(make_predictor("plain", CERTAIN_HOME))

        board = scoreboard.build(_rows(plain, played), played, seasons=[2005])

        assert list(board["note"]) == [""]

    def test_a_stored_predictor_nobody_registered_still_scores(
        self, played: pd.DataFrame, make_predictor: Callable[..., object], registry: dict
    ) -> None:
        """A ledger file can outlive the code that wrote it (ADR 0005). Scoring must not depend on
        the Predictor still being registered — only the note does, and its absence is a blank."""
        board = scoreboard.build(_rows(make_predictor("ghost"), played), played, seasons=[2005])

        assert board.loc[0, "fixtures"] == 2
        assert board.loc[0, "note"] == ""
