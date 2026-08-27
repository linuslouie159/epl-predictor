"""The Naive Baseline walked over the real corpus, end to end.

This is issue #7's headline acceptance criterion — the floor scored over the Evaluation Window at
roughly 0.2292 RPS — re-derived from the ingested data rather than trusted, in the same spirit as
``tests/ingest/test_raw_cache_integrity.py``. It also pins the two facts the design rests on that
only the full corpus can show: that the walk is leak-free at all 952 scored rounds, and that no
Fixture is still being played when an As-Of Instant falls.

Needs a populated ``data/raw/``, which is gitignored, so these skip when it is absent:

    python -m epl.ingest fetch
"""

from __future__ import annotations

import pandas as pd
import pytest

from epl.benchmarks import NAIVE_BASELINE
from epl.ingest import DIVISIONS, FIRST_SEASON, LAST_SEASON, load_matches, raw_season_path
from epl.ledger import backtest, schema, scoreboard
from epl.windows import BURN_IN_WINDOW, EVALUATION_WINDOW

pytestmark = pytest.mark.cache

#: docs/DECISIONS.md, re-verified at stage 1: 21 Seasons x 380 Fixtures.
EVALUATION_FIXTURES = 7980

#: Prediction Rounds inside the Evaluation Window, of the 1,189 across the whole corpus.
EVALUATION_ROUNDS = 952

#: What the Naive Baseline scores walking forward. The published 0.2292 is the *whole-window*
#: figure — computed from rates that already know how the window turned out. Estimating them
#: walk-forward costs 0.0002 RPS, and that difference is the leak being refused rather than a bug.
WALK_FORWARD_RPS = 0.22938
WHOLE_WINDOW_RPS = 0.2292

#: The latest kickoff time anywhere in the corpus. Pinned exactly rather than loosely, because a
#: later one appearing upstream would be real news: it is the premise of timestamping a match row
#: at its kickoff instead of at full time.
LATEST_KICKOFF = "20:15"


def _require_cache() -> None:
    missing = [
        (season, division)
        for season in range(FIRST_SEASON, LAST_SEASON + 1)
        for division in DIVISIONS
        if not raw_season_path(season, division).exists()
    ]
    if missing:
        pytest.skip(f"raw cache incomplete ({len(missing)} files missing)")


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    _require_cache()
    return load_matches()


@pytest.fixture(scope="module")
def rows(matches: pd.DataFrame) -> pd.DataFrame:
    """The Naive Baseline's Backtest Predictions over the whole Evaluation Window."""
    return backtest.backfill(NAIVE_BASELINE, matches)


class TestTheFloorIsWhereItShouldBe:
    def test_it_predicts_every_fixture_in_the_evaluation_window(self, rows: pd.DataFrame) -> None:
        assert len(rows) == EVALUATION_FIXTURES
        assert rows["prediction_round"].nunique() == EVALUATION_ROUNDS
        assert set(rows["season"]) == set(EVALUATION_WINDOW)

    def test_it_scores_roughly_the_published_floor(
        self, rows: pd.DataFrame, matches: pd.DataFrame
    ) -> None:
        board = scoreboard.build(rows, matches)
        line = board.loc[board["predictor"] == "naive_baseline"].iloc[0]

        assert line["fixtures"] == EVALUATION_FIXTURES
        assert line["rps"] == pytest.approx(WALK_FORWARD_RPS, abs=5e-5)
        assert line["rps"] == pytest.approx(WHOLE_WINDOW_RPS, abs=1e-3)

    def test_walking_forward_costs_something_measurable(
        self, rows: pd.DataFrame, matches: pd.DataFrame
    ) -> None:
        """The point of the whole exercise, stated as a number.

        Quoting the rates the full Evaluation Window turned out to have would score better. That
        it scores better is exactly why it is not allowed.
        """
        board = scoreboard.build(rows, matches)
        walk_forward = board.loc[board["predictor"] == "naive_baseline"].iloc[0]["rps"]

        assert walk_forward > WHOLE_WINDOW_RPS

    def test_its_top_pick_is_always_a_home_win(
        self, rows: pd.DataFrame, matches: pd.DataFrame
    ) -> None:
        """The floor, restated: it says Home every time, because Home is the commonest Outcome and
        it has nothing else to go on. Accuracy is therefore the Home-win rate, which is why
        accuracy is never this project's headline (CLAUDE.md)."""
        board = scoreboard.build(rows, matches)
        line = board.loc[board["predictor"] == "naive_baseline"].iloc[0]

        assert (rows["prob_home"] > rows[["prob_draw", "prob_away"]].max(axis=1)).all()
        assert line["accuracy"] == pytest.approx(0.4556, abs=1e-3)


class TestTheWalkIsLeakFree:
    def test_the_whole_backtest_audits_clean(self, rows: pd.DataFrame) -> None:
        assert schema.audit(rows) == []

    def test_no_prediction_saw_a_row_from_its_own_instant_or_later(
        self, rows: pd.DataFrame
    ) -> None:
        assert (rows["latest_input"] < rows["as_of_instant"]).all()

    def test_the_first_scored_prediction_saw_exactly_the_burn_in_window(
        self, rows: pd.DataFrame
    ) -> None:
        """Five Seasons of 380 Premier League Fixtures, warmed up and never scored (ADR 0008)."""
        assert rows.iloc[0]["inputs_seen"] == len(BURN_IN_WINDOW) * 380

    def test_evidence_grows_monotonically_across_the_window(self, rows: pd.DataFrame) -> None:
        """A walk-forward run can only ever accumulate. A dip would mean a round was predicted out
        of order, which is the shape a subtle leak would take."""
        by_round = rows.groupby("prediction_round", sort=True)["inputs_seen"].first()

        assert by_round.is_monotonic_increasing

    def test_no_fixture_is_still_being_played_when_an_as_of_instant_falls(
        self, matches: pd.DataFrame
    ) -> None:
        """What lets a match row be timestamped at its kickoff rather than at full time.

        Every As-Of Instant is a midnight. The latest kickoff anywhere in the corpus is 20:15, so
        no match is in progress at one — kickoff and full time never land on opposite sides of an
        As-Of Instant, and the loose timestamp cannot admit a result the Predictor should not have.
        """
        latest = matches.loc[matches["time"].notna(), "time"].max()

        assert latest == LATEST_KICKOFF
        assert latest < "22:00"  # the claim the timestamping rule actually rests on
