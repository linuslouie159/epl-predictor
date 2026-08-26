"""Where the real champion landed, and what the check is allowed to conclude from it.

Issue #15's validation criterion, tested without fitting a posterior — the machinery is what is
under test here, not the corpus, and a real run is a hundred and twenty-six four-minute fits. The
Seasons below are three-Club leagues with a champion decided by hand, so a claim like "the eventual
champion was the projection's favourite" has a right answer rather than an approximate one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epl.simulate.checkpoints import CheckpointError
from epl.simulate.posterior import Diagnostics, Sampling
from epl.simulate.projection import At, Bands, Projection, ProjectionError, Simulation
from epl.simulate.validation import (
    EVENTS,
    VALIDATION_COLUMNS,
    Validation,
    final_positions,
    rows_for,
)

CLUBS = ("a", "b", "c")
THREE = Bands(title=1, european=1, relegation=1)


def diagnostics() -> Diagnostics:
    return Diagnostics(0, 1.0, 1000.0, 1000.0, 1, 1, 1, 20260825)


def projection_of(
    *, title: tuple[float, ...], season: int = 2015, checkpoint_day: str = "2015-12-25"
) -> Projection:
    """A projection that says exactly these title probabilities, and nothing else interesting.

    Built by hand rather than walked, so a calibration test is about the arithmetic of the check
    and not about whether a hundred simulated Seasons landed where they were expected to.
    """
    seasons = 1000
    finishes = np.zeros((len(CLUBS), len(CLUBS)), dtype=np.int64)
    finishes[:, 0] = np.rint(np.array(title) * seasons).astype(np.int64)
    finishes[:, 2] = seasons - finishes[:, 0]
    return Projection(
        clubs=CLUBS,
        finishes=finishes,
        points=np.zeros(len(CLUBS), dtype=np.int64),
        played=np.zeros(len(CLUBS), dtype=np.int64),
        final_points=np.zeros(len(CLUBS), dtype=np.int64),
        remaining=6,
        draws=4,
        level_pairs=0,
        simulation=Simulation(seasons=seasons),
        bands=THREE,
        diagnostics=diagnostics(),
        at=At(season, pd.Timestamp(checkpoint_day), checkpoint_day),
    )


def finished(*order: str) -> pd.Series:
    return pd.Series(range(1, len(order) + 1), index=list(order), name="position")


def validation_of(*projections: tuple[Projection, pd.Series]) -> Validation:
    frames = [
        rows_for(projection, table, index + 1, len(projections))
        for index, (projection, table) in enumerate(projections)
    ]
    return Validation(
        rows=pd.concat(frames, ignore_index=True),
        bands=THREE,
        simulation=Simulation(),
        sampling=Sampling(),
    )


class TestJoiningAProjectionToWhatHappened:
    def test_every_club_gets_a_row_carrying_both_the_promise_and_the_outcome(self) -> None:
        rows = rows_for(projection_of(title=(0.7, 0.2, 0.1)), finished("b", "a", "c"), 1, 6)

        assert list(rows.columns) == list(VALIDATION_COLUMNS)
        assert len(rows) == 3
        assert list(rows["title_happened"]) == [False, True, False]
        assert list(rows["relegation_happened"]) == [False, False, True]

    def test_a_club_missing_from_the_final_table_is_an_error(self) -> None:
        with pytest.raises(ProjectionError, match="final table"):
            rows_for(projection_of(title=(0.7, 0.2, 0.1)), finished("a", "b"), 1, 6)


class TestWhereTheRealChampionLanded:
    def test_the_champion_is_ranked_inside_the_projection_that_named_a_favourite(self) -> None:
        board = validation_of(
            (projection_of(title=(0.7, 0.2, 0.1)), finished("a", "b", "c")),
            (projection_of(title=(0.7, 0.2, 0.1), season=2016), finished("c", "b", "a")),
        )

        champions = board.champions().set_index("season")

        assert champions.loc[2015, "champion"] == "a"
        assert champions.loc[2015, "rank"] == 1
        assert champions.loc[2016, "champion"] == "c"
        assert champions.loc[2016, "rank"] == 3
        assert list(champions["favourite"]) == ["a", "a"]

    def test_the_probability_the_projection_gave_the_eventual_champion_is_the_headline(
        self,
    ) -> None:
        board = validation_of((projection_of(title=(0.7, 0.2, 0.1)), finished("c", "b", "a")))

        assert board.champions().loc[0, "title"] == pytest.approx(0.1)
        assert "0.100" in board.describe()

    def test_a_projection_counts_once_however_many_clubs_it_covers(self) -> None:
        board = validation_of(
            (projection_of(title=(0.7, 0.2, 0.1)), finished("a", "b", "c")),
            (projection_of(title=(0.5, 0.5, 0.0), season=2016), finished("a", "b", "c")),
        )

        assert board.projections == 2
        assert len(board.rows) == 6


class TestWhetherItIsCalibratedOrMerelyPlausible:
    def test_a_projection_that_promised_what_happened_has_no_gap(self) -> None:
        """Ten Seasons, each given a 100% champion who then won. Perfectly calibrated."""
        board = validation_of(
            *[
                (projection_of(title=(1.0, 0.0, 0.0), season=2000 + n), finished("a", "b", "c"))
                for n in range(10)
            ]
        )

        assert board.calibration_error("title") == pytest.approx(0.0)

    def test_a_projection_certain_of_the_wrong_club_is_caught(self) -> None:
        board = validation_of(
            *[
                (projection_of(title=(1.0, 0.0, 0.0), season=2000 + n), finished("c", "b", "a"))
                for n in range(10)
            ]
        )

        top = board.reliability("title").iloc[-1]

        assert top["mean_predicted"] == pytest.approx(1.0)
        assert top["observed"] == pytest.approx(0.0)
        # Ten certainties that did not happen, at a gap of 1.0, and twenty write-offs of which
        # half did: (10 x 1.0 + 20 x 0.5) / 30.
        assert board.calibration_error("title") == pytest.approx(2 / 3)

    def test_the_diagram_keeps_a_row_for_every_band_it_never_used(self) -> None:
        board = validation_of((projection_of(title=(0.7, 0.2, 0.1)), finished("a", "b", "c")))

        diagram = board.reliability("title")

        assert len(diagram) == 10
        assert diagram["projections"].sum() == 3
        assert diagram.loc[diagram["projections"] == 0, "observed"].isna().all()

    def test_the_three_events_pool_by_default_because_one_alone_is_too_thin(self) -> None:
        board = validation_of((projection_of(title=(0.7, 0.2, 0.1)), finished("a", "b", "c")))

        assert board.reliability().to_numpy().shape == (10, 6)
        assert board.reliability()["projections"].sum() == 3 * len(EVENTS)

    def test_an_event_a_projection_does_not_report_is_an_error(self) -> None:
        board = validation_of((projection_of(title=(0.7, 0.2, 0.1)), finished("a", "b", "c")))

        with pytest.raises(ProjectionError, match="no such event"):
            board.reliability("golden_boot")


class TestTheFinalTableItIsMarkedAgainst:
    def season(self, *results: tuple[str, str, int, int], season: int = 2024) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "season": season,
                "division": "E0",
                "date": pd.to_datetime("2024-08-17").date(),
                "time": pd.NA,
                "home_club": [home for home, _, _, _ in results],
                "away_club": [away for _, away, _, _ in results],
                "home_goals": [scored for *_, scored, _ in results],
                "away_goals": [conceded for *_, conceded in results],
            }
        )

    def test_the_real_table_is_settled_by_the_same_chain_the_simulated_ones_are(self) -> None:
        """a and b level on points, goal difference and goals scored; b won the head-to-head.

        Every Club plays twice, because a Season whose Clubs have played different numbers of
        matches is one this refuses — see the test below it.
        """
        table = self.season(("b", "a", 1, 0), ("a", "c", 1, 0), ("d", "b", 1, 0), ("c", "d", 3, 0))

        positions = final_positions(table, 2024, seed=0)

        assert positions["b"] < positions["a"]

    def test_a_season_still_being_played_has_no_final_table_to_validate_against(self) -> None:
        table = self.season(("a", "b", 1, 0), ("a", "c", 1, 0))

        with pytest.raises(ProjectionError, match="not finished"):
            final_positions(table, 2024, seed=0)

    def test_a_season_the_corpus_does_not_hold_is_an_error(self) -> None:
        with pytest.raises(CheckpointError, match="2019/20"):
            final_positions(self.season(("a", "b", 1, 0)), 2019, seed=0)

    def test_the_real_table_is_reproducible_from_its_seed(self) -> None:
        """The chain ends in a coin flip, so even a finished Season needs a recorded seed."""
        table = self.season(("a", "b", 0, 0))

        first = final_positions(table, 2024, seed=4)
        again = final_positions(table, 2024, seed=4)

        assert first.equals(again)


class TestAnEmptyValidation:
    def test_it_says_so_rather_than_dividing_by_zero(self) -> None:
        board = Validation(
            rows=pd.DataFrame(columns=list(VALIDATION_COLUMNS)),
            bands=THREE,
            simulation=Simulation(),
            sampling=Sampling(),
        )

        assert board.projections == 0
        assert board.describe() == "no projections"
        assert board.champions().empty
        with pytest.raises(ProjectionError, match="no projections"):
            board.calibration_error()
