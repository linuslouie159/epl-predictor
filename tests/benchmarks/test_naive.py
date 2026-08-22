"""The Naive Baseline: base rates, and nothing else.

The floor Predictor (CONTEXT.md). It knows how often Home, Draw and Away have happened and nothing
about which Clubs are playing, so anything that fails to beat it has no value.

Issue #7 asks for one property beyond that: it "estimates base rates only from Seasons already
seen, so even the floor is leak-free". That is what these tests are mostly about — a floor fitted
on the whole history would be a floor that knew the future, and every Predictor measured against it
would look worse than it is.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from epl import predictors
from epl.benchmarks import NAIVE_BASELINE, NaiveBaseline
from epl.predictors import Evidence


@pytest.fixture
def round_fixtures(make_matches: Callable[..., pd.DataFrame]) -> pd.DataFrame:
    return make_matches(
        {"date": "2024-08-17", "home_club": "arsenal", "away_club": "wolves"},
        {"date": "2024-08-17", "home_club": "everton", "away_club": "brighton"},
    )


def _evidence(matches: pd.DataFrame, as_of: str = "2024-08-16") -> Evidence:
    return Evidence.before(matches, pd.Timestamp(as_of))


class TestTheBaseRates:
    def test_it_predicts_the_frequencies_of_the_outcomes_it_has_seen(
        self, round_fixtures: pd.DataFrame, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """Four matches: two Home, one Draw, one Away — so 0.5 / 0.25 / 0.25."""
        played = make_matches(
            {"date": "2024-08-13", "outcome": "H"},
            {"date": "2024-08-13", "outcome": "H"},
            {"date": "2024-08-14", "outcome": "D"},
            {"date": "2024-08-14", "outcome": "A"},
        )

        predicted = NaiveBaseline().predict(round_fixtures, _evidence(played))

        assert list(predicted[0]) == [0.5, 0.25, 0.25]

    def test_it_says_the_same_thing_about_every_fixture_in_the_round(
        self, round_fixtures: pd.DataFrame, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """The definition of the floor: it does not know which Clubs are playing."""
        played = make_matches({"date": "2024-08-13", "outcome": "H"})

        predicted = NaiveBaseline().predict(round_fixtures, _evidence(played))

        assert list(predicted[0]) == list(predicted[1])

    def test_it_counts_only_the_seasons_already_played(
        self, round_fixtures: pd.DataFrame, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """The leak-free floor. The two Away wins after the As-Of Instant are invisible, so the
        rates are the ones a Predictor could actually have quoted that morning."""
        played = make_matches(
            {"date": "2024-08-13", "outcome": "H"},
            {"date": "2024-08-14", "outcome": "D"},
            {"date": "2024-08-17", "outcome": "A"},
            {"date": "2024-08-18", "outcome": "A"},
        )

        predicted = NaiveBaseline().predict(round_fixtures, _evidence(played))

        assert list(predicted[0]) == [0.5, 0.5, 0.0]

    def test_it_ignores_tiers_it_is_not_predicting(
        self, round_fixtures: pd.DataFrame, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """League Two's home advantage is not evidence about the Premier League. The pyramid is
        rated as one for Elo (ADR 0004) because Clubs move between tiers; a base rate has no Club
        in it to move."""
        played = make_matches(
            {"date": "2024-08-13", "division": "E0", "outcome": "H"},
            {"date": "2024-08-13", "division": "E3", "outcome": "A"},
            {"date": "2024-08-13", "division": "E3", "outcome": "A"},
        )

        predicted = NaiveBaseline().predict(round_fixtures, _evidence(played))

        assert list(predicted[0]) == [1.0, 0.0, 0.0]

    def test_having_seen_nothing_it_says_a_third_each(
        self, round_fixtures: pd.DataFrame, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """Its first-ever round. A Predictor owes a Prediction for every Fixture it is handed, and
        a third each is the honest statement of no information."""
        predicted = NaiveBaseline().predict(round_fixtures, _evidence(make_matches()))

        assert list(predicted[0]) == pytest.approx([1 / 3, 1 / 3, 1 / 3])

    def test_it_records_the_rows_it_counted(
        self, round_fixtures: pd.DataFrame, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        played = make_matches(
            {"date": "2024-08-13", "division": "E0"},
            {"date": "2024-08-14", "division": "E3"},
        )
        evidence = _evidence(played)

        NaiveBaseline().predict(round_fixtures, evidence)

        assert evidence.rows_seen == 1
        assert evidence.latest_seen == pd.Timestamp("2024-08-13")


class TestItIsOnTheScoreboard:
    def test_it_is_registered_under_its_slug(self) -> None:
        assert predictors.by_name("naive_baseline") is NAIVE_BASELINE

    def test_it_satisfies_the_predictor_contract(self) -> None:
        assert isinstance(NAIVE_BASELINE, predictors.Predictor)
