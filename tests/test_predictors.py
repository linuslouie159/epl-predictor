"""The Predictor contract: what a Predictor is handed, and what it must hand back.

Issue #7 asks for "one Predictor contract ... with scoring code written once against it that never
special-cases a Predictor". These tests pin the two halves of that: :class:`Evidence`, the view of
the corpus a Predictor is allowed to see, and the registry the scoreboard walks.

The project's one rule — no future data, ever — is enforced here rather than audited later, so the
tests that matter most are the ones asserting a Predictor cannot reach a row it should not have.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from epl import predictors
from epl.predictors import Evidence, Predictor


@pytest.fixture
def two_seasons(make_matches: Callable[..., pd.DataFrame]) -> pd.DataFrame:
    """Three matches on three days, so a cut can be taken between any two of them."""
    return make_matches(
        {"date": "2024-08-16", "home_club": "man_united", "outcome": "H"},
        {"date": "2024-08-17", "home_club": "arsenal", "outcome": "D"},
        {"date": "2024-08-20", "home_club": "everton", "outcome": "A"},
    )


class TestEvidenceHoldsOnlyThePast:
    def test_it_hands_over_the_matches_played_before_the_as_of_instant(
        self, two_seasons: pd.DataFrame
    ) -> None:
        evidence = Evidence.before(two_seasons, datetime(2024, 8, 17))

        assert list(evidence.matches()["home_club"]) == ["man_united"]

    def test_a_match_kicking_off_on_the_as_of_day_is_not_visible(
        self, two_seasons: pd.DataFrame
    ) -> None:
        """Strictly before, not on or before.

        The As-Of Instant is midnight at the start of the anchor day, so a Fixture played that
        same day is still unplayed at the instant it would be predicted from.
        """
        evidence = Evidence.before(two_seasons, datetime(2024, 8, 16))

        assert evidence.matches().empty


class TestEvidenceRecordsWhatItSaw:
    """Issue #7: "each Prediction records which input rows it saw".

    The record is taken at the moment the rows change hands, not guessed afterwards, because it is
    what the stored leak check is later audited against.
    """

    def test_it_has_seen_nothing_until_it_is_read(self, two_seasons: pd.DataFrame) -> None:
        evidence = Evidence.before(two_seasons, datetime(2024, 8, 17))

        assert evidence.rows_seen == 0
        assert evidence.latest_seen is None

    def test_reading_records_the_rows_and_the_latest_of_them(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        matches = make_matches(
            {"date": "2024-08-16", "time": "20:00"},
            {"date": "2024-08-17", "time": "15:00"},
        )
        evidence = Evidence.before(matches, datetime(2024, 8, 20))
        evidence.matches()

        assert evidence.rows_seen == 2
        assert evidence.latest_seen == pd.Timestamp("2024-08-17 15:00")

    def test_narrowing_to_a_tier_records_only_the_rows_handed_over(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        matches = make_matches(
            {"date": "2024-08-16", "division": "E0"},
            {"date": "2024-08-17", "division": "E3"},
        )
        evidence = Evidence.before(matches, datetime(2024, 8, 20))
        evidence.matches(divisions=("E0",))

        assert evidence.rows_seen == 1
        assert evidence.latest_seen == pd.Timestamp("2024-08-16")

    def test_reading_the_same_rows_twice_does_not_count_them_twice(
        self, two_seasons: pd.DataFrame
    ) -> None:
        """A Predictor that reads a tier and then the whole pyramid saw the pyramid, not both."""
        evidence = Evidence.before(two_seasons, datetime(2024, 8, 20))
        evidence.matches(divisions=("E0",))
        evidence.matches()

        assert evidence.rows_seen == 2


class Coin:
    """A Predictor that ignores the Evidence entirely — the smallest thing the contract accepts.

    Structural, like every other Predictor in this project: nothing subclasses the Protocol,
    because the contract is what an object answers rather than what it inherits.
    """

    name = "coin"

    def predict(self, fixtures: pd.DataFrame, evidence: Evidence) -> npt.NDArray[np.float64]:
        return np.tile([1 / 3, 1 / 3, 1 / 3], (len(fixtures), 1))


class TestTheContract:
    def test_anything_with_a_name_and_a_predict_is_a_predictor(self) -> None:
        """Structural, not inherited: the Market Line and a Pundit are not models, and nothing in
        the contract should make them pretend to be."""

        class Duck:
            name = "duck"

            def predict(
                self, fixtures: pd.DataFrame, evidence: Evidence
            ) -> npt.NDArray[np.float64]:
                return np.empty((0, 3))

        assert isinstance(Duck(), Predictor)

    def test_something_without_a_predict_is_not_a_predictor(self) -> None:
        class NotOne:
            name = "not_one"

        assert not isinstance(NotOne(), Predictor)


class TestTheRegistry:
    def test_a_registered_predictor_is_listed(self, registry: dict[str, object]) -> None:
        coin = predictors.register(Coin())

        assert predictors.registered() == (coin,)

    def test_registering_returns_the_predictor_so_it_can_be_named_at_module_level(
        self, registry: dict[str, object]
    ) -> None:
        coin = Coin()

        assert predictors.register(coin) is coin

    def test_a_registered_predictor_can_be_looked_up_by_name(
        self, registry: dict[str, object]
    ) -> None:
        coin = predictors.register(Coin())

        assert predictors.by_name("coin") is coin

    def test_an_unknown_name_names_what_is_registered(self, registry: dict[str, object]) -> None:
        predictors.register(Coin())

        with pytest.raises(predictors.PredictorError, match="coin"):
            predictors.by_name("dice")

    def test_two_predictors_may_not_share_a_name(self, registry: dict[str, object]) -> None:
        """Names key the ledger and name the backtest file. Two Predictors under one name would
        merge two track records into one and nothing would look wrong."""
        predictors.register(Coin())

        with pytest.raises(predictors.PredictorError, match="already"):
            predictors.register(Coin())

    @pytest.mark.parametrize("name", ["Coin", "naive baseline", "naive-baseline", "", "a/b"])
    def test_a_name_that_is_not_a_slug_is_refused(
        self, registry: dict[str, object], name: str
    ) -> None:
        """A Predictor's name becomes a filename under outputs/backtest/ and a value in every
        ledger row, so it is a slug or it is nothing."""
        rogue = Coin()
        rogue.name = name

        with pytest.raises(predictors.PredictorError, match="name"):
            predictors.register(rogue)
