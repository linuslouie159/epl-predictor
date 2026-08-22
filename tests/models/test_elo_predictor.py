"""Elo as a Predictor: one Prediction Round at a time, through the shared contract.

Issue #9 asks for Elo "registered as a Predictor through the shared contract, predicting one
Prediction Round at a time from data strictly before that round's As-Of Instant". The contract is
tested in ``tests/test_predictors.py`` and the ledger in ``tests/ledger/``; what is left for here is
what Elo does with it — that it reads the whole pyramid (ADR 0004), that it reads *only* its
Evidence, and that it says nothing at all until it has been told something.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from epl import predictors
from epl.ledger import schema
from epl.models import ELO, Elo, Settings
from epl.models.elo import FROZEN_LOGIT, FROZEN_SETTINGS
from epl.predictors import Evidence


@pytest.fixture
def round_fixtures(make_matches: Callable[..., pd.DataFrame]) -> pd.DataFrame:
    return make_matches(
        {"date": "2024-08-17", "home_club": "arsenal", "away_club": "wolves"},
        {"date": "2024-08-17", "home_club": "everton", "away_club": "brighton"},
    )


def _evidence(matches: pd.DataFrame, as_of: str = "2024-08-16") -> Evidence:
    return Evidence.before(matches, pd.Timestamp(as_of))


class TestWhatItPredicts:
    def test_knowing_nothing_it_says_the_same_thing_about_every_fixture(
        self, round_fixtures: pd.DataFrame, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """Every Club sits at the conventional starting rating, so every Fixture has the same
        edge — which is the home advantage, and nothing else."""
        predicted = Elo().predict(round_fixtures, _evidence(make_matches()))

        assert list(predicted[0]) == pytest.approx(list(predicted[1]))
        assert predicted[0][0] > predicted[0][2], "home advantage still applies"

    def test_a_club_that_has_been_winning_is_favoured(
        self, round_fixtures: pd.DataFrame, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        played = make_matches(
            {"date": "2024-08-10", "home_club": "arsenal", "away_club": "chelsea",
             "outcome": "H"},
            {"date": "2024-08-13", "home_club": "chelsea", "away_club": "arsenal",
             "outcome": "A"},
        )

        predicted = Elo().predict(round_fixtures, _evidence(played))

        assert predicted[0][0] > predicted[1][0], "Arsenal has a rating; Everton has not"

    def test_every_prediction_is_a_distribution(
        self, round_fixtures: pd.DataFrame, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        predicted = Elo().predict(round_fixtures, _evidence(make_matches({"outcome": "H"})))

        assert predicted.shape == (2, 3)
        assert predicted.sum(axis=1) == pytest.approx([1.0, 1.0])

    def test_the_draw_probability_falls_as_supremacy_grows(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """ADR 0006 end to end: the taper is never coded, only fitted."""
        history = make_matches(
            *[
                {"date": f"2024-0{1 + week // 28}-{1 + week % 28:02d}", "home_club": "arsenal",
                 "away_club": "wolves", "outcome": "H"}
                for week in range(40)
            ]
        )
        fixtures = make_matches(
            {"date": "2024-08-17", "home_club": "everton", "away_club": "brighton"},
            {"date": "2024-08-17", "home_club": "arsenal", "away_club": "wolves"},
        )

        predicted = Elo().predict(fixtures, _evidence(history))

        assert predicted[1][0] - predicted[1][2] > predicted[0][0] - predicted[0][2]
        assert predicted[1][1] < predicted[0][1]


class TestWhatItIsAllowedToSee:
    def test_it_reads_every_tier(
        self, round_fixtures: pd.DataFrame, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """ADR 0004: a Club promoted into the Premier League must arrive with a rating it earned,
        so the Championship and below have to move ratings. The Naive Baseline narrows to the tiers
        it predicts and is the exception, not the pattern."""
        played = make_matches(
            {"date": "2024-08-13", "division": "E0"},
            {"date": "2024-08-13", "division": "E1"},
            {"date": "2024-08-13", "division": "E3"},
        )
        evidence = _evidence(played)

        Elo().predict(round_fixtures, evidence)

        assert evidence.rows_seen == 3

    def test_a_result_after_the_as_of_instant_changes_nothing(
        self, round_fixtures: pd.DataFrame, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """The project's one rule. Arsenal's thrashing of Wolves happens on the Saturday, so the
        Friday's Prediction cannot know about it."""
        before = make_matches({"date": "2024-08-13", "home_club": "arsenal",
                               "away_club": "wolves", "outcome": "H"})
        after = pd.concat(
            [
                before,
                make_matches({"date": "2024-08-17", "home_club": "arsenal",
                              "away_club": "wolves", "outcome": "H"}),
            ],
            ignore_index=True,
        )

        assert Elo().predict(round_fixtures, _evidence(before)) == pytest.approx(
            Elo().predict(round_fixtures, _evidence(after))
        )

    def test_predicting_twice_from_one_evidence_gives_one_answer(
        self, round_fixtures: pd.DataFrame, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """A rating pool is state, and a Predictor that folded the same matches in twice would
        drift a little further from the truth at every round while still looking plausible."""
        played = make_matches({"date": "2024-08-13", "outcome": "H"})
        elo = Elo()

        first = elo.predict(round_fixtures, _evidence(played))
        second = elo.predict(round_fixtures, _evidence(played))

        assert first == pytest.approx(second)

    def test_two_predictors_over_one_corpus_agree(
        self, round_fixtures: pd.DataFrame, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """Elo holds no state between rounds — every Prediction is built from the Evidence handed
        over and from nothing carried in — so a fresh instance and a used one say the same thing."""
        played = make_matches({"date": "2024-08-13", "outcome": "H"})
        used = Elo()
        used.predict(round_fixtures, _evidence(make_matches({"date": "2024-08-01"})))

        assert used.predict(round_fixtures, _evidence(played)) == pytest.approx(
            Elo().predict(round_fixtures, _evidence(played))
        )


class TestItsFittedParameters:
    def test_it_uses_the_frozen_settings_by_default(self) -> None:
        """Fitted in the Burn-In Window and frozen there (ADR 0008). Nothing fits at predict time
        and nothing fits at import."""
        assert Elo().settings == FROZEN_SETTINGS
        assert Elo().logit == FROZEN_LOGIT

    def test_the_frozen_home_advantage_favours_the_home_club(self) -> None:
        assert FROZEN_SETTINGS.home_advantage > 0

    def test_the_frozen_band_is_centred_on_zero(self) -> None:
        """So that an even contest is called even. See ``epl.models.burn_in.fit_logit``."""
        lower, upper = FROZEN_LOGIT.cutpoints
        assert lower == pytest.approx(-upper)

    def test_a_different_elo_can_be_built_for_comparison(
        self, round_fixtures: pd.DataFrame, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """The parameters are injectable so that ``python -m epl.models fit`` can score what it
        found against what is frozen — but only the frozen one is registered."""
        sluggish = Elo(settings=Settings(k=1.0, home_advantage=0.0), name="sluggish_elo")

        assert sluggish.name == "sluggish_elo"
        assert sluggish.predict(round_fixtures, _evidence(make_matches())).shape == (2, 3)


class TestItIsOnTheScoreboard:
    def test_it_is_registered_under_its_slug(self) -> None:
        assert predictors.by_name("elo") is ELO

    def test_it_satisfies_the_predictor_contract(self) -> None:
        assert isinstance(ELO, predictors.Predictor)

    def test_it_covers_every_fixture(self, round_fixtures: pd.DataFrame) -> None:
        """Unlike the Ceiling Line and the Pundits, Elo has something to say about every Fixture
        in the Evaluation Window — its input is results, and results go back to 2000/01."""
        assert schema.covered(ELO, round_fixtures).all()

    def test_it_claims_no_privileged_fixture_columns(self) -> None:
        """Only the Ceiling Line does (ADR 0001)."""
        assert predictors.also_sees(ELO) == ()

    def test_it_produces_ledger_rows_that_audit_clean(
        self, round_fixtures: pd.DataFrame, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """The end of the contract: what Elo says becomes a stored Prediction with a receipt."""
        played = make_matches({"date": "2024-08-13", "division": "E1", "outcome": "H"})

        rows = schema.predictions_for(ELO, round_fixtures, _evidence(played))

        assert schema.audit(rows) == []
        assert list(rows["predictor"]) == ["elo", "elo"]
        assert list(rows["inputs_seen"]) == [1, 1]
        assert list(rows["latest_input"]) == [pd.Timestamp("2024-08-13")] * 2

    def test_it_predicts_a_fixture_it_is_shown_nothing_about(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """A Predictor owes a Prediction for every Fixture it is handed, and a Club nobody has
        seen is a Club at the starting rating rather than a gap in the ledger."""
        fixtures = make_matches({"date": "2024-08-17", "home_club": "barrow",
                                 "away_club": "morecambe"})

        predicted = ELO.predict(fixtures, _evidence(make_matches()))

        assert np.isfinite(predicted).all()
