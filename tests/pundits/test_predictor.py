"""The Pundits as Predictors: a published Scoreline scored as-stated."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from epl import predictors
from epl.ledger import backtest, schema
from epl.predictors import Evidence
from epl.pundits import predictor
from epl.pundits.predictor import Pundit


@pytest.fixture
def a_pundit(make_calls) -> Callable[..., Pundit]:
    """A Pundit holding exactly the calls a test names, rather than the frozen dataset."""

    def _make(*rows: dict[str, object], name: str = "lawrenson") -> Pundit:
        return Pundit(name, "Mark Lawrenson", note="a caveat", calls=make_calls(*rows))

    return _make


def fixtures(*rows: dict[str, object]) -> pd.DataFrame:
    """A round's Fixtures as the ledger hands them over. Empty still carries the columns, because
    a round with nothing in it is still a frame of Fixtures."""
    defaults: dict[str, object] = {
        "season": 2017,
        "division": "E0",
        "date": pd.Timestamp("2017-08-12"),
        "home_club": "arsenal",
        "away_club": "chelsea",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows], columns=list(defaults))


@pytest.fixture
def evidence() -> Evidence:
    return Evidence.before(pd.DataFrame(columns=["season", "division", "date", "time"]),
                           pd.Timestamp("2017-08-11"))


class TestAsStated:
    @pytest.mark.parametrize(
        ("home_goals", "away_goals", "expected"),
        [(2, 0, [1.0, 0.0, 0.0]), (1, 1, [0.0, 1.0, 0.0]), (0, 3, [0.0, 0.0, 1.0])],
    )
    def test_a_scoreline_becomes_the_certainty_it_literally_claims(
        self, home_goals: int, away_goals: int, expected: list[float], evidence: Evidence, a_pundit
    ) -> None:
        """"Taking it at face value as [1, 0, 0] ... punishes a claim of certainty the Pundit never
        made" (ADR 0003). Scoring it anyway, and publishing what it costs, is the point."""
        pundit = a_pundit({"pred_home_goals": home_goals, "pred_away_goals": away_goals})

        assert pundit.predict(fixtures({}), evidence).tolist() == [expected]

    def test_the_predictions_come_back_in_the_order_the_fixtures_were_asked_in(
        self, evidence: Evidence, a_pundit
    ) -> None:
        pundit = a_pundit(
            {"away_club": "chelsea", "pred_home_goals": 2, "pred_away_goals": 0},
            {"away_club": "burnley", "pred_home_goals": 0, "pred_away_goals": 1},
        )

        predicted = pundit.predict(
            fixtures({"away_club": "burnley"}, {"away_club": "chelsea"}), evidence
        )

        assert predicted.tolist() == [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]

    def test_an_empty_round_comes_back_as_a_frame_of_no_predictions(
        self, evidence: Evidence, a_pundit
    ) -> None:
        """`(0, 3)`, not `(0,)`. The backfill filters by `covers` and never asks for one, but the
        live loop at issue #17 has no such guarantee and would fail somewhere further away."""
        predicted = a_pundit({}).predict(fixtures(), evidence)

        assert predicted.shape == (0, 3)

    def test_it_reads_no_history_because_it_has_none_to_read(
        self, evidence: Evidence, a_pundit
    ) -> None:
        """A Pundit's call is a fact about the Fixture, published before it. There is no corpus
        behind it, so `inputs_seen = 0` is correct rather than an oversight — the Market Line
        records the same (CLAUDE.md)."""
        a_pundit({}).predict(fixtures({}), evidence)

        assert evidence.rows_seen == 0
        assert evidence.latest_seen is None


class TestWhatAPunditSpeaksTo:
    def test_it_covers_the_fixtures_it_published_a_call_on(
        self, evidence: Evidence, a_pundit
    ) -> None:
        pundit = a_pundit({"away_club": "chelsea"})

        covers = pundit.covers(fixtures({"away_club": "chelsea"}, {"away_club": "burnley"}))

        assert covers.tolist() == [True, False]

    def test_a_season_the_pundit_never_worked_is_not_covered(
        self, evidence: Evidence, a_pundit
    ) -> None:
        """"A Pundit published in the Seasons they worked and no others" — Lawrenson stopped after
        2021/22, and a Prediction invented for 2023/24 would score."""
        pundit = a_pundit({"season": 2017})

        assert pundit.covers(fixtures({"season": 2023})).tolist() == [False]

    def test_the_other_direction_of_a_pairing_is_a_different_fixture(
        self, evidence: Evidence, a_pundit
    ) -> None:
        pundit = a_pundit({"home_club": "arsenal", "away_club": "chelsea"})

        reversed_pairing = fixtures({"home_club": "chelsea", "away_club": "arsenal"})

        assert pundit.covers(reversed_pairing).tolist() == [False]

    def test_being_asked_to_predict_a_fixture_it_does_not_cover_is_an_error(
        self, evidence: Evidence, a_pundit
    ) -> None:
        """Falling back to a base rate would put a Prediction the Pundit never made onto their
        track record, and it would look exactly like one they did."""
        pundit = a_pundit({})

        with pytest.raises(predictor.PunditError, match="everton v chelsea"):
            pundit.predict(fixtures({"home_club": "everton"}), evidence)


class TestTheRegisteredPundits:
    def test_two_named_forecasters_are_registered_not_one_anonymous_slot(self) -> None:
        """"A Pundit is a *named* public forecaster" (CONTEXT.md), and two people worked these
        nine Seasons. One combined line would be a Predictor that is nobody."""
        assert {predictor.LAWRENSON.name, predictor.SUTTON.name} <= set(
            p.name for p in predictors.registered()
        )

    @pytest.mark.parametrize("pundit", ["lawrenson", "sutton"])
    def test_each_carries_the_caveat_that_has_to_travel_with_its_score(self, pundit: str) -> None:
        """An as-stated RPS beside the market's reads as a verdict on the pundit. It is mostly a
        verdict on the format of the question (ADR 0003), and the scoreboard has to say so."""
        note = predictors.note(pundit)

        assert "as-stated" in note
        assert "0003" in note

    @pytest.mark.parametrize("pundit", ["lawrenson", "sutton"])
    def test_the_note_attributes_the_bbc_as_the_origin(self, pundit: str) -> None:
        """Issue #11: "BBC is attributed as the origin". The scoreboard line is the artifact a
        reader actually receives, so the attribution has to reach it and not only a docstring."""
        note = predictors.note(pundit)

        assert "BBC" in note
        assert "MyFootballFacts" in note

    @pytest.mark.parametrize("pundit", ["lawrenson", "sutton"])
    def test_the_note_says_the_number_is_not_comparable_to_a_full_window_one(
        self, pundit: str
    ) -> None:
        """A Pundit is scored over the Fixtures they called — 1,896 and 1,512 against the board's
        7,980 — so the bare RPS invites the comparison the Ceiling Line's note exists to refuse
        (ADR 0001)."""
        assert "not comparable to a full-window" in predictors.note(pundit)

    @pytest.mark.parametrize(
        ("pundit", "span"), [("lawrenson", "2017/18-2021/22"), ("sutton", "2022/23-2025/26")]
    )
    def test_the_note_names_the_seasons_they_worked_read_off_the_pages(
        self, pundit: str, span: str
    ) -> None:
        """Derived from `myfootballfacts.PAGES` rather than retyped, so it cannot drift from the
        Seasons actually archived."""
        assert span in predictors.note(pundit)

    def test_each_reads_its_own_calls_out_of_the_frozen_dataset(self) -> None:
        assert set(predictor.LAWRENSON.calls["season"]) == set(range(2017, 2022))
        assert set(predictor.SUTTON.calls["season"]) == set(range(2022, 2026))

    def test_neither_claims_a_fixture_column_the_ledger_withholds(self) -> None:
        assert predictors.also_sees(predictor.LAWRENSON) == ()
        assert predictors.also_sees(predictor.SUTTON) == ()


class TestThroughTheLedger:
    def test_the_walk_stores_only_the_fixtures_the_pundit_covered(
        self, make_matches, a_pundit
    ) -> None:
        matches = make_matches(
            {"season": 2017, "date": "2017-08-12", "home_club": "arsenal", "away_club": "chelsea"},
            {"season": 2017, "date": "2017-08-12", "home_club": "everton", "away_club": "burnley"},
        )
        pundit = a_pundit({"home_club": "arsenal", "away_club": "chelsea"})

        rows = backtest.backfill(pundit, matches, seasons=[2017])

        assert list(rows["home_club"]) == ["arsenal"]
        assert schema.audit(rows) == []

    def test_a_stored_row_records_that_it_saw_no_input(self, make_matches, a_pundit) -> None:
        matches = make_matches({"season": 2017, "date": "2017-08-12"})

        rows = backtest.backfill(a_pundit({}), matches, seasons=[2017])

        assert list(rows["inputs_seen"]) == [0]
        assert rows["latest_input"].isna().all()

    def test_a_pundit_covering_nothing_in_the_window_writes_nothing(
        self, make_matches, a_pundit
    ) -> None:
        matches = make_matches({"season": 2023, "date": "2023-08-12"})

        rows = backtest.backfill(a_pundit({"season": 2017}), matches, seasons=[2023])

        assert rows.empty

    def test_an_as_stated_prediction_is_one_hot_and_still_a_valid_prediction(
        self, make_matches, a_pundit
    ) -> None:
        matches = make_matches({"season": 2017, "date": "2017-08-12"})

        rows = backtest.backfill(a_pundit({}), matches, seasons=[2017])

        assert np.allclose(rows[["prob_home", "prob_draw", "prob_away"]].sum(axis=1), 1.0)
