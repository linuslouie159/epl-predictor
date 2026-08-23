"""The Calibrated Pundit: a Pundit's own margin map, fitted walk-forward and quoted as a Predictor.

Two things these tests exist to hold. That the map is fitted on **past** calls only, through
:class:`epl.predictors.Evidence` like every other Predictor's history — which is what makes the
walk auditable off the stored rows rather than asserted here. And that nothing about it reads as a
person: it is a one-feature model built out of somebody's calls (ADR 0003).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from epl.ledger import backtest, schema
from epl.predictors import Corpus, Evidence
from epl.pundits import calibrated, margin
from epl.pundits.calibrated import CalibratedPundit
from epl.pundits.predictor import Pundit, PunditError

#: Enough calls to clear a minimum of four without every test having to spell out forty.
MINIMUM = 4


def a_season(calls: list[tuple[str, int, int, str]], season: int = 2017) -> pd.DataFrame:
    """Calls as ``(away_club, pred_home_goals, pred_away_goals, outcome)``, one Fixture a week.

    Arsenal are always at home and the away Club is what identifies the Fixture, so a test names
    the call and what happened and nothing else. Dates run forward a week at a time so that "past"
    is unambiguous.
    """
    start = pd.Timestamp("2017-08-12")
    return pd.DataFrame(
        [
            {
                "pundit": "lawrenson",
                "season": season,
                "division": "E0",
                "date": (start + pd.Timedelta(weeks=week)).date(),
                "home_club": "arsenal",
                "away_club": away,
                "pred_home_goals": home_goals,
                "pred_away_goals": away_goals,
                "outcome": outcome,
            }
            for week, (away, home_goals, away_goals, outcome) in enumerate(calls)
        ]
    )


def as_matches(calls: pd.DataFrame) -> pd.DataFrame:
    """The same Fixtures as the corpus holds them — played, with an Outcome and no call."""
    goals = {"H": (1, 0), "D": (1, 1), "A": (0, 1)}
    return calls.assign(
        time=pd.NA,
        home_goals=[goals[outcome][0] for outcome in calls["outcome"]],
        away_goals=[goals[outcome][1] for outcome in calls["outcome"]],
    )[
        [
            "season",
            "division",
            "date",
            "time",
            "home_club",
            "away_club",
            "home_goals",
            "away_goals",
            "outcome",
        ]
    ]


def fixtures(calls: pd.DataFrame, *away_clubs: str) -> pd.DataFrame:
    """Some of those Fixtures as the ledger hands them over — no Outcome, no call."""
    wanted = calls.loc[calls["away_club"].isin(away_clubs)]
    return wanted[["season", "division", "date", "home_club", "away_club"]].reset_index(drop=True)


@pytest.fixture
def make_calibrated() -> Callable[..., CalibratedPundit]:
    """A Calibrated Pundit over exactly the calls a test names."""

    def _make(calls: pd.DataFrame, *, minimum: int = MINIMUM) -> CalibratedPundit:
        pundit = Pundit("lawrenson", "Mark Lawrenson", calls=calls.drop(columns=["outcome"]))
        return CalibratedPundit(
            pundit,
            name="margin_map_lawrenson",
            display_name="Margin Map (Lawrenson's calls)",
            note="a caveat",
            minimum=minimum,
        )

    return _make


def evidence_at(matches: pd.DataFrame, instant: str) -> Evidence:
    return Evidence.before(Corpus(matches), pd.Timestamp(instant))


class TestQuotingTheMapRatherThanTheCall:
    def test_a_call_is_quoted_what_that_margin_has_produced_not_what_it_claims(
        self, make_calibrated
    ) -> None:
        """The Pundit says 2-0 and is scored `[1, 0, 0]`; this says what a 2-0 from them is worth.
        Four past 2-0 calls went H, H, D, A, so the fifth is quoted 0.5 / 0.25 / 0.25."""
        calls = a_season(
            [
                ("chelsea", 2, 0, "H"),
                ("burnley", 2, 0, "H"),
                ("everton", 2, 0, "D"),
                ("fulham", 2, 0, "A"),
                ("leeds", 2, 0, "H"),
            ]
        )
        predictor = make_calibrated(calls)

        quoted = predictor.predict(
            fixtures(calls, "leeds"), evidence_at(as_matches(calls), "2017-09-08")
        )

        assert quoted.tolist() == [[0.5, 0.25, 0.25]]

    def test_two_calls_of_different_margins_are_quoted_differently(
        self, make_calibrated
    ) -> None:
        """The point of the whole module: a 3-0 is a stronger claim than a 1-0 and is priced as
        one, where the as-stated reading and the shared isotonic layer both see only "Home"."""
        calls = a_season(
            [
                ("chelsea", 1, 0, "A"),
                ("burnley", 1, 0, "D"),
                ("everton", 1, 0, "H"),
                ("fulham", 1, 0, "A"),
                ("leeds", 3, 0, "H"),
                ("wolves", 3, 0, "H"),
                ("brentford", 3, 0, "H"),
                ("brighton", 3, 0, "H"),
                ("newcastle", 1, 0, "H"),
                ("liverpool", 3, 0, "H"),
            ]
        )
        predictor = make_calibrated(calls)

        weak, strong = predictor.predict(
            fixtures(calls, "newcastle", "liverpool"),
            evidence_at(as_matches(calls), "2017-10-06"),
        )

        assert weak.tolist() == [0.25, 0.25, 0.5]
        assert strong.tolist() == [1.0, 0.0, 0.0]

    def test_the_quotes_come_back_in_the_order_the_fixtures_were_asked_in(
        self, make_calibrated
    ) -> None:
        calls = a_season(
            [
                ("chelsea", 0, 1, "A"),
                ("burnley", 0, 1, "A"),
                ("everton", 0, 1, "A"),
                ("fulham", 0, 1, "A"),
                ("leeds", 2, 0, "H"),
                ("wolves", 2, 0, "H"),
                ("brentford", 2, 0, "H"),
                ("brighton", 2, 0, "H"),
                ("newcastle", 2, 0, "H"),
                ("liverpool", 0, 1, "A"),
            ]
        )
        predictor = make_calibrated(calls)
        asked = fixtures(calls, "liverpool", "newcastle").iloc[::-1].reset_index(drop=True)

        quoted = predictor.predict(asked, evidence_at(as_matches(calls), "2017-10-06"))

        assert asked["away_club"].tolist() == ["liverpool", "newcastle"]
        assert quoted.tolist() == [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]

    def test_an_empty_round_still_comes_back_as_three_columns(self, make_calibrated) -> None:
        calls = a_season([(name, 2, 0, "H") for name in ("a", "b", "c", "d")])
        predictor = make_calibrated(calls)

        quoted = predictor.predict(
            fixtures(calls), evidence_at(as_matches(calls), "2017-09-08")
        )

        assert quoted.shape == (0, 3)


class TestFittedOnPastCallsOnly:
    def test_the_fixture_being_predicted_never_reaches_its_own_map(
        self, make_calibrated
    ) -> None:
        """Issue #12's second acceptance criterion. The fifth call is quoted from four, and if it
        had seen itself the Away it turns into would be in the rate it is quoted."""
        calls = a_season(
            [
                ("chelsea", 2, 0, "H"),
                ("burnley", 2, 0, "H"),
                ("everton", 2, 0, "H"),
                ("fulham", 2, 0, "H"),
                ("leeds", 2, 0, "A"),
            ]
        )
        predictor = make_calibrated(calls)

        quoted = predictor.predict(
            fixtures(calls, "leeds"), evidence_at(as_matches(calls), "2017-09-08")
        )

        assert quoted.tolist() == [[1.0, 0.0, 0.0]]

    def test_the_map_grows_as_the_record_does(self, make_calibrated) -> None:
        """Same Fixture, same call, two instants — the later one has more behind it and says
        something different. A map fitted once over the whole record could not do this."""
        calls = a_season(
            [
                ("chelsea", 2, 0, "H"),
                ("burnley", 2, 0, "H"),
                ("everton", 2, 0, "H"),
                ("fulham", 2, 0, "H"),
                ("leeds", 2, 0, "A"),
                ("wolves", 2, 0, "A"),
            ]
        )
        predictor = make_calibrated(calls)
        matches = as_matches(calls)
        asked = fixtures(calls, "wolves")

        early = predictor.predict(asked, evidence_at(matches, "2017-09-08"))
        late = predictor.predict(asked, evidence_at(matches, "2017-09-15"))

        assert early.tolist() == [[1.0, 0.0, 0.0]]
        assert late.tolist() == [[0.8, 0.0, 0.2]]

    def test_the_map_is_read_through_evidence_so_the_ledger_records_what_it_saw(
        self, make_calibrated
    ) -> None:
        """Unlike a Pundit, which consumes no history and records `inputs_seen = 0`, this reads
        results — so its rows carry a receipt the leak audit can check off the file."""
        calls = a_season([(name, 2, 0, "H") for name in ("a", "b", "c", "d", "e")])
        predictor = make_calibrated(calls)
        evidence = evidence_at(as_matches(calls), "2017-09-08")

        rows = schema.predictions_for(predictor, fixtures(calls, "e"), evidence)

        assert rows["inputs_seen"].tolist() == [4]
        assert rows["latest_input"].iloc[0] < rows["as_of_instant"].iloc[0]

    def test_a_backfill_over_the_window_audits_clean(self, make_calibrated) -> None:
        """The end-to-end guard. Every round's map is fitted on Evidence cut at that round's own
        As-Of Instant, so no stored row may see an input at or after it."""
        calls = a_season(
            [(f"club_{index}", 2, 0, "H" if index % 3 else "A") for index in range(12)],
            season=2017,
        )
        predictor = make_calibrated(calls)

        rows = backtest.backfill(predictor, as_matches(calls), seasons=[2017])

        assert schema.audit(rows) == []
        assert len(rows) == 12 - MINIMUM


class TestWhatItWillNotSpeakTo:
    def test_it_covers_nothing_until_the_map_has_its_minimum(self, make_calibrated) -> None:
        """The first calls of a record have nothing behind them. Quoting them anyway would mean
        a rate read off a handful of matches, so they are not covered at all — and the scoreboard's
        `fixtures` column is where that cost shows up."""
        calls = a_season([(name, 2, 0, "H") for name in ("a", "b", "c", "d", "e", "f")])
        predictor = make_calibrated(calls)

        covered = predictor.covers(fixtures(calls, "a", "b", "c", "d", "e", "f"))

        assert covered.tolist() == [False] * MINIMUM + [True, True]

    def test_it_covers_nothing_the_pundit_never_called(self, make_calibrated) -> None:
        """A model built out of somebody's calls has nothing to say where they made none."""
        calls = a_season([(name, 2, 0, "H") for name in ("a", "b", "c", "d", "e")])
        predictor = make_calibrated(calls)
        uncalled = fixtures(calls, "e").assign(away_club="nobody")

        assert predictor.covers(uncalled).tolist() == [False]

    def test_asking_it_about_an_uncovered_fixture_is_an_error(self, make_calibrated) -> None:
        """An uncovered Fixture is an error, not a fallback — the same rule as the Pundit's, for
        the same reason: a made-up Prediction that scores is worse than an absent one."""
        calls = a_season([(name, 2, 0, "H") for name in ("a", "b", "c", "d", "e")])
        predictor = make_calibrated(calls)
        uncalled = fixtures(calls, "e").assign(away_club="nobody")

        with pytest.raises(PunditError, match="no call"):
            predictor.predict(uncalled, evidence_at(as_matches(calls), "2017-09-08"))

    def test_a_round_before_the_minimum_is_refused_rather_than_guessed(
        self, make_calibrated
    ) -> None:
        """`covers` keeps these off the walk; asked anyway, the map refuses rather than quoting a
        rate it has no sample for."""
        calls = a_season([(name, 2, 0, "H") for name in ("a", "b", "c", "d", "e")])
        predictor = make_calibrated(calls)

        with pytest.raises(margin.MarginMapError, match="past calls"):
            predictor.predict(
                fixtures(calls, "c"), evidence_at(as_matches(calls), "2017-08-25")
            )


class TestTheMapAsADiagnostic:
    def test_the_map_at_an_instant_can_be_asked_for_and_published(
        self, make_calibrated
    ) -> None:
        """Elo exposes `ratings_at` for the same reason: the model's own view of history is the
        only way to ask a question about it, and asking through Evidence keeps even a diagnostic
        on the right side of the project's one rule."""
        calls = a_season([(name, 2, 0, "H") for name in ("a", "b", "c", "d", "e")])
        predictor = make_calibrated(calls)

        fitted = predictor.map_at(evidence_at(as_matches(calls), "2017-09-08"))

        assert fitted.calls == MINIMUM
        assert fitted.table()["margins"].tolist() == ["2", margin.POOLED]


class TestNamedAsAModelRatherThanAPerson:
    def test_two_calibrated_pundits_are_registered_beside_the_two_pundits(self) -> None:
        assert calibrated.MARGIN_MAP_LAWRENSON.name == "margin_map_lawrenson"
        assert calibrated.MARGIN_MAP_SUTTON.name == "margin_map_sutton"

    @pytest.mark.parametrize(
        "predictor", [calibrated.MARGIN_MAP_LAWRENSON, calibrated.MARGIN_MAP_SUTTON]
    )
    def test_the_name_on_the_scoreboard_is_the_map_not_the_forecaster(
        self, predictor: CalibratedPundit
    ) -> None:
        """"It must never be presented as 'Sutton beat the model'" (ADR 0003). The row a reader
        receives is named for the map, so the sentence has no subject to hang on."""
        assert predictor.name.startswith("margin_map_")
        assert predictor.name != predictor.pundit.name

    @pytest.mark.parametrize(
        "predictor", [calibrated.MARGIN_MAP_LAWRENSON, calibrated.MARGIN_MAP_SUTTON]
    )
    def test_the_note_says_it_is_a_model_and_not_the_person(
        self, predictor: CalibratedPundit
    ) -> None:
        """The note is what travels with the number wherever it is reported, so the distinction
        ADR 0003 spends its Consequences section on has to be in it rather than in a docstring."""
        assert "one-feature model" in predictor.note
        assert "not " + predictor.pundit.display_name in predictor.note
        assert "ADR 0003" in predictor.note

    @pytest.mark.parametrize(
        "predictor", [calibrated.MARGIN_MAP_LAWRENSON, calibrated.MARGIN_MAP_SUTTON]
    )
    def test_the_note_says_the_slate_is_not_the_boards(
        self, predictor: CalibratedPundit
    ) -> None:
        """The Ceiling Line's lesson (ADR 0001), applied a third time: an RPS over 1,800 Fixtures
        does not compare to one over 7,980, and the bare number reads as though it does."""
        assert "not comparable" in predictor.note

    def test_it_is_the_shared_calibration_layer_that_it_is_not(self) -> None:
        """A Calibrated Pundit and `epl.calibration` are different maps answering different
        questions, and CLAUDE.md asks that they never be collapsed into one."""
        assert calibrated.MARGIN_MAP_LAWRENSON.pundit is not None
        assert np.isclose(
            calibrated.MARGIN_MAP_LAWRENSON.minimum, margin.MINIMUM_SAMPLE
        )
