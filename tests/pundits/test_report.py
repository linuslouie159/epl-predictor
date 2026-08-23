"""The three-way board, the cost of stating certainty, and every call ranked by its miss.

The frames here are built directly rather than walked out of a backfill: what these functions do is
cut, join and label an already-scored table, and a test that had to run two Predictors over a
window first would be testing the ledger again.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from epl.ledger import scoreboard
from epl.pundits import report
from epl.pundits.calibrated import CalibratedPundit
from epl.pundits.predictor import Pundit

#: A scored row as :func:`epl.ledger.scoreboard.calibrated_predictions` hands it over.
SCORED_DEFAULTS: dict[str, object] = {
    "predictor": "elo",
    "prediction_round": "2017-08-11",
    "as_of_instant": pd.Timestamp("2017-08-11"),
    "season": 2017,
    "division": "E0",
    "kickoff": pd.Timestamp("2017-08-12"),
    "home_club": "arsenal",
    "away_club": "chelsea",
    "prob_home": 0.5,
    "prob_draw": 0.3,
    "prob_away": 0.2,
    "inputs_seen": 10,
    "latest_input": pd.Timestamp("2017-08-10"),
    "outcome": "H",
    "calibrated_prob_home": 0.5,
    "calibrated_prob_draw": 0.3,
    "calibrated_prob_away": 0.2,
    "corrected": True,
    "correction": 0.0,
}


@pytest.fixture
def make_scored() -> Callable[..., pd.DataFrame]:
    """Scored rows, naming only the fields the test is about."""

    def _make(*rows: dict[str, object]) -> pd.DataFrame:
        return pd.DataFrame([{**SCORED_DEFAULTS, **row} for row in rows])

    return _make


def a_calibrated_pundit(calls: pd.DataFrame | None = None) -> CalibratedPundit:
    """A Calibrated Pundit named the way the registered ones are, over calls a test supplies."""
    pundit = Pundit("lawrenson", "Mark Lawrenson", calls=calls)
    return CalibratedPundit(
        pundit,
        name="margin_map_lawrenson",
        display_name="Margin Map (Lawrenson's calls)",
    )


def calls_for(*rows: tuple[str, int, int]) -> pd.DataFrame:
    """A frozen-dataset frame: ``(away_club, pred_home_goals, pred_away_goals)`` per call."""
    return pd.DataFrame(
        [
            {
                "pundit": "lawrenson",
                "season": 2017,
                "division": "E0",
                "date": pd.Timestamp("2017-08-12").date(),
                "home_club": "arsenal",
                "away_club": away,
                "pred_home_goals": home_goals,
                "pred_away_goals": away_goals,
            }
            for away, home_goals, away_goals in rows
        ]
    )


#: The five Predictors a three-way board carries, in the order the helper below emits them.
EVERYONE: tuple[str, ...] = (
    "elo",
    "market_line",
    "naive_baseline",
    "lawrenson",
    "margin_map_lawrenson",
)


def everyone_on(
    away_club: str, *, says: dict[str, dict[str, object]] | None = None, **shared: object
) -> list[dict[str, object]]:
    """One Fixture, called by all five Predictors a three-way board carries.

    ``shared`` overrides every row and ``says`` overrides one named Predictor's — which is how a
    test gives the Pundit the one-hot Prediction its whole point rests on without accidentally
    storing two Predictions for one Fixture.
    """
    named = says or {}
    return [
        {"predictor": name, "away_club": away_club, **shared, **named.get(name, {})}
        for name in EVERYONE
    ]


class TestTheSharedSlate:
    def test_only_the_fixtures_every_predictor_reached_survive(self, make_scored) -> None:
        """ADR 0001's lesson, applied a third time: an RPS over one Predictor's Fixtures and one
        over another's are not comparable, so the comparison is cut rather than the caveat added."""
        scored = make_scored(
            *everyone_on("chelsea"),
            *everyone_on("burnley")[:3],  # the two Pundit readings never covered this one
        )

        cut = report.shared(scored, ("elo", "lawrenson", "margin_map_lawrenson"))

        assert set(cut["away_club"]) == {"chelsea"}
        assert len(cut) == 3

    def test_a_predictor_nothing_was_stored_for_empties_the_slate(self, make_scored) -> None:
        """A three-way comparison missing one of its three is not a comparison to publish."""
        scored = make_scored(*everyone_on("chelsea"))

        assert report.shared(scored, ("elo", "dixon_coles")).empty

    def test_predictors_outside_the_comparison_are_dropped_too(self, make_scored) -> None:
        scored = make_scored(*everyone_on("chelsea"))

        cut = report.shared(scored, ("elo", "market_line"))

        assert sorted(cut["predictor"]) == ["elo", "market_line"]


class TestTheThreeWayBoard:
    def test_it_carries_the_model_the_market_the_floor_and_both_readings(
        self, make_scored
    ) -> None:
        """Issue #12's fourth acceptance criterion. The floor is on it because a Pundit as stated
        scores *below* it, which is the whole of ADR 0003's argument and invisible without it."""
        board = report.three_way(
            make_scored(*everyone_on("chelsea")), a_calibrated_pundit()
        )

        assert sorted(board["predictor"]) == [
            "elo",
            "lawrenson",
            "margin_map_lawrenson",
            "market_line",
            "naive_baseline",
        ]

    def test_every_metric_is_reported_twice(self, make_scored) -> None:
        """Pre-calibration and post-calibration, exactly as the main scoreboard does (ADR 0006) —
        because it *is* the main scoreboard's code, over a narrower slate."""
        board = report.three_way(
            make_scored(*everyone_on("chelsea")), a_calibrated_pundit()
        )

        assert list(board.columns) == list(report.THREE_WAY_COLUMNS)
        for metric in scoreboard.METRICS:
            assert metric in board.columns
            assert f"{scoreboard.CALIBRATED_PREFIX}{metric}" in board.columns

    def test_the_slate_is_named_on_every_row(self, make_scored) -> None:
        """Two boards are published in one file, so a row that did not say whose Fixtures it was
        measured over would be a number with no denominator."""
        board = report.three_way(
            make_scored(*everyone_on("chelsea")), a_calibrated_pundit()
        )

        assert board["slate"].unique().tolist() == ["lawrenson"]

    def test_all_five_are_scored_over_the_same_fixtures(self, make_scored) -> None:
        scored = make_scored(
            *everyone_on("chelsea"),
            *everyone_on("burnley")[:4],  # the margin map has no call on this one
        )

        board = report.three_way(scored, a_calibrated_pundit())

        assert board["fixtures"].unique().tolist() == [1]


class TestTheCostOfStatingCertainty:
    def test_the_gap_between_the_two_readings_is_named_rather_than_left_to_subtraction(
        self, make_scored
    ) -> None:
        """Issue #12's fifth acceptance criterion. A reader should not have to find two rows and
        work out what the difference between them means."""
        scored = make_scored(
            *everyone_on(
                "chelsea",
                says={
                    "lawrenson": {
                        "prob_home": 0.0,
                        "prob_draw": 0.0,
                        "prob_away": 1.0,
                        "calibrated_prob_home": 0.0,
                        "calibrated_prob_draw": 0.0,
                        "calibrated_prob_away": 1.0,
                    }
                },
            )
        )
        pundit = a_calibrated_pundit()

        (line,) = report.certainty(
            report.boards(scored, [pundit]), [pundit]
        ).to_dict("records")

        assert line["as_stated_rps"] == 1.0
        assert line["calibrated_rps"] == 0.145
        assert line["cost_of_certainty"] == pytest.approx(0.855)

    def test_it_names_both_readings_so_neither_can_be_published_alone(
        self, make_scored
    ) -> None:
        pundit = a_calibrated_pundit()

        (line,) = report.certainty(
            report.boards(make_scored(*everyone_on("chelsea")), [pundit]), [pundit]
        ).to_dict("records")

        assert line["as_stated"] == "lawrenson"
        assert line["calibrated"] == "margin_map_lawrenson"
        assert line["slate"] == "lawrenson"

    def test_accuracy_rides_along_beside_rps(self, make_scored) -> None:
        """The one metric on which the as-stated reading is not obviously unfair — it asks only who
        they picked. The pair says whether the map found information or merely padded a one-hot."""
        pundit = a_calibrated_pundit()

        (line,) = report.certainty(
            report.boards(make_scored(*everyone_on("chelsea")), [pundit]), [pundit]
        ).to_dict("records")

        assert "as_stated_accuracy" in line
        assert "calibrated_accuracy" in line

    def test_a_slate_with_no_pundit_rows_produces_no_line(self, make_scored) -> None:
        """Rather than a row of NaNs, which reads as a real measurement."""
        pundit = a_calibrated_pundit()
        scored = make_scored({"predictor": "elo"})

        assert report.certainty(report.boards(scored, [pundit]), [pundit]).empty


class TestCallsRankedByMiss:
    def test_the_best_calls_come_first_and_the_worst_last(self, make_scored) -> None:
        """Spec, user story 34. The miss is the RPS of the fair reading — what the call was still
        wrong by once the Scoreline had been read as what such a call is worth."""
        scored = make_scored(
            *everyone_on("chelsea", outcome="H"),
            *everyone_on("burnley", outcome="A"),
        )
        pundit = a_calibrated_pundit(calls_for(("chelsea", 2, 0), ("burnley", 2, 0)))

        ranked = report.calls_by_miss(scored, pundit, pundit.pundit.calls)

        # (0.5, 0.3, 0.2) cumulates to (0.5, 0.8); against Home that is
        # ((0.5-1)^2 + (0.8-1)^2)/2 = 0.145, and against Away (0.5^2 + 0.8^2)/2 = 0.445.
        assert ranked["away_club"].tolist() == ["chelsea", "burnley"]
        assert ranked["miss"].tolist() == pytest.approx([0.145, 0.445])

    def test_every_call_carries_its_scoreline_and_its_margin(self, make_scored) -> None:
        """A miss with no call beside it cannot be checked by a reader, and the margin is the one
        feature the map actually used."""
        scored = make_scored(*everyone_on("chelsea"))
        pundit = a_calibrated_pundit(calls_for(("chelsea", 3, 1)))

        (call,) = report.calls_by_miss(scored, pundit, pundit.pundit.calls).to_dict("records")

        assert (call["pred_home_goals"], call["pred_away_goals"]) == (3, 1)
        assert call["margin"] == 2

    def test_the_as_stated_reading_of_the_same_call_rides_along(self, make_scored) -> None:
        """So the list can be read the other way: the calls certainty cost the most are the bold
        ones that came off, where the as-stated reading scored zero and the fair one did not."""
        scored = make_scored(
            *everyone_on(
                "chelsea",
                outcome="H",
                says={"lawrenson": {"prob_home": 1.0, "prob_draw": 0.0, "prob_away": 0.0}},
            )
        )
        pundit = a_calibrated_pundit(calls_for(("chelsea", 2, 0)))

        (call,) = report.calls_by_miss(scored, pundit, pundit.pundit.calls).to_dict("records")

        assert call["as_stated_rps"] == 0.0
        assert call["miss"] == 0.145
        assert call["cost_of_certainty"] == pytest.approx(-0.145)

    def test_a_pundit_whose_map_reached_nothing_produces_no_rows(self, make_scored) -> None:
        pundit = a_calibrated_pundit(calls_for(("chelsea", 2, 0)))
        scored = make_scored({"predictor": "lawrenson", "away_club": "chelsea"})

        ranked = report.calls_by_miss(scored, pundit, pundit.pundit.calls)

        assert ranked.empty
        assert list(ranked.columns) == list(report.CALL_COLUMNS)

    def test_the_columns_are_the_published_ones(self, make_scored) -> None:
        scored = make_scored(*everyone_on("chelsea"))
        pundit = a_calibrated_pundit(calls_for(("chelsea", 2, 0)))

        ranked = report.calls_by_miss(scored, pundit, pundit.pundit.calls)

        assert list(ranked.columns) == list(report.CALL_COLUMNS)


class TestPublishingPlainFiles:
    def test_a_table_is_written_where_it_says_it_is(
        self, project_root, make_scored
    ) -> None:
        """Issue #12's seventh criterion: plain files under `outputs/`, no presentation logic, so a
        frontend can be built without any modelling logic leaking into it."""
        board = report.three_way(
            make_scored(*everyone_on("chelsea")), a_calibrated_pundit()
        )

        written = report.write(board, "three_way")

        assert written == project_root / "outputs" / "three_way.csv"
        assert list(pd.read_csv(written).columns) == list(report.THREE_WAY_COLUMNS)

    def test_writing_twice_writes_the_same_bytes(self, project_root, make_scored) -> None:
        """Regenerable has to mean identical, or a rebuild is indistinguishable from a change."""
        board = report.three_way(
            make_scored(*everyone_on("chelsea")), a_calibrated_pundit()
        )

        first = report.write(board, "three_way").read_bytes()
        second = report.write(board, "three_way").read_bytes()

        assert first == second
