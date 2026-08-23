"""Grading a published Scoreline two ways — the strict reading and the lenient one."""

from __future__ import annotations

import pandas as pd

from epl.pundits import grading


def played(*rows: dict[str, object], make_matches) -> pd.DataFrame:
    defaults: dict[str, object] = {
        "season": 2017,
        "division": "E0",
        "date": "2017-08-12",
        "home_club": "arsenal",
        "away_club": "chelsea",
    }
    return make_matches(*[{**defaults, **row} for row in rows])


class TestGradingOneCall:
    def test_the_exact_score_and_the_outcome_are_graded_separately(
        self, make_matches, make_calls
    ) -> None:
        """A 2-0 call on a 1-0 win is wrong strictly and right leniently, and issue #11 asks for
        both readings rather than a choice between them."""
        (graded,) = grading.grade(
            make_calls({}), played({"home_goals": 1, "away_goals": 0, "outcome": "H"},
                              make_matches=make_matches)
        ).to_dict("records")

        assert not graded["exact_score"]
        assert graded["correct_outcome"]

    def test_calling_the_score_is_calling_the_outcome_too(self, make_matches, make_calls) -> None:
        (graded,) = grading.grade(
            make_calls({}), played({"home_goals": 2, "away_goals": 0, "outcome": "H"},
                              make_matches=make_matches)
        ).to_dict("records")

        assert graded["exact_score"]
        assert graded["correct_outcome"]

    def test_the_wrong_outcome_is_wrong_both_ways(self, make_matches, make_calls) -> None:
        (graded,) = grading.grade(
            make_calls({}), played({"home_goals": 0, "away_goals": 2, "outcome": "A"},
                              make_matches=make_matches)
        ).to_dict("records")

        assert not graded["exact_score"]
        assert not graded["correct_outcome"]

    def test_the_call_and_what_happened_both_survive_into_the_row(
        self, make_matches, make_calls
    ) -> None:
        """A grade with no scoreline beside it cannot be checked by a reader, and surfacing a
        pundit's best and worst calls needs the calls themselves (spec, user story 34)."""
        (graded,) = grading.grade(
            make_calls({}), played({"home_goals": 1, "away_goals": 1, "outcome": "D"},
                              make_matches=make_matches)
        ).to_dict("records")

        assert (graded["pred_home_goals"], graded["pred_away_goals"]) == (2, 0)
        assert (graded["home_goals"], graded["away_goals"]) == (1, 1)
        assert (graded["predicted_outcome"], graded["outcome"]) == ("H", "D")

    def test_a_call_on_an_unplayed_fixture_is_not_graded(self, make_matches, make_calls) -> None:
        """The current Season's remaining Fixtures have a call and no Outcome yet."""
        graded = grading.grade(make_calls({"home_club": "everton"}),
                               played(make_matches=make_matches))

        assert graded.empty


class TestTheReport:
    def test_it_counts_both_readings_per_pundit_and_season(self, make_matches, make_calls) -> None:
        graded = grading.grade(
            make_calls(
                {"away_club": "chelsea", "pred_home_goals": 1, "pred_away_goals": 0},
                {"away_club": "burnley", "pred_home_goals": 3, "pred_away_goals": 0},
            ),
            played(
                {"away_club": "chelsea", "home_goals": 1, "away_goals": 0, "outcome": "H"},
                {"away_club": "burnley", "home_goals": 2, "away_goals": 1, "outcome": "H"},
                make_matches=make_matches,
            ),
        )

        (row,) = grading.summary(graded).to_dict("records")

        assert row["calls"] == 2
        assert row["exact_scores"] == 1
        assert row["correct_outcomes"] == 2
        assert row["exact_rate"] == 0.5
        assert row["outcome_rate"] == 1.0

    def test_it_can_be_taken_across_a_whole_career(self, make_matches, make_calls) -> None:
        graded = grading.grade(
            make_calls({"season": 2017}, {"season": 2018, "away_club": "burnley"}),
            played(
                {"season": 2017, "home_goals": 2, "away_goals": 0, "outcome": "H"},
                {
                    "season": 2018,
                    "date": "2018-08-12",
                    "away_club": "burnley",
                    "home_goals": 2,
                    "away_goals": 0,
                    "outcome": "H",
                },
                make_matches=make_matches,
            ),
        )

        by_season = grading.summary(graded)
        career = grading.summary(graded, by=("pundit",))

        assert len(by_season) == 2
        assert list(career["calls"]) == [2]
