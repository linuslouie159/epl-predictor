"""The frozen Pundit dataset: resolving, choosing, locating and cross-checking the calls.

The parse is :mod:`epl.pundits.myfootballfacts`'s job; everything here is about turning what the
page said into facts about Fixtures that the corpus agrees exist.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from epl.clubs import UnknownAliasError
from epl.pundits import dataset


@pytest.fixture
def matches(make_matches):
    """Two 2017/18 Premier League Fixtures, played."""
    return make_matches(
        {
            "season": 2017,
            "date": "2017-08-12",
            "home_club": "arsenal",
            "away_club": "chelsea",
            "home_goals": 1,
            "away_goals": 0,
            "outcome": "H",
        },
        {
            "season": 2017,
            "date": "2017-08-19",
            "home_club": "everton",
            "away_club": "burnley",
            "home_goals": 0,
            "away_goals": 1,
            "outcome": "A",
        },
    )


class TestWhatACallMeans:
    @pytest.mark.parametrize(
        ("home_goals", "away_goals", "outcome"),
        [(2, 0, "H"), (1, 1, "D"), (0, 3, "A"), (0, 0, "D"), (4, 3, "H")],
    )
    def test_a_scoreline_implies_an_outcome(
        self, home_goals: int, away_goals: int, outcome: str
    ) -> None:
        """"A Scoreline implies an Outcome; an Outcome does not imply a Scoreline" (CONTEXT.md).

        It lives beside the calls rather than in the grading, because the as-stated Predictor
        needs it too and should not have to import a reporting module to read its own call."""
        implied = dataset.outcomes_of(pd.Series([home_goals]), pd.Series([away_goals]))

        assert list(implied) == [outcome]

    def test_a_fixture_is_the_club_pairing_inside_a_season(self, make_calls) -> None:
        """One key, spelled once. The grading joins on it, the Predictor looks calls up by it and
        the backfill report subtracts with it; three spellings would be three chances to differ."""
        keyed = make_calls({"season": 2017, "home_club": "arsenal", "away_club": "chelsea"})

        assert dataset.fixture_keys(keyed) == [(2017, "arsenal", "chelsea")]
        assert dataset.FIXTURE_KEY == ("season", "home_club", "away_club")


class TestResolvingClubs:
    def test_the_sources_spellings_become_slugs(self, matches, make_listings) -> None:
        built = dataset.build(matches, make_listings({}))

        assert list(built.calls["home_club"]) == ["arsenal"]
        assert list(built.calls["away_club"]) == ["chelsea"]

    def test_a_misspelling_the_alias_table_knows_resolves(
        self, matches, make_matches, make_listings
    ) -> None:
        """`Wolverhampton Wand` is the source misspelling a Club, so it is an Alias row."""
        fixture = make_matches(
            {"season": 2018, "home_club": "wolves", "away_club": "arsenal", "date": "2018-09-01"}
        )
        built = dataset.build(
            fixture,
            make_listings(
                {
                    "season": 2018,
                    "home_name": "Wolverhampton Wand",
                    "away_name": "Arsenal",
                    "published_home_goals": 1,
                    "published_away_goals": 0,
                }
            ),
        )

        assert list(built.calls["home_club"]) == ["wolves"]

    def test_a_spelling_the_alias_table_does_not_know_fails_loudly(
        self, matches, make_listings
    ) -> None:
        """An unmapped spelling would split one Club's calls in two, exactly as it would split a
        rating history. It is an error, never a new Club (spec, user story 9)."""
        with pytest.raises(UnknownAliasError, match="Woolwich Arsenal"):
            dataset.build(matches, make_listings({"home_name": "Woolwich Arsenal"}))


class TestChoosingBetweenTwoCallsOnOneFixture:
    def test_the_postponed_listing_gives_way_to_the_played_one(
        self, matches, make_listings
    ) -> None:
        """A Fixture postponed and replayed is listed twice, and the pundit called it twice — for
        2022/23's Leicester against Aston Villa, 1-2 first and 2-2 second. The call that stands is
        the one published for the date it was actually played; the other was for a date on which
        nothing happened."""
        built = dataset.build(
            matches,
            make_listings(
                {"pred_home_goals": 1, "pred_away_goals": 2, "played": False},
                {"pred_home_goals": 2, "pred_away_goals": 2},
            ),
        )

        assert len(built.calls) == 1
        assert list(built.calls["pred_home_goals"]) == [2]

    def test_the_order_the_page_lists_them_in_does_not_decide(self, matches, make_listings) -> None:
        """2020/21 lists the played one first and 2022/23 lists it second."""
        built = dataset.build(
            matches,
            make_listings(
                {"pred_home_goals": 2, "pred_away_goals": 2},
                {"pred_home_goals": 1, "pred_away_goals": 2, "played": False},
            ),
        )

        assert list(built.calls["pred_home_goals"]) == [2]

    def test_a_call_that_was_only_ever_postponed_is_still_the_call(
        self, matches, make_listings
    ) -> None:
        """Twice the page never re-lists the rearranged Fixture. The pundit's only call on it is
        the one they made, and dropping it would lose a Fixture they did speak to."""
        built = dataset.build(
            matches, make_listings({"pred_home_goals": 3, "pred_away_goals": 1, "played": False})
        )

        assert list(built.calls["pred_home_goals"]) == [3]

    def test_two_played_listings_of_one_fixture_are_refused(self, matches, make_listings) -> None:
        """Not observed in the nine pages, and it would mean the parse has doubled a table."""
        with pytest.raises(dataset.PunditDatasetError, match="arsenal v chelsea"):
            dataset.build(
                matches,
                make_listings({"pred_home_goals": 2}, {"pred_home_goals": 3}),
            )


class TestLocatingTheFixture:
    def test_the_date_comes_from_football_data_not_from_the_page(
        self, matches, make_listings
    ) -> None:
        """The matchday headings carry typos — 2024/25's opener is dated `16/08/25` — so the date
        stored is the corpus's, which is also the one the As-Of Instant is derived from."""
        built = dataset.build(matches, make_listings({}))

        assert list(built.calls["date"]) == [pd.Timestamp("2017-08-12").date()]

    def test_the_tier_is_recorded_so_the_fixture_key_is_complete(
        self, matches, make_listings
    ) -> None:
        built = dataset.build(matches, make_listings({}))

        assert list(built.calls["division"]) == ["E0"]

    def test_a_call_on_a_fixture_the_corpus_has_never_heard_of_is_refused(
        self, matches, make_listings
    ) -> None:
        """Both Clubs resolved and the Season is real, so this is a call placed on a Fixture that
        was never played — which means the two names went to the wrong slugs, or the wrong way
        round. Inventing the Fixture would put a scored Prediction on a match nobody played."""
        with pytest.raises(dataset.PunditDatasetError, match="chelsea v arsenal"):
            dataset.build(matches, make_listings({"home_name": "Chelsea", "away_name": "Arsenal"}))


class TestCrossCheckingAgainstFootballData:
    def test_agreeing_rows_are_not_reported(self, matches, make_listings) -> None:
        built = dataset.build(matches, make_listings({}))

        assert built.disagreements.empty

    def test_a_published_result_football_data_contradicts_is_reported(
        self, matches, make_listings
    ) -> None:
        """Four of the 3,406 published results are transcription slips. They are reported rather
        than raised on: the result is a check on the parse, and is never stored."""
        built = dataset.build(matches, make_listings({"published_home_goals": 4}))

        assert list(built.disagreements["home_club"]) == ["arsenal"]
        assert list(built.disagreements["published_home_goals"]) == [4]
        assert list(built.disagreements["home_goals"]) == [1]

    def test_a_postponed_listing_has_no_result_to_check(self, matches, make_listings) -> None:
        built = dataset.build(matches, make_listings({"played": False}))

        assert built.disagreements.empty
        assert built.checked == 0

    def test_a_fixture_football_data_has_no_score_for_is_not_checked_either(
        self, make_matches, make_listings
    ) -> None:
        """An abandoned match reaches the corpus with blank goals. There is nothing to compare, so
        it is neither an agreement nor a disagreement."""
        unscored = make_matches(
            {
                "season": 2017,
                "date": "2017-08-12",
                "home_club": "arsenal",
                "away_club": "chelsea",
                "home_goals": pd.NA,
                "away_goals": pd.NA,
                "outcome": pd.NA,
            }
        )

        built = dataset.build(unscored, make_listings({}))

        assert built.disagreements.empty
        assert built.checked == 0
        assert len(built.calls) == 1

    def test_a_season_that_mostly_disagrees_is_refused(self, matches, make_listings) -> None:
        """This is what a column shift or a swapped home and away looks like: not a slip in one
        row, but a Season that stops agreeing. A handful of typos never trips it."""
        with pytest.raises(dataset.PunditDatasetError, match="2017/18"):
            dataset.build(
                matches,
                make_listings(
                    {"published_home_goals": 7},
                    {
                        "home_name": "Everton",
                        "away_name": "Burnley",
                        "published_home_goals": 7,
                        "published_away_goals": 7,
                    },
                ),
                min_checked=2,
            )

    def test_too_few_checked_rows_to_tell_a_slip_from_a_shift_is_not_a_verdict(
        self, matches, make_listings
    ) -> None:
        """One disagreement out of one is 0% agreement and says nothing. The floor is a statement
        about a Season, and a Season is 380 Fixtures."""
        built = dataset.build(matches, make_listings({"published_home_goals": 4}))

        assert len(built.disagreements) == 1

    def test_the_agreement_floor_leaves_room_for_the_slips_that_are_really_there(self) -> None:
        """Four disagreements in 3,406 is 99.9%; a swapped home and away would agree only on the
        draws, about a quarter of the time."""
        assert 0.5 < dataset.MIN_AGREEMENT < 0.999


class TestTheFrozenFile:
    def test_only_the_facts_are_stored(self, matches, make_listings) -> None:
        """Fixture, predicted Scoreline, Predictor, date — and no prose, no matchday heading, and
        not the result, which is an Outcome and belongs nowhere near a stored Prediction."""
        built = dataset.build(matches, make_listings({}))

        assert list(built.calls.columns) == list(dataset.CALL_COLUMNS)
        assert "published_home_goals" not in built.calls.columns

    def test_it_round_trips_through_the_file(self, matches, tmp_path: Path, make_listings) -> None:
        built = dataset.build(matches, make_listings({}))
        path = tmp_path / "predictions.csv"

        dataset.write(built.calls, path)

        pd.testing.assert_frame_equal(dataset.read(path), built.calls)

    def test_rewriting_it_produces_the_same_bytes(
        self, matches, tmp_path: Path, make_listings
    ) -> None:
        """Frozen and committed: a rebuild that shuffled rows would show up as a diff and stop
        meaning anything."""
        built = dataset.build(matches, make_listings({}))

        first = dataset.write(built.calls, tmp_path / "a.csv").read_bytes()
        second = dataset.write(built.calls, tmp_path / "b.csv").read_bytes()

        assert first == second
