"""The rating pool: one Elo across all four tiers, folded in kickoff order.

ADR 0004 is what these are mostly about. A promoted Club must arrive in the Premier League with a
rating it earned in the Championship, a relegated Club must keep updating instead of freezing, and
a yo-yo Club must need no special case at all — all three of which are the same property, which is
that there is one pool and no per-tier bookkeeping anywhere.

Expected ratings are worked by hand from Elo's own arithmetic, with parameters chosen to make the
expected score a round number. The hyperparameters that are actually used are fitted in the Burn-In
Window (``tests/models/test_burn_in.py``) and measured over the corpus
(``tests/models/test_elo_over_the_corpus.py``); nothing here depends on them.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from epl.models import ModelError, Ratings, Settings

#: No home advantage and K = 20, so two Clubs at the same rating expect exactly half a point each
#: and a win moves them by exactly K/2.
EVEN = Settings(k=20.0, home_advantage=0.0)

#: 400 rating points is one factor of ten in Elo's own units, so the stronger Club expects exactly
#: 10/11 of a point. Used with a 400-point gap to keep the arithmetic exact.
DECADE = 400.0


class TestOneUpdate:
    def test_a_home_win_between_equals_moves_both_by_half_of_k(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """Both start at 1500 and expect 0.5, so the home Club gains 20 x (1 - 0.5) = 10."""
        ratings = Ratings.through(make_matches({"outcome": "H"}), EVEN)

        assert ratings.rating("arsenal") == pytest.approx(1510.0)
        assert ratings.rating("chelsea") == pytest.approx(1490.0)

    def test_an_away_win_between_equals_moves_them_the_other_way(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        ratings = Ratings.through(make_matches({"outcome": "A"}), EVEN)

        assert ratings.rating("arsenal") == pytest.approx(1490.0)
        assert ratings.rating("chelsea") == pytest.approx(1510.0)

    def test_a_draw_between_equals_moves_nobody(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """A Draw is exactly what two equal Clubs with no home advantage were expected to score."""
        ratings = Ratings.through(make_matches({"outcome": "D"}), EVEN)

        assert ratings.rating("arsenal") == pytest.approx(1500.0)
        assert ratings.rating("chelsea") == pytest.approx(1500.0)

    def test_the_favourite_gains_little_by_winning(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """400 points ahead is 10/11 of a point expected, so a win is worth only 20 x 1/11."""
        ratings = Ratings(EVEN, start={"arsenal": 1900.0, "chelsea": 1500.0})

        ratings.update(make_matches({"outcome": "H"}))

        assert ratings.rating("arsenal") == pytest.approx(1900.0 + 20 / 11)

    def test_a_draw_against_a_much_weaker_club_costs_the_favourite(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """The gap is handed to the pool as a settled rating rather than simulated, because
        reaching exactly 400 points by playing matches would take dozens of them.

        Expected 10/11, scored 1/2, K = 20: the favourite loses 20 x 9/22 = 8.1818 points.
        """
        ratings = Ratings(EVEN, start={"arsenal": 1900.0, "chelsea": 1500.0})

        ratings.update(make_matches({"outcome": "D"}))

        assert ratings.rating("arsenal") == pytest.approx(1900.0 - 90 / 11)
        assert ratings.rating("chelsea") == pytest.approx(1500.0 + 90 / 11)

    def test_home_advantage_is_worth_rating_points_in_the_expectation(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """Equal Clubs, but the home one is credited with 400 points for being at home, so it
        expects 10/11 and a home win is worth only 20 x 1/11."""
        ratings = Ratings.through(
            make_matches({"outcome": "H"}), Settings(k=20.0, home_advantage=DECADE)
        )

        assert ratings.rating("arsenal") == pytest.approx(1500.0 + 20 / 11)

    def test_a_bigger_k_moves_ratings_further(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        played = make_matches({"outcome": "H"})
        gentle = Ratings.through(played, Settings(k=10.0, home_advantage=0.0))
        brisk = Ratings.through(played, Settings(k=40.0, home_advantage=0.0))

        assert brisk.rating("arsenal") - 1500 == pytest.approx(
            4 * (gentle.rating("arsenal") - 1500)
        )


class TestThePoolAsAWhole:
    def test_rating_points_are_only_ever_moved_between_clubs(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """Elo is zero-sum, which is what makes ratings comparable across tiers connected only by
        promotion and relegation (ADR 0004): a Club can only be strong at another's expense."""
        ratings = Ratings.through(
            make_matches(
                {"date": "2024-08-17", "division": "E0", "outcome": "H"},
                {"date": "2024-08-17", "division": "E3", "home_club": "wolves",
                 "away_club": "fulham", "outcome": "A"},
                {"date": "2024-08-24", "division": "E1", "home_club": "arsenal",
                 "away_club": "fulham", "outcome": "D"},
            ),
            EVEN,
        )

        assert sum(ratings.rating(club) for club in ratings.clubs) == pytest.approx(4 * 1500.0)

    def test_a_club_keeps_one_rating_across_tiers(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """The point of rating the whole pyramid. Nothing here knows a tier changed."""
        ratings = Ratings.through(
            make_matches(
                {"date": "2024-08-17", "division": "E1", "outcome": "H"},
                {"date": "2024-08-24", "division": "E0", "outcome": "H"},
            ),
            EVEN,
        )

        assert ratings.rating("arsenal") > 1510.0
        assert ratings.played("arsenal") == 2

    def test_a_club_it_has_never_seen_starts_at_the_conventional_rating(self) -> None:
        """A Club promoted into League Two from outside the corpus starts cold, which ADR 0004
        accepts — it is four tiers from the Premier League."""
        ratings = Ratings(EVEN)

        assert ratings.rating("barrow") == pytest.approx(1500.0)
        assert ratings.played("barrow") == 0

    def test_it_folds_matches_in_kickoff_order_however_they_arrive(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """Elo is path-dependent: the same matches in a different order give different ratings.
        A frame that arrived sorted by anything else must not quietly produce a different pool."""
        rows = [
            {"date": "2024-08-17", "outcome": "H"},
            {"date": "2024-08-24", "home_club": "chelsea", "away_club": "wolves",
             "outcome": "H"},
            {"date": "2024-08-31", "outcome": "A"},
        ]
        forwards = Ratings.through(make_matches(*rows), EVEN)
        shuffled = Ratings.through(make_matches(*reversed(rows)), EVEN)

        assert shuffled.rating("arsenal") == pytest.approx(forwards.rating("arsenal"))
        assert shuffled.rating("chelsea") == pytest.approx(forwards.rating("chelsea"))

    def test_folding_in_two_goes_matches_folding_in_one(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """What the walk relies on: each Prediction Round hands over the matches since the last."""
        early = make_matches({"date": "2024-08-17", "outcome": "H"})
        late = make_matches({"date": "2024-08-24", "outcome": "A"})

        incremental = Ratings.through(early, EVEN)
        incremental.update(late)

        assert incremental.rating("arsenal") == pytest.approx(
            Ratings.through(pd.concat([early, late], ignore_index=True), EVEN).rating("arsenal")
        )

    def test_promoted_clubs_arrive_with_ratings_that_differ(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """Issue #9's acceptance criterion, and the whole reason for ADR 0004: any Premier
        League-only prior would give all three promoted Clubs the same rating on day one."""
        ratings = Ratings.through(
            make_matches(
                {"division": "E1", "home_club": "leeds", "away_club": "burnley", "outcome": "H"},
                {"division": "E1", "home_club": "sunderland", "away_club": "burnley",
                 "outcome": "D"},
            ),
            EVEN,
        )

        promoted = {club: ratings.rating(club) for club in ("leeds", "burnley", "sunderland")}

        assert len(set(promoted.values())) == 3
        assert all(rating != 1500.0 for rating in promoted.values())

    def test_nothing_played_leaves_every_rating_where_it_started(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        ratings = Ratings.through(make_matches(), EVEN)

        assert ratings.clubs == ()
        assert ratings.rating("arsenal") == pytest.approx(1500.0)


class TestReplayingHistory:
    """``walk`` is ``update`` that also says what the model thought on the way through.

    The Burn-In fit needs exactly that — the edge each match was played at, before its own result
    moved anything — and so does any question about how a rating was arrived at.
    """

    def test_it_reports_the_edge_each_match_was_played_at(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """The first match is between equals, so its edge is zero. The second is played by the
        winner of the first, 20 points clear of a Club that has not played."""
        played = make_matches(
            {"date": "2024-08-17", "outcome": "H"},
            {"date": "2024-08-24", "home_club": "arsenal", "away_club": "wolves",
             "outcome": "H"},
        )

        replayed = Ratings.through(make_matches(), EVEN).walk(played)

        assert list(replayed["edge"]) == pytest.approx([0.0, 10.0])

    def test_the_edge_is_what_the_model_knew_before_the_result(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """The project's one rule, inside the fit: a match must not be scored against a rating its
        own Outcome helped produce."""
        ratings = Ratings(EVEN)

        replayed = ratings.walk(make_matches({"outcome": "H"}))

        assert replayed["edge"].iloc[0] == pytest.approx(0.0)
        assert ratings.rating("arsenal") == pytest.approx(1510.0)

    def test_it_hands_the_matches_back_in_kickoff_order(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """So an edge and the match it belongs to cannot come apart."""
        replayed = Ratings(EVEN).walk(
            make_matches(
                {"date": "2024-08-24", "outcome": "H"},
                {"date": "2024-08-17", "outcome": "A"},
            )
        )

        assert list(pd.to_datetime(replayed["date"])) == [
            pd.Timestamp("2024-08-17"),
            pd.Timestamp("2024-08-24"),
        ]

    def test_it_leaves_the_frame_it_was_handed_alone(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        played = make_matches({"outcome": "H"})

        Ratings(EVEN).walk(played)

        assert "edge" not in played.columns

    def test_walking_nothing_still_answers_with_the_column(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        replayed = Ratings(EVEN).walk(make_matches())

        assert "edge" in replayed.columns
        assert len(replayed) == 0


class TestTheEdge:
    def test_the_edge_is_the_gap_plus_home_advantage(self) -> None:
        """One definition, used by the update and by the ordered logit alike — so the Supremacy a
        Prediction is built from is the same quantity the ratings were learned through."""
        ratings = Ratings(
            Settings(k=20.0, home_advantage=60.0), start={"arsenal": 1700.0, "chelsea": 1500.0}
        )

        assert ratings.edge("arsenal", "chelsea") == pytest.approx(260.0)
        assert ratings.edge("chelsea", "arsenal") == pytest.approx(-140.0)

    def test_it_answers_one_edge_per_fixture(self) -> None:
        ratings = Ratings(EVEN, start={"arsenal": 1700.0})
        edges = ratings.edges(["arsenal", "chelsea"], ["chelsea", "arsenal"])

        assert list(edges) == pytest.approx([200.0, -200.0])

    def test_no_fixtures_is_no_edges(self) -> None:
        assert len(Ratings(EVEN).edges([], [])) == 0


class TestItRefusesToBeBuiltWrong:
    def test_k_must_be_positive(self) -> None:
        """A K of zero is a pool that never learns anything and still reports ratings."""
        with pytest.raises(ModelError, match="K-factor"):
            Settings(k=0.0, home_advantage=0.0)

    def test_an_outcome_it_does_not_recognise_stops_the_fold(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """Skipping the row would be worse: the pool would quietly differ from the one the same
        corpus produced yesterday, and nothing downstream would look wrong."""
        with pytest.raises(ModelError, match="outcome"):
            Ratings.through(make_matches({"outcome": "X"}), EVEN)

    def test_an_unplayed_fixture_stops_the_fold(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """Evidence holds matches that have kicked off, so a missing Outcome here means the corpus
        was cut wrong rather than that a Fixture is pending."""
        with pytest.raises(ModelError, match="outcome"):
            Ratings.through(make_matches({"outcome": pd.NA}), EVEN)
