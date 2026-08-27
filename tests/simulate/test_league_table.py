"""The final league table, and the chain that breaks a tie in it.

Issue #15's most testable acceptance criterion: "the full tiebreaker chain is implemented and
unit-tested — points, goal difference, goals scored, head-to-head points, head-to-head away goals,
with the neutral-ground play-off treated as a coin flip. Ties are routine — 24 of 26 Seasons had at
least one, averaging 3.3 tied pairs."

Every league here is hand-built and tiny, because what is being checked is the *order* of the
chain rather than arithmetic over a real Season. The sharpest tests are the ones where two steps
disagree: a league where goal difference says one thing and the head-to-head record says the
opposite is the only kind of league that can tell a correct chain from a plausible one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epl.simulate.table import TIEBREAKERS, Slate, TableError


def league(*results: tuple[str, str, int, int]) -> pd.DataFrame:
    """A finished Season of exactly these matches: home Club, away Club, home goals, away goals.

    Not a round robin. A table is a fold over whatever Fixtures exist, so a test that needs three
    matches to make its point writes three rather than padding thirty out with goalless draws.
    """
    return pd.DataFrame(
        [
            {"home_club": home, "away_club": away, "home_goals": scored, "away_goals": conceded}
            for home, away, scored, conceded in results
        ]
    )


def finished(*results: tuple[str, str, int, int]) -> Slate:
    return Slate.finished(league(*results))


def order_of(slate: Slate, seed: int = 0) -> list[str]:
    """The Clubs of a finished Slate, best first."""
    rng = np.random.default_rng(seed)
    ranks = slate.positions(None, rng)[0]
    return [slate.clubs[club] for club in np.argsort(ranks)]


class TestTheTable:
    def test_a_win_is_three_points_and_a_draw_is_one(self) -> None:
        slate = finished(("a", "b", 2, 0), ("b", "c", 1, 1))

        standings = slate.standings(None)

        assert dict(zip(slate.clubs, standings.points[0], strict=True)) == {"a": 3, "b": 1, "c": 1}

    def test_goals_are_counted_for_both_clubs_and_both_ways(self) -> None:
        slate = finished(("a", "b", 3, 1))

        standings = slate.standings(None)

        assert list(standings.goals_for[0]) == [3, 1]
        assert list(standings.goals_against[0]) == [1, 3]
        assert list(standings.goal_difference[0]) == [2, -2]

    def test_played_won_drawn_and_lost_add_up(self) -> None:
        slate = finished(("a", "b", 2, 0), ("b", "a", 1, 1), ("a", "b", 0, 4))

        standings = slate.standings(None)

        assert list(standings.played[0]) == [3, 3]
        assert list(standings.won[0]) == [1, 1]
        assert list(standings.drawn[0]) == [1, 1]
        assert list(standings.lost[0]) == [1, 1]

    def test_the_readable_table_is_ordered_and_carries_every_column(self) -> None:
        slate = finished(("a", "b", 2, 0), ("c", "a", 1, 0))
        rng = np.random.default_rng(0)

        table = slate.table(None, rng)

        assert list(table["position"]) == [1, 2, 3]
        assert list(table["club"]) == ["a", "c", "b"]
        assert list(table["points"]) == [3, 3, 0]


class TestTheTiebreakerChain:
    """Each test makes one step of the chain disagree with the step below it."""

    def test_points_come_first(self) -> None:
        """b has by far the better goal difference and goals scored, and finishes second."""
        slate = finished(("a", "b", 1, 0), ("a", "c", 1, 0), ("b", "c", 9, 0))

        assert order_of(slate) == ["a", "b", "c"]

    def test_goal_difference_comes_before_the_head_to_head_record(self) -> None:
        """All three level on points. b beat a; a finishes above b on goal difference."""
        slate = finished(("b", "a", 1, 0), ("a", "c", 6, 0), ("c", "b", 3, 0))

        assert order_of(slate) == ["a", "b", "c"]

    def test_goals_scored_comes_before_the_head_to_head_record(self) -> None:
        """Level on points and on goal difference. b is top on goals scored alone.

        And the two Clubs it leaves behind are level on all three, so the same league also shows
        the head-to-head step firing where the chain above it has run out: c beat a 2-0.
        """
        slate = finished(("a", "b", 3, 1), ("c", "a", 2, 0), ("b", "c", 3, 1))

        assert order_of(slate) == ["b", "c", "a"]

    def test_head_to_head_points_decide_when_the_first_three_are_level(self) -> None:
        """a and b are level on points, goal difference and goals scored.

        Over their two meetings a took four points to b's one — and they scored one away goal
        each, so the step below this one could not have separated them either.
        """
        slate = finished(("a", "b", 2, 1), ("b", "a", 1, 1), ("c", "a", 1, 0), ("b", "d", 1, 0))

        assert order_of(slate) == ["a", "b", "c", "d"]

    def test_head_to_head_away_goals_decide_when_head_to_head_points_are_level(self) -> None:
        """Two draws, so the head-to-head points are level; a scored twice away and b once."""
        slate = finished(("a", "b", 1, 1), ("b", "a", 2, 2))

        assert order_of(slate) == ["a", "b"]

    def test_a_coin_flip_decides_when_nothing_else_does(self) -> None:
        """Two Clubs, one goalless draw, every step of the chain level including away goals."""
        slate = finished(("a", "b", 0, 0))

        firsts = {order_of(slate, seed=seed)[0] for seed in range(20)}

        assert firsts == {"a", "b"}

    def test_the_coin_flip_is_a_coin_flip(self) -> None:
        """A play-off at a neutral ground, so neither Club may be favoured by the tie's shape."""
        slate = finished(("a", "b", 0, 0))
        rng = np.random.default_rng(15)

        firsts = [slate.clubs[int(np.argmin(slate.positions(None, rng)[0]))] for _ in range(400)]

        assert 0.44 < firsts.count("a") / len(firsts) < 0.56

    def test_the_head_to_head_record_reads_only_the_matches_among_the_tied_clubs(self) -> None:
        """b beat a; a drew level with b by beating c. The head-to-head step must not hear about c.

        The point is negative, so it is checked across many seeds: if a's win over c were counted
        into the mini-league the two would be level there too, the chain would fall through to the
        coin flip, and the order would stop being the same every time.
        """
        slate = finished(("b", "a", 1, 0), ("a", "c", 1, 0), ("d", "b", 1, 0))

        orders = {tuple(order_of(slate, seed=seed)) for seed in range(20)}

        assert orders == {("d", "b", "a", "c")}

    def test_three_clubs_level_are_settled_as_a_mini_league_and_not_in_pairs(self) -> None:
        """A multi-way tie, which the spec asks for by name and a pair cannot stand in for.

        a, b and c all finish on 7 points, +1 goal difference and 5 goals scored. Their three
        meetings give b 4 points, c 3 and a 1, so the mini-league orders them b, c, a — an order no
        single pairing produces: a *drew* with b and *lost* to c, so settling the tie pairwise in
        any order would put a somewhere it does not belong.

        All three also played d, twice each, and none of that may count here.
        """
        slate = finished(
            # The three meetings that settle it.
            ("a", "b", 1, 1),
            ("b", "c", 2, 0),
            ("c", "a", 1, 0),
            # And the matches against d that made them level, which the mini-league must not see.
            ("a", "d", 3, 2),
            ("a", "d", 1, 0),
            ("b", "d", 2, 0),
            ("d", "b", 3, 0),
            ("c", "d", 3, 1),
            ("c", "d", 1, 1),
        )

        standings = slate.standings(None)
        level = dict(zip(slate.clubs, standings.points[0], strict=True))
        orders = {tuple(order_of(slate, seed=seed)) for seed in range(20)}

        assert level["a"] == level["b"] == level["c"] == 7
        assert list(standings.goal_difference[0][:3]) == [1, 1, 1]
        assert orders == {("b", "c", "a", "d")}

    def test_the_chain_is_stated_in_order(self) -> None:
        assert TIEBREAKERS[:3] == ("points", "goal difference", "goals scored")
        assert "head-to-head points" in TIEBREAKERS[3]
        assert "head-to-head away goals" in TIEBREAKERS[4]
        assert "coin flip" in TIEBREAKERS[5]


class TestReproducibility:
    def test_the_same_seed_gives_the_same_table(self) -> None:
        slate = finished(("a", "b", 0, 0), ("c", "d", 0, 0))

        assert order_of(slate, seed=7) == order_of(slate, seed=7)

    def test_positions_are_a_permutation_of_every_place_in_the_league(self) -> None:
        slate = finished(("a", "b", 0, 0), ("b", "c", 0, 0), ("c", "d", 1, 0))
        rng = np.random.default_rng(3)

        ranks = slate.positions(None, rng)

        assert sorted(ranks[0]) == [1, 2, 3, 4]


class TestWhatASlateWillNotLetAProjectionDo:
    """The Slate holds the played results and *only* those. That is the leak rule, structurally."""

    def test_the_remaining_fixtures_bring_their_clubs_and_not_their_results(self) -> None:
        played = league(("a", "b", 1, 0))
        remaining = league(("b", "a", 9, 9))

        slate = Slate.of(played, remaining)

        assert slate.played == 1
        assert slate.remaining == 1
        assert slate.results.shape == (1, 2)
        assert 9 not in slate.results

    def test_the_simulated_goals_are_the_ones_that_land_in_the_table(self) -> None:
        """The played Fixture was drawn, so the Fixture still to come decides the league."""
        slate = Slate.of(league(("a", "b", 0, 0)), league(("b", "a", 0, 0)))

        standings = slate.standings(np.array([[[4, 0]], [[0, 4]]]))

        assert list(standings.points[0]) == [1, 4]
        assert list(standings.points[1]) == [4, 1]

    def test_a_slate_with_fixtures_left_refuses_to_produce_a_table_from_nothing(self) -> None:
        slate = Slate.of(league(("a", "b", 1, 0)), league(("b", "a", 0, 0)))

        with pytest.raises(TableError, match="1 Fixture"):
            slate.standings(None)

    def test_the_wrong_number_of_simulated_fixtures_is_an_error(self) -> None:
        slate = Slate.of(league(("a", "b", 1, 0)), league(("b", "a", 0, 0)))

        with pytest.raises(TableError, match="2"):
            slate.standings(np.zeros((3, 2, 2), dtype=np.int64))

    def test_every_simulated_season_shares_the_fixtures_already_played(self) -> None:
        slate = Slate.of(league(("a", "b", 3, 0)), league(("b", "a", 0, 0)))

        standings = slate.standings(np.zeros((5, 1, 2), dtype=np.int64))

        assert list(standings.goals_for[:, 0]) == [3] * 5
