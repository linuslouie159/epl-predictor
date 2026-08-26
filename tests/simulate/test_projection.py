"""The Monte Carlo Season Projection: ten thousand Seasons, drawn strengths, one recorded seed.

Issue #15's acceptance criteria, minus the ones that need the corpus (those are in
``test_projection_over_the_corpus.py``). Nothing here fits a posterior — a real fit is minutes
— so every test hands :func:`epl.simulate.projection.simulate` draws it built itself. That is the
point of the seam: the expensive half is one function call away from the cheap half, and the cheap
half is where every claim about the projection actually lives.

The sharpest test is :class:`TestTheDrawsAreDrawn`. A projection that quietly ran on
``Posterior.mean()`` would produce a perfectly plausible table and would be the exact mistake
ADR 0007 spends minutes a fit to avoid, so it is checked by building a posterior that
disagrees with itself and asserting the disagreement survives into the projected table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epl.simulate import projection as projection_module
from epl.simulate.checkpoints import CheckpointError
from epl.simulate.posterior import Diagnostics, Posterior
from epl.simulate.projection import (
    BANDS,
    PROJECTION_COLUMNS,
    At,
    Bands,
    Projection,
    ProjectionError,
    Simulation,
    draw_order,
    simulate,
    slate_at,
)
from epl.simulate.table import Slate

CLUBS = ("a", "b", "c", "d")

#: Four Clubs is too small a league for :data:`epl.simulate.projection.BANDS`, so every test that
#: is not about the bands names its own: the champion, the runner-up as "Europe", the bottom Club
#: relegated.
FOUR = Bands(title=1, european=2, relegation=1)


def diagnostics(seed: int = 20260825) -> Diagnostics:
    return Diagnostics(
        divergences=0,
        max_r_hat=1.0,
        min_ess_bulk=1000.0,
        min_ess_tail=1000.0,
        draws=1,
        tune=1,
        chains=1,
        seed=seed,
    )


def posterior_of(*attacks: tuple[float, ...], clubs: tuple[str, ...] = CLUBS) -> Posterior:
    """A posterior with one draw per row of ``attacks``, and nothing else varying."""
    attack = np.array(attacks, dtype=np.float64)
    return Posterior(
        clubs=clubs,
        attack=attack,
        defence=np.zeros_like(attack),
        home_advantage=np.zeros(len(attack)),
        correction=np.zeros(len(attack)),
        diagnostics=diagnostics(),
    )


def fixtures(*pairs: tuple[str, str]) -> pd.DataFrame:
    """A Fixture list with the two Club columns and nothing else — never any goals."""
    return pd.DataFrame(
        {"home_club": [home for home, _ in pairs], "away_club": [away for _, away in pairs]}
    )


def results(*played: tuple[str, str, int, int]) -> pd.DataFrame:
    frame = fixtures(*[(home, away) for home, away, _, _ in played])
    return frame.assign(
        home_goals=[scored for *_, scored, _ in played],
        away_goals=[conceded for *_, conceded in played],
    )


def round_robin(clubs: tuple[str, ...] = CLUBS) -> pd.DataFrame:
    """Every Club home and away against every other — a Season with nothing played yet."""
    return fixtures(*[(home, away) for home in clubs for away in clubs if home != away])


def unplayed(clubs: tuple[str, ...] = CLUBS) -> Slate:
    return Slate.of(results(), round_robin(clubs))


def run(
    slate: Slate,
    posterior: Posterior,
    seasons: int,
    *,
    seed: int = 20260826,
    bands: Bands = FOUR,
) -> Projection:
    return simulate(
        slate, posterior, simulation=Simulation(seasons=seasons, seed=seed), bands=bands
    )


def title_of(projection: Projection) -> dict[str, float]:
    return dict(zip(projection.clubs, projection.probability(1, 1), strict=True))


class TestWhatAProjectionReports:
    def test_every_club_gets_a_probability_of_the_title_europe_and_relegation(self) -> None:
        projection = run(unplayed(), posterior_of((0.0, 0.0, 0.0, 0.0)), 200)

        table = projection.table()

        assert list(table.columns) == list(PROJECTION_COLUMNS)
        assert set(table["club"]) == set(CLUBS)
        assert abs(table["title"].sum() - 1.0) < 1e-12

    def test_every_simulated_season_puts_every_club_in_exactly_one_position(self) -> None:
        projection = run(unplayed(), posterior_of((0.0, 0.0, 0.0, 0.0)), 137)

        assert projection.seasons == 137
        assert list(projection.finishes.sum(axis=1)) == [137] * 4
        assert list(projection.finishes.sum(axis=0)) == [137] * 4

    def test_the_stronger_club_wins_the_title_more_often(self) -> None:
        projection = run(unplayed(), posterior_of((1.5, 0.0, 0.0, 0.0)), 400)

        assert title_of(projection)["a"] > 0.8

    def test_relegation_is_the_bottom_of_the_table_and_europe_is_the_top(self) -> None:
        projection = run(unplayed(), posterior_of((1.5, 0.0, 0.0, -1.5)), 400)

        table = projection.table().set_index("club")

        assert table.loc["a", "european"] > table.loc["d", "european"]
        assert table.loc["d", "relegation"] > table.loc["a", "relegation"]

    def test_the_points_already_on_the_board_are_reported_beside_the_projected_ones(self) -> None:
        slate = Slate.of(results(("a", "b", 3, 0)), fixtures(("c", "d")))

        table = run(slate, posterior_of((0.0, 0.0, 0.0, 0.0)), 50).table().set_index("club")

        assert table.loc["a", "points"] == 3
        assert table.loc["a", "played"] == 1
        assert table.loc["c", "played"] == 0
        assert table.loc["a", "mean_points"] == 3.0

    def test_a_finished_season_projects_the_table_it_already_is(self) -> None:
        slate = Slate.of(results(("a", "b", 3, 0), ("c", "d", 0, 0)), fixtures())

        projection = run(slate, posterior_of((0.0, 0.0, 0.0, 0.0)), 50)

        assert projection.remaining == 0
        assert title_of(projection)["a"] == 1.0


class TestTheDrawsAreDrawn:
    """Strengths are sampled from the posterior on each simulation, not fixed at a point estimate.

    ADR 0007 buys the expensive fit for exactly one reason: parameter uncertainty "compounds across
    380 simulated Fixtures into a final table, and ignoring it is what makes naive season
    simulators report a title probability of 48% where the honest answer is 34%". A projection run
    from ``Posterior.mean()`` would be that naive simulator and would look entirely normal.

    So the two tests below are one test, run over a posterior that cannot decide whether a or b is
    the strong Club. Measured over 4,000 simulated Seasons, c and d take the title 6.3% of the time
    from the draws and **20.4%** from their mean, because the mean of "a is dominant" and "b is
    dominant" is "a and b are both fairly good" — a claim neither draw made. It cuts the other way
    too: a finishes bottom 17.4% of the time from the draws and 11.1% from the mean, since the mean
    has quietly forgotten the half of the posterior in which a is ordinary.
    """

    UNDECIDED = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))

    def test_a_posterior_that_disagrees_with_itself_leaves_the_rest_little_chance(self) -> None:
        projection = run(unplayed(), posterior_of(*self.UNDECIDED), 2000)

        title = title_of(projection)

        assert title["a"] > 0.4 and title["b"] > 0.4
        assert title["c"] + title["d"] < 0.10
        assert projection.probability(4, 4)[0] > 0.14

    def test_the_mean_of_the_same_draws_says_something_else_entirely(self) -> None:
        """The control, and the mistake this whole design exists to make impossible."""
        collapsed = posterior_of(tuple(posterior_of(*self.UNDECIDED).mean().attack))

        projection = run(unplayed(), collapsed, 2000)

        assert title_of(projection)["c"] + title_of(projection)["d"] > 0.15
        assert projection.probability(4, 4)[0] < 0.14

    def test_one_draw_plays_out_a_whole_simulated_season(self) -> None:
        """Within-Season strength drift is deliberately not modelled (ADR 0007), and this is it.

        Measured across 520 club-Seasons, first-half to second-half variation was indistinguishable
        from sampling noise — so a Season is played out at one draw rather than re-drawing as it
        goes. Re-drawing per Fixture would leave no test failing anywhere else, and it would be a
        different model: a Club the posterior cannot place would average out to the middle of the
        table instead of finishing first or last.

        So the posterior below is certain that a is either the best Club or the worst, and the
        check is that a lands at one end. Fresh strengths per Fixture would put it in between.
        """
        either_way = posterior_of((2.0, 0.0, 0.0, 0.0), (-2.0, 0.0, 0.0, 0.0))

        projection = run(unplayed(), either_way, 2000)

        ends = projection.probability(1, 1)[0] + projection.probability(4, 4)[0]
        assert ends > 0.85

    def test_every_draw_is_used_about_equally_often(self) -> None:
        """Spread rather than sampled with replacement, and never walked in order.

        ``fit`` concatenates its chains, so the first quarter of a posterior's draws are one
        chain's — a projection that walked them in order would explore a quarter of what it paid
        for and would look fine.
        """
        used = np.bincount(draw_order(np.random.default_rng(1), 7, 30), minlength=7)

        assert used.sum() == 30
        assert used.max() - used.min() <= 1


class TestTheSeedIsRecordedAndHonoured:
    def test_the_same_seed_reproduces_a_projection_exactly(self) -> None:
        posterior = posterior_of((0.4, 0.1, 0.0, -0.3), (0.0, 0.5, -0.2, 0.0))

        first = run(unplayed(), posterior, 250, seed=15)
        again = run(unplayed(), posterior, 250, seed=15)

        assert np.array_equal(first.finishes, again.finishes)

    def test_the_chunk_size_the_walk_uses_internally_changes_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The walk draws its uniforms in blocks to bound memory, and says that costs nothing.

        It is true because numpy fills a ``(seasons, fixtures)`` block row by row from one stream,
        so two blocks give what one would have — but "a published projection does not depend on an
        internal memory constant" is worth a test rather than a comment.
        """
        posterior = posterior_of((0.4, 0.1, 0.0, -0.3), (0.0, 0.5, -0.2, 0.0))

        whole = run(unplayed(), posterior, 250, seed=15)
        monkeypatch.setattr(projection_module, "CHUNK_CELLS", 1)
        in_pieces = run(unplayed(), posterior, 250, seed=15)

        assert np.array_equal(whole.finishes, in_pieces.finishes)

    def test_a_different_seed_gives_a_different_walk(self) -> None:
        posterior = posterior_of((0.4, 0.1, 0.0, -0.3))

        first = run(unplayed(), posterior, 250, seed=1)
        other = run(unplayed(), posterior, 250, seed=2)

        assert not np.array_equal(first.finishes, other.finishes)

    def test_both_halves_of_the_randomness_reach_the_output(self) -> None:
        """The sampler's seed and the walk's — a published projection has to be re-runnable."""
        projection = run(unplayed(), posterior_of((0.0, 0.0, 0.0, 0.0)), 20, seed=99)

        assert projection.simulation.seed == 99
        assert projection.diagnostics.seed == 20260825
        assert "99" in projection.describe()
        assert "20260825" in projection.describe()

    def test_a_projection_says_which_season_and_instant_it_is_of(self) -> None:
        at = At(season=2015, as_of=pd.Timestamp("2015-10-20"), prediction_round="2015-10-20")

        projection = simulate(
            unplayed(),
            posterior_of((0.0, 0.0, 0.0, 0.0)),
            simulation=Simulation(seasons=10),
            bands=FOUR,
            at=at,
        )

        assert "2015/16" in projection.describe()
        assert "2015-10-20" in projection.describe()


class TestSplittingASeasonAtAnInstant:
    def season(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "season": 2024,
                    "division": division,
                    "date": date,
                    "time": pd.NA,
                    "home_club": home,
                    "away_club": away,
                    "home_goals": 2,
                    "away_goals": 1,
                }
                for division, date, home, away in [
                    ("E0", "2024-08-17", "a", "b"),
                    ("E0", "2024-08-24", "c", "d"),
                    ("E0", "2024-09-14", "b", "a"),
                    ("E1", "2024-08-17", "e", "f"),
                ]
            ]
        )

    def test_the_fixtures_behind_the_instant_are_played_and_the_rest_are_not(self) -> None:
        slate = slate_at(self.season(), 2024, pd.Timestamp("2024-08-27"))

        assert slate.played == 2
        assert slate.remaining == 1

    def test_a_historical_seasons_unplayed_results_do_not_reach_the_table(self) -> None:
        """Every row of the corpus carries a result. Only the played ones may be read."""
        slate = slate_at(self.season(), 2024, pd.Timestamp("2024-08-27"))

        assert slate.results.shape == (2, 2)
        # b beat a 2-1 in the corpus. Simulated 0-0 it is a point each, so a has 3 + 1 and b has
        # 0 + 1; had the corpus row been read instead, both would sit on 3.
        assert list(slate.standings(np.zeros((1, 1, 2), dtype=np.int64)).points[0]) == [4, 1, 3, 0]

    def test_only_the_projected_division_reaches_the_table(self) -> None:
        slate = slate_at(self.season(), 2024, pd.Timestamp("2024-12-01"))

        assert slate.clubs == ("a", "b", "c", "d")

    def test_a_season_the_corpus_does_not_hold_is_an_error(self) -> None:
        with pytest.raises(CheckpointError, match="2019/20"):
            slate_at(self.season(), 2019, pd.Timestamp("2020-01-01"))


class TestTheBandsAreStatedRatherThanGuessed:
    def test_the_default_bands_are_the_title_the_top_four_and_the_bottom_three(self) -> None:
        assert (BANDS.title, BANDS.european, BANDS.relegation) == (1, 4, 3)

    def test_bands_that_overlap_in_this_league_are_an_error(self) -> None:
        with pytest.raises(ProjectionError, match="4 Clubs"):
            run(unplayed(), posterior_of((0.0, 0.0, 0.0, 0.0)), 10, bands=BANDS)

    def test_a_position_outside_the_league_is_an_error_rather_than_a_clipped_slice(self) -> None:
        projection = run(unplayed(), posterior_of((0.0, 0.0, 0.0, 0.0)), 10)

        with pytest.raises(ProjectionError, match="4 Clubs"):
            projection.probability(1, 5)


class TestWhatTheProjectionRefuses:
    def test_a_club_the_posterior_never_fitted_is_an_error_and_not_a_guess(self) -> None:
        slate = unplayed(clubs=("a", "b", "c", "e"))

        with pytest.raises(ProjectionError, match="'e'"):
            run(slate, posterior_of((0.0, 0.0, 0.0, 0.0)), 10)

    def test_an_empty_posterior_is_an_error(self) -> None:
        empty = Posterior(
            clubs=CLUBS,
            attack=np.empty((0, 4)),
            defence=np.empty((0, 4)),
            home_advantage=np.empty(0),
            correction=np.empty(0),
            diagnostics=diagnostics(),
        )

        with pytest.raises(ProjectionError, match="no draws"):
            run(unplayed(), empty, 10)

    def test_asking_for_no_seasons_is_an_error(self) -> None:
        with pytest.raises(ProjectionError, match="at least one"):
            Simulation(seasons=0)
