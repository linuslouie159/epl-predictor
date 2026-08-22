"""Fitting inside the Burn-In Window, and being unable to fit outside it.

ADR 0008 is the whole point of this module. Tuning K, the home-advantage constant and the ordered
logit's cutpoints against the full history and then walking a backtest over that same history still
looks rigorous and still reports a number — it is just too good. So the fit is confined to
2000/01-2004/05, and issue #9 asks that "tuning against data outside it is not possible by
accident". These tests are mostly about that word *accident*: the confinement has to be structural,
not a convention someone remembers.

The ordered logit fit is checked by recovery against synthetic Outcomes drawn from a known logit —
an independent source of truth, rather than the fit agreeing with itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epl.metrics import OUTCOMES
from epl.models import ModelError, OrderedLogit, Settings
from epl.models import burn_in as fitting
from epl.windows import BURN_IN_WINDOW, EVALUATION_WINDOW

#: A grid small enough that a unit test can afford the walk it costs. The real one is
#: ``fitting.K_GRID`` and ``fitting.HOME_ADVANTAGE_GRID``, exercised over the real corpus by
#: ``tests/models/test_elo_over_the_corpus.py``.
TINY = {"k": (10.0, 20.0), "home_advantage": (0.0, 60.0), "refine": 0}


def synthetic_pyramid(
    seasons: range, *, clubs_per_tier: int = 6, divisions: tuple[str, ...] = ("E0", "E1")
) -> pd.DataFrame:
    """A small four-tier-shaped corpus: every Club plays every other one at home, every Season.

    Outcomes come from a fixed strength ordering rather than at random, so the fit has something
    real to find and the test does not depend on a seed.
    """
    rng = np.random.default_rng(20260822)
    rows = []
    for season in seasons:
        for division in divisions:
            names = [f"{division}_{index}".lower() for index in range(clubs_per_tier)]
            strength = {name: index for index, name in enumerate(names)}
            day = pd.Timestamp(f"{season}-08-12")
            for home in names:
                for away in names:
                    if home == away:
                        continue
                    day = day + pd.Timedelta(days=1)
                    gap = strength[home] - strength[away] + 1
                    outcome = str(
                        rng.choice(
                            OUTCOMES, p=_shares(gap)  # a real signal, plus noise around it
                        )
                    )
                    rows.append(
                        {
                            "season": season,
                            "division": division,
                            "date": day.date(),
                            "time": pd.NA,
                            "home_club": home,
                            "away_club": away,
                            "outcome": outcome,
                        }
                    )
    return pd.DataFrame(rows)


def _shares(gap: int) -> list[float]:
    """Home / Draw / Away shares for a strength gap, drawn so the stronger Club wins more."""
    home = min(0.75, max(0.12, 0.42 + 0.06 * gap))
    away = min(0.75, max(0.12, 0.42 - 0.06 * gap))
    return [home, 1.0 - home - away, away]


class TestTheOrderedLogitFit:
    def test_it_recovers_the_logit_the_outcomes_were_drawn_from(self) -> None:
        """The independent source of truth: Outcomes are generated from a logit whose parameters
        the fit is not told, and the fit has to find them again."""
        rng = np.random.default_rng(20260822)
        truth = OrderedLogit(scale=150.0, cutpoints=(-0.55, 0.55))
        edges = rng.normal(0.0, 200.0, 40_000)
        cumulative = truth.probabilities(edges).cumsum(axis=1)
        drawn = rng.random(40_000)
        outcomes = np.asarray(OUTCOMES)[(drawn[:, None] > cumulative).sum(axis=1)]

        fitted = fitting.fit_logit(edges, outcomes)

        for edge in (-300.0, -100.0, 0.0, 100.0, 300.0):
            assert fitted.probabilities([edge])[0] == pytest.approx(
                truth.probabilities([edge])[0], abs=0.02
            )

    def test_it_always_returns_an_ordered_pair_of_cutpoints(self) -> None:
        """The search runs over an unconstrained parameterisation, so this is a claim about the
        parameterisation rather than about the data — an ordinal model whose cutpoints crossed
        would emit negative draw probabilities."""
        rng = np.random.default_rng(7)
        outcomes = rng.choice(OUTCOMES, 500)

        fitted = fitting.fit_logit(rng.normal(0.0, 50.0, 500), outcomes)

        assert fitted.cutpoints[0] < fitted.cutpoints[1]
        assert fitted.scale > 0

    def test_the_same_edges_and_outcomes_give_the_same_logit(self) -> None:
        """A backtest is regenerable only if the parameters behind it are (ADR 0005)."""
        rng = np.random.default_rng(11)
        edges = rng.normal(0.0, 120.0, 800)
        outcomes = rng.choice(OUTCOMES, 800)

        assert fitting.fit_logit(edges, outcomes) == fitting.fit_logit(edges, outcomes)

    def test_it_needs_more_than_one_outcome_to_fit_anything(self) -> None:
        with pytest.raises(ModelError, match="Outcome"):
            fitting.fit_logit([10.0, 20.0, 30.0], ["H", "H", "H"])

    def test_the_draw_band_is_centred_on_zero(self) -> None:
        """An edge already carries what playing at home is worth, so an edge of zero is an even
        contest and Home and Away must come out equal. Letting the centre float makes it a second
        home-advantage parameter pointing the other way: measured on the Burn-In Window it buys
        0.0003 RPS while dragging the home-advantage constant to 155 rating points."""
        rng = np.random.default_rng(3)
        edges = rng.normal(80.0, 150.0, 3_000)
        cumulative = OrderedLogit(180.0, (-0.6, 0.6)).probabilities(edges).cumsum(axis=1)
        outcomes = np.asarray(OUTCOMES)[
            (rng.random(3_000)[:, None] > cumulative).sum(axis=1)
        ]

        fitted = fitting.fit_logit(edges, outcomes)

        lower, upper = fitted.cutpoints
        assert lower == pytest.approx(-upper)
        home, _, away = fitted.probabilities([0.0])[0]
        assert home == pytest.approx(away)


class TestItCannotSeeOutsideTheBurnInWindow:
    def test_it_refuses_the_evaluation_window(self) -> None:
        with pytest.raises(ModelError, match="Burn-In Window"):
            fitting.fit(synthetic_pyramid(BURN_IN_WINDOW), seasons=EVALUATION_WINDOW, **TINY)

    def test_it_refuses_a_range_that_reaches_past_the_boundary(self) -> None:
        """The likely accident: a range written by hand that is one Season too long."""
        with pytest.raises(ModelError, match="2005"):
            fitting.fit(synthetic_pyramid(BURN_IN_WINDOW), seasons=range(2001, 2006), **TINY)

    def test_evaluation_window_matches_are_not_even_folded_into_the_ratings(self) -> None:
        """The structural half. Confining the *scored* Seasons would still let the ratings learn
        from the future, so the corpus is cut before anything is walked over — handing this the
        whole 26 Seasons is indistinguishable from handing it the five."""
        whole_history = pd.concat(
            [synthetic_pyramid(BURN_IN_WINDOW), synthetic_pyramid(range(2005, 2010))],
            ignore_index=True,
        )

        fitted = fitting.fit(whole_history, **TINY)

        assert fitted.matches_seen == len(synthetic_pyramid(BURN_IN_WINDOW))

    def test_it_warms_on_the_whole_pyramid_and_scores_only_the_premier_league(self) -> None:
        """ADR 0004 in the fit: a Club promoted into the Premier League must arrive with a rating
        it earned, so the lower tiers move ratings even though none of them is ever scored."""
        corpus = synthetic_pyramid(BURN_IN_WINDOW)

        fitted = fitting.fit(corpus, **TINY)

        scored = corpus.loc[
            (corpus["division"] == "E0") & corpus["season"].isin(list(fitting.FITTING_SEASONS))
        ]
        assert fitted.fixtures == len(scored)
        assert fitted.matches_seen > fitted.fixtures

    def test_the_first_burn_in_season_warms_up_and_is_not_scored(self) -> None:
        """Open risk 4: cross-tier ratings have no burn-in at the very start of the corpus, where
        every Club is still at the conventional 1500. Fitting on a Season in which the model knows
        nothing would mostly teach it to learn fast."""
        assert min(fitting.FITTING_SEASONS) > min(BURN_IN_WINDOW)
        assert set(fitting.FITTING_SEASONS) <= set(BURN_IN_WINDOW)


class TestWhatItFinds:
    def test_it_beats_the_base_rate_on_its_own_sample(self) -> None:
        """The weakest possible claim that fitting did something at all: a model that knows which
        Clubs are playing should beat one that does not, on the data it was fitted to."""
        corpus = synthetic_pyramid(BURN_IN_WINDOW)
        fitted = fitting.fit(corpus, **TINY)

        assert fitted.rps < fitting.base_rate_rps(corpus)

    def test_it_chooses_from_the_grid_it_was_given(self) -> None:
        corpus = synthetic_pyramid(BURN_IN_WINDOW)

        fitted = fitting.fit(corpus, **TINY)

        assert fitted.settings.k in TINY["k"]
        assert fitted.settings.home_advantage in TINY["home_advantage"]

    def test_a_winner_against_the_wall_of_its_own_grid_is_refused(self) -> None:
        """The bug this caught: the home-advantage grid stopped at 140 points and the fit's answer
        sat on it, so the refinement passes re-centred on the boundary and reported it as a fitted
        value. A search that stops at the edge of what it was allowed to consider has not found an
        optimum."""
        corpus = synthetic_pyramid(BURN_IN_WINDOW)

        with pytest.raises(ModelError, match="edge of its own search"):
            fitting.fit(corpus, k=(2.0, 4.0, 6.0), home_advantage=(0.0, 10.0, 20.0), refine=0)

    def test_two_candidates_have_no_interior_so_nothing_is_against_a_wall(self) -> None:
        """Every point of a two-point grid is a wall, so there the check has nothing to say — which
        is what lets these tests afford a grid at all."""
        fitting.fit(synthetic_pyramid(BURN_IN_WINDOW), **TINY)

    def test_the_same_corpus_gives_the_same_parameters(self) -> None:
        corpus = synthetic_pyramid(BURN_IN_WINDOW)

        assert fitting.fit(corpus, **TINY) == fitting.fit(corpus, **TINY)

    def test_it_reports_both_metrics_over_the_fitting_sample(self) -> None:
        """RPS is what is minimised, because it is what the scoreboard reports (CLAUDE.md). Log
        loss is reported beside it so a reader can see the two do not disagree."""
        fitted = fitting.fit(synthetic_pyramid(BURN_IN_WINDOW), **TINY)

        assert 0.0 < fitted.rps < 0.5
        assert 0.0 < fitted.log_loss < 2.0

    def test_what_it_finds_is_a_pair_of_frozen_settings(self) -> None:
        """So that nothing downstream can adjust a hyperparameter after it was fitted."""
        fitted = fitting.fit(synthetic_pyramid(BURN_IN_WINDOW), **TINY)

        assert isinstance(fitted.settings, Settings)
        assert isinstance(fitted.logit, OrderedLogit)
        with pytest.raises(AttributeError):
            fitted.settings.k = 99.0  # type: ignore[misc]
