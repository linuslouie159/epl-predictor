"""Dixon-Coles walked over the real corpus, end to end.

Issue #13 states most of its acceptance in numbers over 7,980 Fixtures, which is the one thing a
unit test cannot check. These re-derive them from the ingested cache rather than trusting
docs/DECISIONS.md, in the same spirit as ``tests/models/test_elo_over_the_corpus.py``:

* the frozen half-life is still what the Burn-In fit produces, and the tier scope still what it
  prefers
* it beats the Naive Baseline and Elo, and does not beat the market
* a full walk over the Evaluation Window takes minutes rather than hours
* no Scoreline probability the correction produces is ever negative, over every Fixture
* what predicting per Fixture instead of per Prediction Round would have been worth (ADR 0002)

They need a populated ``data/raw/``, which is gitignored, so they skip when it is absent:

    python -m epl.ingest fetch
    python -m epl.ingest build
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from epl import metrics
from epl.ingest import DIVISIONS, FIRST_SEASON, LAST_SEASON, load_matches, raw_season_path
from epl.ledger import backtest, schema, scoreboard
from epl.models import burn_in
from epl.models.dixon_coles import DIXON_COLES, FITTED_DIVISIONS, FROZEN_DECAY, fit
from epl.models.elo import ELO, newcomers
from epl.models.likelihood import LOW_SCORES, Strengths, low_score_factor
from epl.predictors import Evidence
from epl.rounds import as_of_instant
from epl.windows import EVALUATION_WINDOW

pytestmark = pytest.mark.cache

#: What Dixon-Coles scores over the Evaluation Window's 7,980 Fixtures, walked forward.
DIXON_COLES_RPS = 0.19752

#: The three it is read between: the floor it must beat, the Elo it should improve on, and the
#: market it must not beat — a goals model that outscored the book would be evidence of a leak.
NAIVE_RPS = 0.22938
ELO_RPS = 0.19943
MARKET_RPS = 0.19362

#: The README's stated success target: the market plus 0.005.
TARGET_RPS = 0.1986

#: What the Burn-In fit finds, and what fitting the Premier League alone finds instead. The gap is
#: the promoted Clubs — ADR 0004's argument, measured for the goals model rather than assumed from
#: Elo's.
BURN_IN_RPS = 0.20165
BURN_IN_RPS_PREMIER_LEAGUE_ONLY = 0.20382

#: How long a full walk over the Evaluation Window may take. Issue #13 asks for "minutes rather
#: than hours"; this is a ceiling with room in it, not a benchmark, because a loaded machine should
#: not fail a build.
MINUTES_ALLOWED = 20.0

#: The low-score correction Dixon and Coles fitted in 1997, and what this corpus gives at the first
#: scored Prediction Round. See docs/DECISIONS.md: the dependence they found in four Seasons of one
#: division is much weaker across twenty-six of four.
PUBLISHED_CORRECTION = -0.13
CORRECTION_AT_FIRST_ROUND = -0.0582

#: What holding the correction at zero — two independent Poissons — costs on the Burn-In Window.
CORRECTION_IS_WORTH = 0.00011

#: What playing at home is worth at the same instant, in log-goals: the home Club is expected to
#: score 34.5% more than the same Club would away.
HOME_ADVANTAGE = 0.2964


def _require_cache() -> None:
    missing = [
        (season, division)
        for season in range(FIRST_SEASON, LAST_SEASON + 1)
        for division in DIVISIONS
        if not raw_season_path(season, division).exists()
    ]
    if missing:
        pytest.skip(f"raw cache incomplete ({len(missing)} files missing)")


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    _require_cache()
    return load_matches()


@pytest.fixture(scope="module")
def walked(matches: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """The whole Evaluation Window walked once, and how long it took.

    Timed here rather than in its own test because the walk is the expensive thing in this file and
    running it twice to measure it would be the joke that measures the joke.
    """
    started = time.perf_counter()
    rows = backtest.backfill(DIXON_COLES, matches)
    elapsed = time.perf_counter() - started
    return scoreboard.scored_predictions(rows, matches), elapsed


@pytest.fixture(scope="module")
def scored(walked: tuple[pd.DataFrame, float]) -> pd.DataFrame:
    return walked[0]


class TestWhatItScores:
    def test_it_beats_the_naive_baseline(self, scored: pd.DataFrame) -> None:
        card = scoreboard.build(scored.drop(columns="outcome"), _matches_of(scored))

        assert card.loc[card["predictor"] == "dixon_coles", "rps"].iloc[0] == pytest.approx(
            DIXON_COLES_RPS, abs=5e-5
        )
        assert DIXON_COLES_RPS < NAIVE_RPS

    def test_it_predicts_every_fixture_in_the_window(self, scored: pd.DataFrame) -> None:
        """Its input is goals, and goals go back to 2000/01 — so like Elo it has something to say
        about all 21 Seasons, and unlike the Ceiling Line it says it everywhere."""
        assert len(scored) == 7980
        assert set(scored["season"]) == set(EVALUATION_WINDOW)

    def test_it_improves_on_elo(self, scored: pd.DataFrame) -> None:
        """The reason issue #13 exists. Elo reduces a Club to one number; this one reads the goals,
        and the Season Projection needs them anyway (ADR 0007)."""
        assert DIXON_COLES_RPS < ELO_RPS

    def test_it_reaches_the_target_without_reaching_the_market(self, scored: pd.DataFrame) -> None:
        """The README's success condition, and the leak check beside it. A model that beat the
        book on the book's own information set would be news; here it is a symptom."""
        assert DIXON_COLES_RPS <= TARGET_RPS
        assert DIXON_COLES_RPS > MARKET_RPS, "beating the market would be news; check for a leak"

    def test_a_full_walk_takes_minutes_rather_than_hours(
        self, walked: tuple[pd.DataFrame, float]
    ) -> None:
        """Issue #13's fifth acceptance criterion, in the words it uses. A refit at every one of
        952 Prediction Rounds is what makes this worth asserting rather than assuming."""
        _, elapsed = walked

        assert elapsed < MINUTES_ALLOWED * 60


@pytest.fixture(scope="module")
def found(matches: pd.DataFrame) -> burn_in.DecayFit:
    """The Burn-In decay fit, run once. Every candidate walks 189 Prediction Rounds and refits at
    each, so this is minutes — and three tests below ask about the same run."""
    return burn_in.fit_decay(matches)


@pytest.fixture(scope="module")
def strengths(matches: pd.DataFrame) -> Strengths:
    """Dixon-Coles as it stood at the first scored Prediction Round."""
    return _strengths_at(matches, min(EVALUATION_WINDOW))[1]


@pytest.fixture(scope="module")
def readings(matches: pd.DataFrame) -> pd.DataFrame:
    """ADR 0002's diagnostic over the whole window — see :class:`TestWhatTheWeeklyBatchGivesUp`."""
    return backtest.sequential(ELO, matches)


class TestTheFrozenHalfLife:
    def test_the_fit_still_produces_what_is_frozen(self, found: burn_in.DecayFit) -> None:
        """ADR 0008 freezes the half-life as a literal, so the literal and the fit behind it can
        drift apart without anything failing. This is the thing that fails."""
        assert found.decay == FROZEN_DECAY

    def test_it_is_fitted_on_the_burn_in_window_alone(self, found: burn_in.DecayFit) -> None:
        assert found.fixtures == 1520
        assert found.rps == pytest.approx(BURN_IN_RPS, abs=5e-5)

    def test_it_beats_the_base_rate_where_it_was_fitted(
        self, found: burn_in.DecayFit, matches: pd.DataFrame
    ) -> None:
        assert found.rps < burn_in.base_rate_rps(matches)

    def test_rating_the_whole_pyramid_is_measured_and_not_assumed(
        self, found: burn_in.DecayFit, matches: pd.DataFrame
    ) -> None:
        """ADR 0004 is an argument about Elo, and Elo is zero-sum where this model is not. So the
        same question is asked again of the goals model, in the only window it may be asked in: a
        Premier-League-only fit has nothing to say about a promoted Club, and it costs 0.002 RPS.
        """
        premier_league_only = burn_in.fit_decay(matches, divisions=("E0",), refine=0)

        assert found.divisions == FITTED_DIVISIONS
        assert premier_league_only.rps == pytest.approx(
            BURN_IN_RPS_PREMIER_LEAGUE_ONLY, abs=5e-5
        )
        assert found.rps < premier_league_only.rps


class TestTheModelItself:
    def test_playing_at_home_is_worth_a_third_of_a_goal(self, strengths: Strengths) -> None:
        """Not much of a fitted claim, and a good check on the units: the home Club is expected to
        score about 35% more than the same Club would away, which is what football looks like."""
        assert strengths.home_advantage == pytest.approx(HOME_ADVANTAGE, abs=5e-4)
        assert np.exp(strengths.home_advantage) == pytest.approx(1.345, abs=0.005)

    def test_the_low_score_correction_has_all_but_vanished(
        self, strengths: Strengths, matches: pd.DataFrame
    ) -> None:
        """Dixon and Coles fitted about -0.13 on four Seasons of one division in the early 1990s.
        Over four tiers with time decay this corpus does not reproduce it: the fitted value wanders
        around zero and changes sign, and holding it *at* zero — two independent Poissons, which is
        Dixon-Coles with the Dixon-Coles taken out — costs a tenth of a thousandth of an RPS.

        A measurement about modern football rather than a defect, and not a reason to delete the
        parameter: it belongs to the likelihood the Bayesian fit shares, so dropping it would change
        the model rather than simplify the code (ADR 0007).
        """
        over_the_window = [
            _strengths_at(matches, season)[1].correction for season in (2005, 2015, 2025)
        ]

        assert strengths.correction == pytest.approx(CORRECTION_AT_FIRST_ROUND, abs=5e-4)
        assert max(abs(value) for value in over_the_window) < abs(PUBLISHED_CORRECTION) / 2
        assert min(over_the_window) < 0.0 < max(over_the_window), "it changes sign"

    def test_what_the_low_score_correction_is_worth_is_almost_nothing(
        self, matches: pd.DataFrame
    ) -> None:
        """Measured by walking the comparison rather than reasoning about the fitted value, and in
        the Burn-In Window, which is the only place a question about model structure may be asked
        (ADR 0008)."""
        one = (FROZEN_DECAY.half_life_days,)
        fitted = burn_in.fit_decay(matches, half_lives=one, refine=0)
        pinned = burn_in.fit_decay(matches, half_lives=one, refine=0, correction=0.0)

        assert pinned.rps - fitted.rps == pytest.approx(CORRECTION_IS_WORTH, abs=2e-5)
        assert 0.0 < pinned.rps - fitted.rps < 0.001

    def test_promoted_clubs_arrive_with_strengths_that_differ(
        self, strengths: Strengths, matches: pd.DataFrame
    ) -> None:
        """ADR 0004's question asked of the goals model. Three Clubs come up for 2005/06, and none
        of them may arrive at the neutral zero a Club nobody has seen would get."""
        table = strengths.table().set_index("club")
        arriving = newcomers(matches, min(EVALUATION_WINDOW))

        assert len(arriving) == 3
        attacks = [float(table.loc[club, "attack"]) for club in arriving]
        assert len(set(np.round(attacks, 6))) == 3
        assert all(abs(attack) > 1e-6 for attack in attacks)

    def test_the_tiers_come_out_in_order_without_anything_ordering_them(
        self, matches: pd.DataFrame
    ) -> None:
        """The one structural risk in fitting a goals model across four tiers. No Club ever plays
        outside its own, so nothing in the likelihood knows a division exists and the tiers are
        joined only by the Clubs that changed tier inside the decay horizon. If that were too thin
        a bridge the four would drift into four incomparable scales — and a promoted Club would
        arrive rated as though the Championship were the Premier League.

        Measured at the first scored round: mean attack falls monotonically from E0 to E3 and the
        E0-to-E3 span is more than a goal, from Clubs that never met.
        """
        instant, strengths = _strengths_at(matches, min(EVALUATION_WINDOW))
        table = strengths.table().set_index("club")
        recent = matches.loc[
            (pd.to_datetime(matches["date"]) < instant)
            & (matches["season"] == min(EVALUATION_WINDOW) - 1)
        ]

        means = [
            float(
                table.loc[
                    sorted(
                        set(recent.loc[recent["division"] == tier, "home_club"])
                        & set(table.index)
                    ),
                    "attack",
                ].mean()
            )
            for tier in DIVISIONS
        ]

        assert means == sorted(means, reverse=True), dict(zip(DIVISIONS, means, strict=True))
        assert means[0] - means[-1] > 0.6

    def test_no_scoreline_probability_is_ever_negative(self, matches: pd.DataFrame) -> None:
        """The one way the correction could misbehave: ``1 + lambda * rho`` goes negative for a
        Fixture between a very strong attack and a very weak defence, and a Scoreline probability
        below zero reaches the ledger as a Prediction that does not sum to one.
        :func:`epl.models.likelihood.scorelines` clips before it normalises; this is the check that
        the clip never has anything to do, taken at every Season's opening round rather than
        argued from the bound on rho.
        """
        for season in EVALUATION_WINDOW:
            _, strengths, home_rate, away_rate = _rates_at(matches, season)
            for goals_home, goals_away in LOW_SCORES:
                factor = low_score_factor(
                    np.full(len(home_rate), float(goals_home)),
                    np.full(len(home_rate), float(goals_away)),
                    home_rate,
                    away_rate,
                    strengths.correction,
                )

                assert factor.min() > 0.5, (season, goals_home, goals_away)


class TestWhatTheWeeklyBatchGivesUp:
    """ADR 0002's diagnostic, and the one number in this file that must never be quoted as a score.

    The project predicts in weekly Prediction Rounds so that the model, the market and the Pundits
    all see the same information. What that costs is measured here.

    **Measured on Elo rather than on Dixon-Coles, and that is an economy rather than a claim.** The
    question is about the As-Of rule, which is the same for every Predictor, and the walk takes one
    fit per distinct kickoff instead of one per round — 3,130 of them against 952. Elo pays six
    minutes for the whole Evaluation Window where Dixon-Coles pays thirty-four, and half an hour
    inside a test suite is a test nobody runs. Dixon-Coles' own figure is +0.00003, published in
    docs/DECISIONS.md and produced by ``python -m epl.models sequential``.
    """

    def test_knowing_the_earlier_kickoffs_is_worth_almost_nothing(
        self, readings: pd.DataFrame
    ) -> None:
        """The finding, and a happier one than ADR 0002 needed. Predicting each Fixture from its own
        kickoff instead of from its round is worth **0.00001 RPS** — three orders of magnitude below
        the 0.0019 that reading the goals is worth, and below the resolution anything here is
        reported at. The comparability the weekly batch buys is very nearly free.

        Bounded rather than pinned, because a difference this small is not a quantity: what the test
        protects is the claim that it is negligible, in either direction.
        """
        batch, later = _both_scores(readings)

        assert abs(batch - later) < 0.0005

    def test_it_is_no_larger_where_the_corpus_records_a_kickoff_time(
        self, readings: pd.DataFrame
    ) -> None:
        """The span where it could be larger, and is not.

        Football-Data records no kickoff time before 2019/20, so an untimed Fixture sits at midnight
        and cannot see the earlier kickoffs of its own day — the whole-window figure therefore
        understates what a per-Fixture model would know. From 2019/20 every Fixture is timed, and
        the cut is if anything generous (:func:`epl.ledger.backtest.sequential`). Both readings come
        out negligible, which is what makes the finding a finding rather than an artefact of missing
        timestamps.
        """
        timed = readings.loc[readings["season"] >= 2019]
        batch, later = _both_scores(timed)

        assert abs(batch - later) < 0.0005
        assert len(timed) == 2660

    def test_the_batch_reading_is_the_one_on_the_scoreboard(
        self, readings: pd.DataFrame
    ) -> None:
        """The diagnostic re-derives the stored Prediction rather than a different one, which is
        what makes the two columns comparable at all."""
        batch, _ = _both_scores(readings)

        assert batch == pytest.approx(ELO_RPS, abs=5e-5)

    def test_it_is_never_written_to_either_store(self, readings: pd.DataFrame) -> None:
        """A sequential Prediction is not a Prediction this project ever made (ADR 0002), and it
        carries the Outcome, which no ledger row may (ADR 0005)."""
        assert "outcome" in readings.columns
        assert set(readings.columns) != set(schema.LEDGER_COLUMNS)


class TestTheHorizonIsAToleranceAndNotAKnob:
    def test_dropping_matches_worth_less_than_a_hundredth_changes_nothing(
        self, matches: pd.DataFrame
    ) -> None:
        """The weight floor decides how far back a sample reaches, so it looks like a
        hyperparameter. Measured, it is not: over the Burn-In Window, five times the floor moves
        the score by 0.00001 RPS.
        """
        shipped = burn_in.fit_decay(
            matches, half_lives=(FROZEN_DECAY.half_life_days,), refine=0
        )
        coarser = burn_in.fit_decay(
            matches, half_lives=(FROZEN_DECAY.half_life_days,), refine=0, floor=0.05
        )

        assert FROZEN_DECAY.floor == 0.01
        assert coarser.decay.horizon < shipped.decay.horizon
        assert coarser.rps == pytest.approx(shipped.rps, abs=1e-4)


def _matches_of(scored: pd.DataFrame) -> pd.DataFrame:
    """The match table :func:`epl.ledger.scoreboard.build` needs, rebuilt from the scored rows."""
    return scored[["season", "division", "home_club", "away_club", "outcome"]]


def _strengths_at(matches: pd.DataFrame, season: int) -> tuple[pd.Timestamp, Strengths]:
    """The fit at the instant a Season's first Prediction Round is made from, and that instant.

    Through :class:`epl.predictors.Evidence`, so even a diagnostic in a test file is cut the way a
    Prediction would have been.
    """
    played = matches.loc[matches["season"] == season]
    instant = pd.Timestamp(as_of_instant(pd.to_datetime(played["date"]).min().date()))
    return instant, DIXON_COLES.strengths_at(Evidence.before(matches, instant))


def _rates_at(
    matches: pd.DataFrame, season: int
) -> tuple[pd.Timestamp, Strengths, np.ndarray, np.ndarray]:
    """The goals each side of every Premier League Fixture of a Season is expected to score.

    Every Fixture of the Season rather than only its opening round, so the check that no Scoreline
    probability goes negative reaches the widest mismatches the Season contains.
    """
    played = matches.loc[matches["season"] == season]
    instant = pd.Timestamp(as_of_instant(pd.to_datetime(played["date"]).min().date()))
    fixtures = played.loc[played["division"] == "E0"]
    home_clubs = fixtures["home_club"].to_numpy(dtype=object)
    away_clubs = fixtures["away_club"].to_numpy(dtype=object)

    sample = DIXON_COLES.sample_at(
        Evidence.before(matches, instant), also=[*home_clubs, *away_clubs]
    )
    strengths = fit(sample)
    home_rate, away_rate = strengths.rates(
        sample.index_of(home_clubs), sample.index_of(away_clubs)
    )
    return instant, strengths, home_rate, away_rate


def _both_scores(readings: pd.DataFrame) -> tuple[float, float]:
    outcomes = readings["outcome"].to_numpy(dtype=object)
    return (
        metrics.rps(readings[["prob_home", "prob_draw", "prob_away"]].to_numpy(float), outcomes),
        metrics.rps(
            readings[
                ["sequential_prob_home", "sequential_prob_draw", "sequential_prob_away"]
            ].to_numpy(float),
            outcomes,
        ),
    )
