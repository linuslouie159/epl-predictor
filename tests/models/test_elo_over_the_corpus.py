"""Elo walked over the real corpus, end to end.

Issue #9 states most of its acceptance in numbers, and numbers over 7,980 Fixtures are the one
thing a unit test cannot check. These re-derive them from the ingested cache rather than trusting
docs/DECISIONS.md, in the same spirit as ``tests/ledger/test_the_corpus.py``:

* the frozen hyperparameters are still what the Burn-In fit produces
* Elo beats the Naive Baseline, and by how much of what the market finds
* the draw probability falls monotonically as Supremacy grows, over ten deciles
* promoted Clubs arrive with ratings that differ, in every scored Season
* nothing cross-tier and cold ever reaches a scored Prediction (open risk 4)

They need a populated ``data/raw/``, which is gitignored, so they skip when it is absent:

    python -m epl.ingest fetch
    python -m epl.ingest build
"""

from __future__ import annotations

import itertools

import pandas as pd
import pytest

from epl.benchmarks import MARKET_LINE
from epl.ingest import DIVISIONS, FIRST_SEASON, LAST_SEASON, load_matches, raw_season_path
from epl.ledger import backtest, scoreboard
from epl.models import ELO, burn_in, draw_curve
from epl.models.elo import FROZEN_LOGIT, FROZEN_SETTINGS, newcomers
from epl.predictors import Evidence
from epl.rounds import as_of_instant
from epl.windows import EVALUATION_WINDOW

pytestmark = pytest.mark.cache

#: What Elo scores over the Evaluation Window's 7,980 Fixtures, walked forward.
ELO_RPS = 0.19943

#: The two it is read between. The floor has no value below it; the market is the opponent.
NAIVE_RPS = 0.22938
MARKET_RPS = 0.19362

#: The draw probability Elo quotes at the two ends of its own Supremacy range.
DRAW_AT_EVEN = 0.3016
DRAW_AT_WIDEST = 0.1450

#: What issue #9 asks the range be measured against. It is the *observed* draw rate bucketed by
#: **market** Supremacy, not by a model's — which is why Elo does not reach it and is not expected
#: to. See docs/DECISIONS.md, "Measured at stage 5".
DESIGN_DRAW_AT_EVEN = 0.323
DESIGN_DRAW_AT_WIDEST = 0.134

#: How many matches the least-warmed Premier League rating rests on at the first scored round.
#: Five Burn-In Seasons of football, which is what ADR 0008 buys and what open risk 4 needs.
COLDEST_AT_FIRST_SCORED_ROUND = 190


def _require_cache() -> None:
    missing = [
        (season, division)
        for season in range(FIRST_SEASON, LAST_SEASON + 1)
        for division in DIVISIONS
        if not raw_season_path(season, division).exists()
    ]
    if missing:
        pytest.skip(f"raw cache incomplete ({len(missing)} of 104 files missing)")


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    _require_cache()
    return load_matches()


@pytest.fixture(scope="module")
def scored(matches: pd.DataFrame) -> pd.DataFrame:
    """Elo walked over the whole Evaluation Window, with the Outcome that followed each call."""
    rows = backtest.backfill(ELO, matches)
    return scoreboard.scored_predictions(rows, matches)


@pytest.fixture(scope="module")
def market_curve(matches: pd.DataFrame) -> pd.DataFrame:
    """The Market Line's draw curve over the same 7,980 Fixtures, cut the same way.

    Elo's numbers only mean anything beside something, and the something issue #9 names is the
    design's 32.3% -> 13.4%. Deriving the market's curve here rather than quoting it is what turns
    "that figure came from the market's ordering" from a claim into a measurement.
    """
    rows = backtest.backfill(MARKET_LINE, matches)
    scored = scoreboard.scored_predictions(rows, matches)
    return _curve(scored)


@pytest.fixture(scope="module")
def fitted(matches: pd.DataFrame) -> burn_in.Fit:
    """The Burn-In fit, run once. It walks the pyramid under every candidate on the grid, which is
    most of a minute — and three tests below ask about the same run, not three of them."""
    return burn_in.fit(matches)


class TestWhatItScores:
    def test_it_beats_the_naive_baseline(self, scored: pd.DataFrame) -> None:
        """Issue #9's headline: an Elo that does not beat the floor has no value at all."""
        card = scoreboard.build(scored.drop(columns="outcome"), _matches_of(scored))

        assert card.loc[card["predictor"] == "elo", "rps"].iloc[0] == pytest.approx(
            ELO_RPS, abs=5e-5
        )
        assert ELO_RPS < NAIVE_RPS

    def test_it_predicts_every_fixture_in_the_window(self, scored: pd.DataFrame) -> None:
        """Elo's input is results, and results go back to 2000/01 — so unlike the Ceiling Line it
        has something to say about all 21 Seasons."""
        assert len(scored) == 7980
        assert set(scored["season"]) == set(EVALUATION_WINDOW)

    def test_it_finds_most_of_what_the_market_finds(self, scored: pd.DataFrame) -> None:
        """The honest framing of the gap. The market takes 0.0358 RPS out of the floor; Elo takes
        0.0300 of that, and is 0.0058 short of the market — a first model, before calibration
        (issue #10) or Dixon-Coles (issue #13)."""
        market_edge = NAIVE_RPS - MARKET_RPS
        elo_edge = NAIVE_RPS - ELO_RPS

        assert elo_edge / market_edge > 0.8
        assert ELO_RPS > MARKET_RPS, "beating the market would be news; check for a leak"


class TestTheDrawBandOverTheCorpus:
    def test_the_predicted_draw_rate_falls_at_every_step(self, scored: pd.DataFrame) -> None:
        """ADR 0006's claim over 7,980 Fixtures: monotone across all ten deciles, and not one line
        of taper anywhere in the model."""
        curve = _curve(scored)

        assert len(curve) == 10
        assert all(
            earlier > later
            for earlier, later in itertools.pairwise(curve["predicted_draw"])
        )

    def test_its_range_is_narrower_than_the_one_the_market_induces(
        self, scored: pd.DataFrame
    ) -> None:
        """Issue #9 asks for "the measured range of 32.3% ... down to 13.4%". Elo does not quite
        reach it: it spans 30.2% to 14.5%, a 2.08x fall against the design's 2.4x.

        The design figure is the *observed* draw rate bucketed by **market** Supremacy, and it
        still measures at 32.0% to 13.7%. Elo's Supremacy is noisier, so its most-even decile is a
        genuinely less even set of Fixtures than the market's and draws less often there. The shape
        is reproduced; the endpoints are Elo's own, and are pinned as Elo's own rather than as the
        criterion having been met.
        """
        curve = _curve(scored)

        assert curve["predicted_draw"].iloc[0] == pytest.approx(DRAW_AT_EVEN, abs=5e-4)
        assert curve["predicted_draw"].iloc[-1] == pytest.approx(DRAW_AT_WIDEST, abs=5e-4)
        assert DRAW_AT_EVEN < DESIGN_DRAW_AT_EVEN
        assert DRAW_AT_EVEN / DRAW_AT_WIDEST < DESIGN_DRAW_AT_EVEN / DESIGN_DRAW_AT_WIDEST

    def test_the_draw_rate_that_actually_happened_falls_too(self, scored: pd.DataFrame) -> None:
        """The half that is not the model's own arithmetic. Bucketed by Elo's Supremacy, the
        observed draw rate runs 27.6% to 13.8% — so the ordering means something."""
        curve = _curve(scored)

        assert curve["observed_draw"].iloc[0] > curve["observed_draw"].iloc[-1] * 1.8

    def test_it_quotes_draws_a_little_too_often(self, scored: pd.DataFrame) -> None:
        """Frozen hyperparameters drift, and ADR 0008 accepts it in exchange for a protocol that
        is trivially verifiable as leak-free. The band was fitted on 2001/02-2004/05 and the draw
        rate has fallen since, so Elo is long draws in every bucket. This is what issue #10's
        shared calibration layer is for, and pinning it here is what stops that being forgotten."""
        curve = _curve(scored)

        assert (curve["predicted_draw"] > curve["observed_draw"]).all()

    def test_the_design_figure_is_the_market_ordering_and_not_a_models(
        self, market_curve: pd.DataFrame
    ) -> None:
        """The claim the whole reading of issue #9's fourth criterion rests on, measured.

        The criterion asks for "the measured range of 32.3% for evenly matched Clubs down to 13.4%
        at the widest mismatch". Nothing in the design says which Supremacy those buckets were cut
        by — and at design time there was no model, so the only one available was the market's.
        Bucketing the *observed* draw rate by the Market Line's Supremacy reproduces the design
        figure to within a few tenths of a point, which is what makes that reading the right one.

        Without this the reading is an assertion in a docstring, and the criterion could be waved
        through on a story rather than on a number. docs/DECISIONS.md states it; this derives it.
        """
        observed = market_curve["observed_draw"]

        assert observed.iloc[0] == pytest.approx(DESIGN_DRAW_AT_EVEN, abs=0.005)
        assert observed.iloc[-1] == pytest.approx(DESIGN_DRAW_AT_WIDEST, abs=0.005)

    def test_a_noisier_supremacy_is_what_costs_elo_the_top_of_the_range(
        self, scored: pd.DataFrame, market_curve: pd.DataFrame
    ) -> None:
        """The mechanism behind the shortfall, rather than the fact of it.

        Both orderings are cut into ten equal buckets over the same 7,980 Fixtures, so the only
        difference is *which* Fixtures land in the most-even one. The market sorts them more
        sharply, so its closest bucket is genuinely closer and draws more often; Elo's is a blunter
        cut of the same corpus and draws less often. That is a property of the ordering, not of the
        draw band — which is why widening the band would not close the gap, and would only make
        Elo quote draws it already over-quotes.
        """
        elo_even = _curve(scored)["observed_draw"].iloc[0]
        market_even = market_curve["observed_draw"].iloc[0]

        assert elo_even < market_even
        # The far end is a much easier ordering problem — both know a mismatch when they see one —
        # so the two agree there, and the gap is specific to the even end.
        assert _curve(scored)["observed_draw"].iloc[-1] == pytest.approx(
            market_curve["observed_draw"].iloc[-1], abs=0.02
        )


class TestTheRatingsThemselves:
    def test_promoted_clubs_arrive_with_ratings_that_differ(
        self, matches: pd.DataFrame
    ) -> None:
        """Issue #9, and the whole reason for ADR 0004: any Premier League-only prior would give
        all three the same rating on day one. Checked in every scored Season, not just one."""
        for season in EVALUATION_WINDOW:
            pool = _pool_at(matches, season)
            arriving = newcomers(matches, season)
            ratings = [pool.rating(club) for club in arriving]

            assert len(set(ratings)) == len(arriving), f"{season}: shared arrival ratings"
            assert all(pool.played(club) > 0 for club in arriving), f"{season}: cold arrival"

    def test_no_cold_cross_tier_rating_reaches_a_scored_prediction(
        self, matches: pd.DataFrame
    ) -> None:
        """Open risk 4. Cross-tier ratings have no burn-in at the very start of the corpus, where
        every Club sits at the conventional 1500 — but the Burn-In Window is five Seasons long, so
        by the first scored round the thinnest Premier League rating rests on 190 matches."""
        pool = _pool_at(matches, min(EVALUATION_WINDOW))
        opening = matches.loc[
            (matches["season"] == min(EVALUATION_WINDOW)) & (matches["division"] == "E0")
        ]
        clubs = set(opening["home_club"]) | set(opening["away_club"])

        assert min(pool.played(club) for club in clubs) == COLDEST_AT_FIRST_SCORED_ROUND

    def test_relegated_clubs_keep_updating_instead_of_freezing(
        self, matches: pd.DataFrame
    ) -> None:
        """The other half of ADR 0004. A Club relegated out of the Premier League goes on playing,
        so its rating goes on moving — which is why it can come back up with an earned one."""
        early = _pool_at(matches, 2006)
        late = _pool_at(matches, 2010)
        left_behind = set(early.clubs) - set(
            matches.loc[matches["season"] == 2009, "home_club"]
        )

        moved = [club for club in left_behind if late.played(club) > early.played(club)]
        assert moved, "no Club outside the Premier League kept playing"


class TestTheFrozenHyperparameters:
    def test_the_fit_still_produces_what_is_frozen(self, fitted: burn_in.Fit) -> None:
        """ADR 0008 freezes the hyperparameters as literals, which means the literals and the fit
        behind them can drift apart in silence. This is the only thing that notices.

        K and the home-advantage constant are grid points and compared exactly. The logit's two are
        the output of a continuous search, so they are compared through the Predictions they make —
        a scipy release that moved the last few digits should not fail this, and one that moved the
        model should.
        """
        assert fitted.settings == FROZEN_SETTINGS
        for edge in (-400.0, -100.0, 0.0, 100.0, 400.0):
            assert fitted.logit.probabilities([edge])[0] == pytest.approx(
                FROZEN_LOGIT.probabilities([edge])[0], abs=1e-4
            )

    def test_it_is_fitted_on_the_burn_in_window_alone(
        self, fitted: burn_in.Fit, matches: pd.DataFrame
    ) -> None:
        """1,520 Premier League Fixtures across four Seasons, warmed by a fifth. Not one of the
        Evaluation Window's 7,980 is among them."""
        assert fitted.fixtures == 1520
        assert fitted.matches_seen == len(
            matches.loc[matches["season"].isin(range(2000, 2005))]
        )

    def test_it_beats_the_base_rate_where_it_was_fitted(
        self, fitted: burn_in.Fit, matches: pd.DataFrame
    ) -> None:
        assert fitted.rps < burn_in.base_rate_rps(matches)


def _curve(scored: pd.DataFrame) -> pd.DataFrame:
    return draw_curve(
        scored[list(scoreboard.PROBABILITY_COLUMNS)].to_numpy(float),
        scored["outcome"].to_numpy(dtype=object),
    )


def _pool_at(matches: pd.DataFrame, season: int):
    """Elo's ratings as they stood at the first Prediction Round of ``season``."""
    opening = matches.loc[matches["season"] == season]
    instant = pd.Timestamp(as_of_instant(pd.to_datetime(opening["date"]).min().date()))
    return ELO.ratings_at(Evidence.before(matches, instant))


def _matches_of(scored: pd.DataFrame) -> pd.DataFrame:
    """The Fixture-to-Outcome table the scoreboard joins against, back out of a scored frame."""
    return scored[["season", "division", "home_club", "away_club", "outcome"]]
