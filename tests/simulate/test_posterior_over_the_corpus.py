"""The posterior fitted over real football, which is where ADR 0007's claim actually lives.

``tests/simulate/test_posterior.py`` makes the same checks on a synthetic sample small enough to
sample in seconds. That is worth having, and it is not the claim: ADR 0007 permits maximum
likelihood at all 952 scored rounds *because* "with ~10,000 observations and ~50 parameters,
parameter uncertainty barely moves any single Fixture's probability", and a 1,500-match sample of
eight made-up Clubs is not that. This fits the real thing — 11,873 weighted matches across four
tiers and 206 parameters — and checks:

* the posterior mean and the MLE quote one real Fixture the same, which is the acceptance criterion
  issue #14 states in those words
* the fit converges, by its own diagnostics, on the settings the module actually defaults to
* the corpus overwhelms the priors, which is what makes "stated rather than fitted" defensible
* the draws genuinely disagree with each other, which is the entire reason the expensive fit exists
* one fit costs minutes, which is what makes six checkpoints a Season affordable

Everything here is measured at a Season inside the Evaluation Window, and none of it is tuning:
every comparison is against the MLE of the same model on the same matches, and not one looks at an
Outcome. :mod:`epl.simulate.posterior` says so at more length, and ADR 0008 is why it has to.

Needs a populated ``data/raw/``, so it skips when that is absent:

    python -m epl.ingest fetch
    python -m epl.ingest build
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import pytest

from epl.ingest import DIVISIONS, FIRST_SEASON, LAST_SEASON, load_matches, raw_season_path
from epl.models.dixon_coles import DIXON_COLES
from epl.models.dixon_coles import fit as fit_mle
from epl.models.likelihood import Sample, Strengths
from epl.predictors import Evidence
from epl.rounds import as_of_instant
from epl.simulate.posterior import PRIORS, SAMPLING, Posterior, fit

pytestmark = pytest.mark.cache

#: The Prediction Round this file fits at: the first of 2015/16, in the middle of the Evaluation
#: Window. Mid-window rather than at either end, so the sample is a full decay horizon deep and is
#: not the unusual first or last one.
SEASON = 2015

#: How long one posterior fit may take. A ceiling with room in it rather than a benchmark — a
#: loaded machine should not fail the suite — but low enough to catch the fit becoming an
#: overnight job, which is the outcome ADR 0007 splits the two fits to avoid.
SECONDS_CEILING = 1_200


@dataclass(frozen=True)
class OneFit:
    """Everything the tests below read, from a single posterior taken once.

    A record rather than a tuple because four things travel together here and most tests want two
    of them: ``drawn, _ = timed_fit`` in every signature is the shape of a missing type.
    """

    sample: Sample
    fixture: pd.Series
    drawn: Posterior
    seconds: float

    @property
    def mle(self) -> Strengths:
        return fit_mle(self.sample)

    def quote(self, strengths: Strengths) -> np.ndarray:
        """The three Outcome probabilities this Fixture gets under these strengths."""
        return strengths.outcomes_for(
            self.sample.index_of([self.fixture["home_club"]]),
            self.sample.index_of([self.fixture["away_club"]]),
        )[0]


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
def one_fit() -> OneFit:
    """One posterior at the module's own defaults, taken once for the whole file.

    Module-scoped because this is minutes, not seconds. The settings are
    :data:`epl.simulate.posterior.SAMPLING` itself rather than something smaller, because "does it
    converge" is a question about what the module actually does.
    """
    _require_cache()
    matches = load_matches()
    premier_league = matches.loc[(matches["season"] == SEASON) & (matches["division"] == "E0")]
    opening = pd.to_datetime(premier_league["date"]).min()
    instant = pd.Timestamp(as_of_instant(opening.date()))

    fixture = premier_league.loc[pd.to_datetime(premier_league["date"]) == opening].iloc[0]
    evidence = Evidence.before(matches, instant)
    sample = DIXON_COLES.sample_at(evidence, also=[fixture["home_club"], fixture["away_club"]])

    clock = time.perf_counter()
    drawn = fit(sample, sampling=replace(SAMPLING, seed=11))
    return OneFit(sample, fixture, drawn, time.perf_counter() - clock)


class TestTheSampleItIsFittedOver:
    def test_it_is_the_scale_adr_0007_argues_about(self, one_fit: OneFit) -> None:
        """The ADR's arithmetic is about ~10,000 observations against ~50 parameters."""
        assert len(one_fit.sample) > 10_000
        assert one_fit.sample.club_count > 90


class TestTheTwoFitsAgree:
    def test_posterior_mean_and_mle_quote_one_real_fixture_the_same(self, one_fit: OneFit) -> None:
        """Issue #14's stated acceptance criterion, on the corpus rather than on a made-up sample.

        If this failed, the split ADR 0007 makes would be unsound: every match probability the
        project publishes comes from the MLE path, and they would be a different model's answers
        from the ones a Season Projection is built on.
        """
        assert one_fit.quote(one_fit.drawn.mean()) == pytest.approx(
            one_fit.quote(one_fit.mle), abs=0.01
        )


class TestTheFitIsSound:
    def test_it_converges_on_the_settings_the_module_defaults_to(self, one_fit: OneFit) -> None:
        assert one_fit.drawn.diagnostics.healthy, one_fit.drawn.diagnostics.describe()

    def test_one_fit_costs_minutes_rather_than_hours(self, one_fit: OneFit) -> None:
        """What makes six checkpoints across 21 Seasons a run someone will actually make."""
        assert one_fit.seconds < SECONDS_CEILING


class TestThePriorsDoNotShrink:
    """The priors are scaffolding, and this is the test that says so in numbers.

    A Bayesian fit with informative priors regresses every Club toward the mean. That is ordinary
    practice and it is wrong here: :mod:`epl.models.dixon_coles` "regresses no Club to the mean and
    carries no prior", so a posterior that shrank would be a *different model*, which is the one
    thing ADR 0007's shared likelihood exists to prevent.

    Measured rather than asserted, because it was measured going the other way first: a strength
    prior of 0.5 pulled attack to 0.65 times the MLE's and defence to 0.56, and moved one Fixture's
    Home probability by 0.079.
    """

    def test_attack_is_not_pulled_toward_the_mean(self, one_fit: OneFit) -> None:
        slope = float(np.polyfit(one_fit.mle.attack, one_fit.drawn.mean().attack, 1)[0])

        assert slope == pytest.approx(1.0, abs=0.06), f"attack came back at {slope:.3f}x the MLE"

    def test_defence_is_not_pulled_toward_the_mean_either(self, one_fit: OneFit) -> None:
        """The half that needed the model reshaped rather than the prior widened.

        Defence arrived at 0.56 times the MLE's with the first parameterisation and 0.875 with this
        one, because a plain Normal on each defence constrains the pyramid's overall scoring rate —
        which the likelihood determines and the MLE has no opinion about.
        :func:`epl.simulate.posterior.model_for` gives that level its own parameter.

        The band is wide on the low side deliberately, and 0.875 is not the shrinkage the 0.56 was.
        It is what a posterior *mean* does to a weakly-identified log-rate whose marginal is skewed,
        and it points the other way: the Clubs carrying it — Grimsby, Stockport, Darlington — are
        pushed **further** from zero than the MLE puts them, not toward it. All are in the fourth
        tier on under two weighted matches. See :data:`Diagnostics.DIVERGENCE_RATE_CEILING`, which
        is the same handful of Clubs seen from the sampler's side.
        """
        slope = float(np.polyfit(one_fit.mle.defence, one_fit.drawn.mean().defence, 1)[0])

        assert 0.83 <= slope <= 1.05, f"defence came back at {slope:.3f}x the MLE"

    def test_the_corpus_decides_home_advantage_not_the_prior(self, one_fit: OneFit) -> None:
        """The prior is centred at zero and a standard deviation wide; the corpus finds ~0.21.

        If the posterior were still sitting near the prior, the scaffolding would be load-bearing.
        """
        assert one_fit.drawn.home_advantage.std() < PRIORS.home_advantage_sigma / 20
        assert 0.15 < one_fit.drawn.home_advantage.mean() < 0.30


class TestWhatTheDrawsAdd:
    def test_the_draws_disagree_with_each_other(self, one_fit: OneFit) -> None:
        """The whole point of the expensive fit, measured on one Fixture's Home probability.

        A spread of zero would mean the posterior had collapsed onto the MLE, and a Season
        Projection built on it would be the overconfident one ADR 0007 describes — 48% where the
        honest answer is 34%.
        """
        spread = np.array(
            [
                one_fit.quote(one_fit.drawn.draw(index))[0]
                for index in range(0, len(one_fit.drawn), 20)
            ]
        )

        assert spread.std() > 0.005
        assert spread.max() - spread.min() > 0.02
