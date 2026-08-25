"""The Bayesian half of Dixon-Coles: the same likelihood, sampled instead of maximised.

Issue #14. The likelihood itself is tested in ``tests/models/test_likelihood.py`` and the
maximum-likelihood fit over it in ``tests/models/test_dixon_coles.py``; what is left for here is
what the *posterior* adds and what it must not change:

* the two fits agree — ADR 0007's central claim, in miniature (the corpus-scale version is in
  ``tests/simulate/test_posterior_over_the_corpus.py``)
* every draw arrives in the same gauge a fitted :class:`Strengths` arrives in, so a draw and an
  MLE can be read side by side
* the sampler's own account of itself is recorded rather than discarded
* the same seed gives the same posterior, because a published projection has to be reproducible

Samples here are synthetic and generated at strengths the test knows, so "agree" is measured
against something independent of both fits. They are also as small as the claim allows: sampling
is thousands of likelihood evaluations where the MLE is hundreds, and a unit test that took ten
minutes would be a unit test nobody runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from epl.models.dixon_coles import fit as fit_mle
from epl.models.likelihood import (
    Sample,
    Strengths,
    low_score_factor,
    negative_log_likelihood,
    pack,
)
from epl.simulate.posterior import Diagnostics, Posterior, Sampling, fit, log_likelihood_at

#: Small enough to sample in seconds, large enough that the likelihood — not the prior — is what
#: decides the answer. ADR 0007's claim is about ~10,000 observations against ~50 parameters, and
#: a sample that inverted that ratio would be testing the prior instead.
CLUBS = tuple(f"club_{index}" for index in range(8))
MATCHES = 1_500


def synthetic_sample(seed: int = 20260825) -> tuple[Sample, Strengths]:
    """Matches generated at strengths the test chose, and those strengths.

    The generating truth is returned so a failure can say whether the posterior missed the MLE or
    both missed the football.
    """
    rng = np.random.default_rng(seed)
    count = len(CLUBS)
    truth = Strengths(
        clubs=CLUBS,
        attack=np.linspace(-0.35, 0.35, count),
        defence=np.linspace(0.30, -0.30, count),
        home_advantage=0.25,
        correction=0.0,
    ).centred()

    home = rng.integers(0, count, MATCHES).astype(np.intp)
    away = ((home + rng.integers(1, count, MATCHES)) % count).astype(np.intp)
    home_rate, away_rate = truth.rates(home, away)
    return (
        Sample(
            clubs=CLUBS,
            position={club: index for index, club in enumerate(CLUBS)},
            home=home,
            away=away,
            home_goals=rng.poisson(home_rate).astype(np.float64),
            away_goals=rng.poisson(away_rate).astype(np.float64),
            weight=np.ones(MATCHES, dtype=np.float64),
        ),
        truth,
    )


#: What this file samples with — smaller than `posterior.SAMPLING`, which is what a real fit uses.
#: Named for the difference so the two are not confused: sampling is thousands of likelihood
#: evaluations where the MLE is hundreds, and a unit test that took ten minutes is one nobody runs.
TEST_SAMPLING = Sampling(draws=400, tune=600, chains=2, seed=11)


@pytest.fixture(scope="module")
def drawn() -> Posterior:
    """The one fit this file is written around, taken once and read by every test below."""
    sample, _ = synthetic_sample()
    return fit(sample, sampling=TEST_SAMPLING)


def outcome_probabilities(strengths: Strengths, home: int, away: int) -> np.ndarray:
    """What one Fixture's three Outcomes come to under these strengths."""
    return strengths.outcomes_for(
        np.array([home], dtype=np.intp), np.array([away], dtype=np.intp)
    )[0]


class TestWhereTheModelIsDefined:
    """The support check, which is what makes any of the rest of this converge.

    Written after the defect rather than before it. The symptom was a chain that crawled and
    converged somewhere wrong; the cause was that Dixon-Coles' low-score correction goes negative
    for a large enough ``rho``, where the model is not a probability distribution at all, and
    :data:`epl.models.likelihood.CORRECTION_FLOOR` keeps the log-density smooth there while putting
    ``1 / 1e-12`` into the gradient. These pin both halves so the fix cannot be tidied away.
    """

    def test_a_correction_that_turns_a_scoreline_negative_is_outside_the_support(self) -> None:
        sample, truth = synthetic_sample()
        outside = Strengths(
            truth.clubs, truth.attack, truth.defence, truth.home_advantage, correction=0.6
        )

        value, gradient = log_likelihood_at(pack(outside), sample)

        assert value == -np.inf
        assert not gradient.any()

    def test_the_clamped_gradient_it_replaces_was_astronomical(self) -> None:
        """Why ``-inf`` rather than letting the shared likelihood's floor handle it.

        The floor is right for L-BFGS-B, whose line search only accepts an improving step. Handed
        to a sampler it is a cliff: the log-density barely moves while the gradient gains ten orders
        of magnitude, and one leapfrog step against that throws the chain to infinity.
        """
        sample, truth = synthetic_sample()

        def at(correction: float) -> tuple[float, float]:
            """The smallest correction factor over the sample, and the largest gradient."""
            strengths = Strengths(
                truth.clubs, truth.attack, truth.defence, truth.home_advantage, correction
            )
            home_rate, away_rate = strengths.rates(sample.home, sample.away)
            factor = low_score_factor(
                sample.home_goals, sample.away_goals, home_rate, away_rate, correction
            )
            _, gradient = negative_log_likelihood(pack(strengths), sample)
            return float(factor.min()), float(np.abs(gradient).max())

        # The boundary is where a Scoreline probability would turn negative, and where that sits in
        # `rho` depends on the scoring rates — so the test asks the factor, never a magic number.
        inside_factor, inside_gradient = at(0.1)
        outside_factor, outside_gradient = at(0.6)

        assert inside_factor > 0 and outside_factor < 0
        assert inside_gradient < 1e4
        assert outside_gradient > 1e10


class TestWhatTheDiagnosticsSay:
    """The health verdict, tested on made-up numbers rather than by sampling badly on purpose."""

    def healthy_one(self, **overrides: float) -> Diagnostics:
        fields: dict[str, float] = {
            "divergences": 0,
            "max_r_hat": 1.0,
            "min_ess_bulk": 2_000,
            "min_ess_tail": 2_000,
            "draws": 1_000,
            "tune": 1_000,
            "chains": 4,
            "seed": 11,
        }
        fields.update(overrides)
        return Diagnostics(**fields)  # type: ignore[arg-type]

    def test_a_clean_fit_has_nothing_to_say(self) -> None:
        assert self.healthy_one().concerns() == []
        assert self.healthy_one().healthy

    def test_a_few_divergences_are_tolerated_and_named_when_they_are_not(self) -> None:
        """Not zero, and the reason is the fourth tier rather than a concession.

        A Club with half a weighted match has a posterior that is its prior, and draws from it walk
        into the region the support check refuses. Narrowing the prior to stop that would bring back
        the shrinkage `Priors` exists to avoid.
        """
        assert self.healthy_one(divergences=120).healthy
        assert not self.healthy_one(divergences=1_000).healthy
        assert "diverged" in self.healthy_one(divergences=1_000).concerns()[0]

    def test_it_says_which_check_failed_rather_than_that_something_did(self) -> None:
        """"Did not converge" tells a reader to act without saying what to do."""
        assert "sample for longer" in self.healthy_one(min_ess_tail=50).concerns()[0]
        assert "chains disagree" in self.healthy_one(max_r_hat=1.4).concerns()[0]


class TestWhatIsRecorded:
    def test_the_sampler_reports_on_itself(self, drawn: Posterior) -> None:
        """Issue #14 asks for sampling diagnostics with each fit, and they are not decoration.

        A posterior that failed to converge produces a Season Projection that looks exactly like one
        that did — a table of plausible probabilities. These are the only thing that says otherwise.
        """
        health = drawn.diagnostics

        assert health.divergences == 0
        assert health.max_r_hat < 1.05
        assert health.min_ess_bulk > 100
        assert (health.draws, health.tune, health.chains) == (400, 600, 2)
        assert health.seed == 11


class TestWhatADrawIs:
    def test_every_draw_is_the_type_the_rest_of_the_project_speaks(self, drawn: Posterior) -> None:
        """ADR 0007's "both paths share one likelihood" reaches as far as the type they return.

        Issue #15 simulates a Season from a draw, and it should not need to know which fit produced
        the strengths it was handed.
        """
        one = drawn.draw(0)

        assert isinstance(one, Strengths)
        assert one.clubs == CLUBS
        assert len(drawn) == 800  # 400 draws x 2 chains, concatenated

    def test_every_draw_arrives_in_the_gauge_a_fitted_one_does(self, drawn: Posterior) -> None:
        """Attack averaging zero, on every draw and not merely on the mean.

        The likelihood is flat along "add a constant to every attack and every defence", so a draw
        that had not been put in a gauge would be an arbitrary point on that line — and a Season
        Projection averaging over draws in *different* gauges would be averaging over nothing.
        """
        assert drawn.attack.mean(axis=1) == pytest.approx(0.0, abs=1e-9)
        assert drawn.draw(17).attack.mean() == pytest.approx(0.0, abs=1e-9)

    def test_the_draws_actually_differ(self, drawn: Posterior) -> None:
        """The whole reason the expensive fit exists.

        Parameter uncertainty is what a Season Projection compounds across 380 Fixtures, so draws
        that were all the same would be an MLE wearing 800 hats — and would report the 48% title
        probability ADR 0007 exists to avoid.
        """
        assert drawn.attack.std(axis=0).min() > 0.001


class TestReproducibility:
    def test_the_same_seed_gives_the_same_draws(self) -> None:
        """A published projection has to be re-runnable, which starts here rather than at #15.

        Deliberately a small fit: this is about the seed, not about convergence.
        """
        sample, _ = synthetic_sample()
        settings = Sampling(draws=50, tune=100, chains=1, seed=4)

        first = fit(sample, sampling=settings)
        again = fit(sample, sampling=settings)

        assert first.attack == pytest.approx(again.attack)
        assert first.home_advantage == pytest.approx(again.home_advantage)


class TestTheTwoFitsAgree:
    def test_posterior_mean_and_mle_quote_one_fixture_the_same(self, drawn: Posterior) -> None:
        """ADR 0007's load-bearing claim, and the reason the split is allowed to exist.

        The ADR permits maximum likelihood for all 952 rounds *because* parameter uncertainty
        barely moves a single Fixture's probability. If that were false, every match probability
        this project publishes would be the wrong fit of the model.

        The tolerance is set from what the two fits actually do here rather than from what would
        be tolerable: measured across four pairings they differ by 0.0003 to 0.0018, so 0.005 has
        room for sampling noise and would still catch a fit that had drifted. It is *not* the
        corpus-scale number — that is 0.0090 over 206 parameters, and it is asserted in
        ``test_posterior_over_the_corpus.py``, which is where the claim really lives.
        """
        sample, _ = synthetic_sample()
        point = fit_mle(sample)

        for home, away in ((0, 7), (7, 0), (3, 4)):
            assert outcome_probabilities(drawn.mean(), home, away) == pytest.approx(
                outcome_probabilities(point, home, away), abs=0.005
            ), f"club_{home} v club_{away}"
