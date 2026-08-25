"""Dixon-Coles as a full posterior, fitted only where a Season Projection is produced.

The second of ADR 0007's two fits, and the expensive one. :mod:`epl.models.dixon_coles` maximises
the likelihood at every one of the 952 scored Prediction Rounds; this samples it, and is run at a
handful of Season Projection points instead. The ADR's arithmetic is the whole reason for the
split: with ~10,000 observations and ~50 parameters, parameter uncertainty barely moves any single
Fixture's probability — but it compounds across 380 simulated Fixtures into a final table, and
ignoring it is what makes a naive season simulator report a title probability of 48% where the
honest answer is 34%.

**There is no second likelihood here, and that is the point of the module.** ADR 0007's consequence
is stated as "both paths share one likelihood function, so the models cannot drift apart", and the
obvious way to write this file — re-expressing the rates, the Poissons and the low-score correction
in PyTensor so the sampler can differentiate them — would produce exactly the second
implementation the ADR exists to prevent. It would also be undetectable: two Dixon-Coles
likelihoods that disagreed in the fourth decimal would both look entirely reasonable, and only the
Season Projection would quietly be a different model from the match probabilities.

So :func:`epl.models.likelihood.negative_log_likelihood` is handed to PyMC as a single opaque node
(:class:`LogLikelihood`). That function already returns its own analytic gradient, which is exactly
the pair a PyTensor ``Op`` needs, so the sampler differentiates the same arithmetic the optimiser
descends — not a copy of it. Whatever is true of one fit's likelihood is true of the other's by
construction rather than by review.

One thing genuinely differs between the two fits, and it is not optional. The likelihood is flat
along the direction that adds a constant to every attack *and* every defence
(:meth:`epl.models.likelihood.Strengths.centred`), and the two fits cannot treat that the same way.
An optimiser walks to some arbitrary point on the ridge and is put in a gauge on the way out; a
sampler handed the same ridge has an improper posterior and will wander it forever, reporting
divergences, a tree depth at its ceiling and an ESS in the tens. The gauge therefore has to be part
of the *model* here rather than a tidy-up after it, which is what :class:`~pymc.ZeroSumNormal` on
attack is doing below. A reader who replaces it with a plain Normal will not get a wrong number;
they will get a fit that never finishes converging.

The priors are weakly informative and **stated rather than fitted**. ADR 0008 allows a
hyperparameter to be tuned only inside the Burn-In Window, and these could not be tuned even if
that were not so: a prior that had been chosen by looking at how well it predicted would be a prior
carrying information from the data it is meant to be combined with. They are set from what the
model's own units mean — attack and defence are log-goals, and
:data:`epl.models.likelihood.STRENGTH_BOUND` already says 5.0 is absurd — and they are wide enough
that the corpus overwhelms them, which is the claim
``tests/simulate/test_posterior_over_the_corpus.py`` checks rather than assumes.

**What keeps this off all 1,189 Prediction Rounds is structural, not a guard in :func:`fit`.**
:func:`fit` takes a :class:`~epl.models.likelihood.Sample` and cannot tell which round it came
from, so it could not refuse one even in principle. What actually prevents an all-rounds run is
that **nothing here is a Predictor**: no class in this module implements the
:mod:`epl.predictors` contract, nothing calls :func:`epl.predictors.register`, and so
``python -m epl.ledger backfill`` — the one thing in the project that walks every round — has no
way to reach it. :mod:`epl.simulate.checkpoints` then says where a projection *should* be taken.
Registering a Predictor over this would silently turn a four-minute fit into a four-day backfill,
which is precisely the outcome ADR 0007 splits the two fits to avoid.

**One thing about that needs saying out loud, because it looks like a rule being broken.** The
numbers quoted throughout this module — the shrinkage table behind :class:`Priors`, the sampler
settings in :class:`Sampling`, the divergence rate in :class:`Diagnostics` — were all measured at
the first Prediction Round of **2015/16**, which is inside the Evaluation Window, and CLAUDE.md says
"never tune a hyperparameter using data from outside the Burn-In Window" (ADR 0008).

Nothing here was tuned in the sense that rule means. Every one of those measurements compares this
fit against *the MLE of the same model on the same matches* — posterior mean against maximum
likelihood, chains against each other, wall clock — and **not one of them looks at an Outcome.** No
quantity here was chosen because it scored better. That is the same argument
:data:`epl.pundits.margin.MINIMUM_SAMPLE` makes for being stated rather than fitted, and it is
stated here for the same reason: a reader who finds an Evaluation-Window season number in a
constant's docstring is right to stop, and is owed the reason it is not a leak.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from epl.models.likelihood import Sample, Strengths, low_score_factor, unpack
from epl.models.likelihood import negative_log_likelihood as _negative_log_likelihood
from epl.models.ordered_logit import ModelError

if TYPE_CHECKING:  # pragma: no cover - import-time cost, see `_pymc`
    import pymc as pm


@dataclass(frozen=True)
class Priors:
    """Scaffolding, not modelling. What holds the posterior together without moving it.

    **These are deliberately too wide to do anything**, and that is the requirement rather than a
    lazy default. :mod:`epl.models.dixon_coles` says of the MLE path that "nothing here regresses a
    Club to the mean, and nothing carries a prior — the time decay is the only thing that forgets".
    A prior that shrinks is a prior that makes this a *different model* from that one, and ADR 0007
    forbids exactly that: the two fits share one likelihood so that "the models cannot drift apart".

    That is not hypothetical. Measured at the first Prediction Round of 2015/16, over 11,873
    weighted matches and 102 Clubs, a strength prior of 0.5 pulls fitted attack to **0.65 times**
    the MLE's and defence to 0.56 times — Darlington's attack from -0.645 to -0.127 — and moves one
    Fixture's Home probability by 0.079. That is textbook shrinkage, it is defensible Bayesian
    practice in general, and it is wrong *here*, because the point estimate is supposed to be the
    same model's and only the uncertainty around it is new.

    :attr:`strength_sigma` is therefore set from what the corpus has ever produced rather than from
    what seems reasonable: fitted attack has a spread of about 0.32 across the pyramid and a range
    inside +/-1, so 2.0 puts every strength this project has ever fitted within half a standard
    deviation, and :data:`epl.models.likelihood.STRENGTH_BOUND` — the likelihood's own "absurd"
    threshold, 148 goals a match — at two and a half. At that width the shrinkage is gone: attack
    comes back at 1.005 times the MLE.

    One width for both attack and defence, because the two are the same kind of quantity and
    :meth:`epl.models.likelihood.Strengths.centred` treats them as one. See :func:`model_for` for
    why the overall scoring *level* is a separate parameter rather than something these widths
    control.
    """

    #: The spread of attack and of defence around their own means, in log-goals.
    strength_sigma: float = 2.0

    #: How far the pyramid's overall scoring rate may sit from one goal a match, in log-goals.
    #: Wide, because this is precisely what the data pins down — it is the mean of every Scoreline
    #: in the sample — and the prior has no business having an opinion about it.
    level_sigma: float = 1.0

    #: Home advantage, centred at zero rather than at the 0.25 the corpus fits. Centring a prior on
    #: the answer is how a prior stops being scaffolding, and the data determines this to about
    #: +/-0.025, so the width costs nothing.
    home_advantage_sigma: float = 1.0

    #: The one prior that is deliberately *not* diffuse, and for a reason that is not statistical.
    #: Beyond about 0.35 the low-score correction turns Scoreline probabilities negative and the
    #: model stops existing (:func:`log_likelihood_at`), so a wide prior here would put most of its
    #: mass in a region every draw gets rejected from. The corpus fits about 0.003, so 0.1 is
    #: already thirty times the answer and does no shrinking.
    correction_sigma: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "strength_sigma",
            "level_sigma",
            "home_advantage_sigma",
            "correction_sigma",
        ):
            if not getattr(self, name) > 0:
                raise ModelError(f"{name} must be positive; got {getattr(self, name)}")


@dataclass(frozen=True)
class Sampling:
    """How the posterior is drawn, and the seed that makes it reproducible.

    ``seed`` is not a convenience. Issue #15 has to be able to re-run a published Season Projection
    and get the published numbers back, and the sampler is the first of the two places randomness
    enters (the Monte Carlo walk is the second). It is recorded on every
    :class:`Diagnostics` for that reason.
    """

    draws: int = 1_000
    tune: int = 1_000
    chains: int = 4
    #: Above PyMC's 0.8 default, and set from measurement rather than taste. The posterior has a
    #: few hundred correlated parameters and a wide prior on the Clubs it knows least about, so a
    #: step size tuned for a looser target spends its draws on divergences. 0.99 was tried and
    #: abandoned: it did not finish a corpus-scale fit in twenty minutes, where 0.95 takes four.
    target_accept: float = 0.95
    seed: int = 20260825
    #: How many chains run at once. It does **not** affect the draws: PyMC derives one seed per
    #: chain from ``seed`` and the chain count alone, and four chains run on four cores come back
    #: bit-identical to the same four run on one. It barely affects the clock either — measured at
    #: 58 s against 62 s, a 6% saving rather than the fourfold one the chain count suggests, because
    #: the likelihood is a Python function called through objmode and the chains queue for the GIL.
    #: That is the price of *not* writing a second likelihood in PyTensor (see the module
    #: docstring), and worth knowing before issue #15 plans a validation run around parallelism.
    cores: int | None = 1

    def __post_init__(self) -> None:
        if self.draws < 1 or self.tune < 1 or self.chains < 1:
            raise ModelError(
                f"draws, tune and chains must each be at least 1; got {self.draws}, "
                f"{self.tune}, {self.chains}"
            )
        if not 0.0 < self.target_accept < 1.0:
            raise ModelError(
                f"target_accept must sit strictly between 0 and 1; got {self.target_accept}"
            )


#: The priors every fit uses unless a caller says otherwise. Named rather than built at each call
#: site so that "what did this projection assume?" has one answer.
PRIORS = Priors()

#: The sampler settings every fit uses unless a caller says otherwise. Four chains because R-hat
#: needs more than two to mean much, and 1,000 tuning steps because the mass matrix has a few
#: hundred correlated parameters to learn.
SAMPLING = Sampling()


@dataclass(frozen=True)
class Diagnostics:
    """What the sampler says about its own run, recorded with every fit.

    Issue #14 asks for these by name, and they are not decoration. A posterior that never converged
    produces a Season Projection that looks exactly like one that did: a tidy table of plausible
    probabilities, with nothing on its face to say the chains disagreed. These are the only thing
    that says otherwise, so they travel with the draws rather than being printed and forgotten.

    :attr:`divergences` counts trajectories NUTS abandoned. A handful is the sampler declining to
    go somewhere the model is not defined — see :func:`log_likelihood_at`, which is what puts that
    boundary where the sampler can see it — and a great many means the geometry defeated it.
    :attr:`max_r_hat` compares the chains against each other, and :attr:`min_ess_bulk` says how many
    genuinely independent draws the correlated ones are worth.

    The settings are recorded beside the results because "R-hat was 1.01" means one thing after
    1,000 tuning steps and another after 100, and because ``seed`` is what makes a published
    projection reproducible.
    """

    divergences: int
    max_r_hat: float
    min_ess_bulk: float
    min_ess_tail: float
    draws: int
    tune: int
    chains: int
    seed: int

    #: What :attr:`max_r_hat` may reach before the chains are not describing one distribution.
    #: The usual modern threshold, and stated rather than defaulted so a reader can argue with it.
    R_HAT_CEILING = 1.01

    #: How many effectively independent draws are needed before a posterior mean means anything.
    ESS_FLOOR = 400

    #: What share of draws may be divergent. **Not zero, and the reason is measured rather than
    #: conceded.** A corpus-scale fit comes in at about 3.3% — 132 of 4,000 — and the divergences
    #: are not spread across the model. They belong to the Clubs with almost no football inside the
    #: decay horizon: the correlation between a Club's log weight and the width of its posterior is
    #: **-0.977**, and at the first round of 2015/16 Grimsby carries half a weighted match, a
    #: posterior standard deviation of 1.18, and draws reaching an attack of 5.85 — a rate of 330
    #: goals a match. Those draws walk into the region :func:`log_likelihood_at` refuses, and each
    #: refusal is counted as a divergence.
    #:
    #: That is the wide prior working as intended rather than failing. Narrowing it would remove
    #: the divergences and reintroduce the shrinkage :class:`Priors` exists to avoid, and the Clubs
    #: concerned are in the fourth tier, four promotions from anything a Season Projection contains.
    #: Raising ``target_accept`` to 0.99 does not finish a corpus-scale fit in twenty minutes.
    DIVERGENCE_RATE_CEILING = 0.05

    @property
    def divergence_rate(self) -> float:
        """What share of the draws NUTS abandoned."""
        total = self.draws * self.chains
        return self.divergences / total if total else 0.0

    def concerns(self) -> list[str]:
        """Which health checks this fit failed, each as a sentence naming its number.

        A list rather than a flag because the three failures mean different things and want
        different responses: too few effective draws is usually "sample for longer", a high R-hat
        is "the chains disagree", and a divergence rate is "the geometry defeated it somewhere".
        Collapsing them into "did not converge" tells a reader to do something without saying what.
        """
        failures = []
        if self.divergence_rate > self.DIVERGENCE_RATE_CEILING:
            failures.append(
                f"{self.divergences} of {self.draws * self.chains} draws diverged "
                f"({self.divergence_rate:.1%}, ceiling {self.DIVERGENCE_RATE_CEILING:.0%})"
            )
        if self.max_r_hat > self.R_HAT_CEILING:
            failures.append(
                f"the chains disagree: r_hat reaches {self.max_r_hat:.4f} "
                f"against a ceiling of {self.R_HAT_CEILING}"
            )
        if min(self.min_ess_bulk, self.min_ess_tail) < self.ESS_FLOOR:
            failures.append(
                f"too few effective draws: {self.min_ess_bulk:.0f} bulk and "
                f"{self.min_ess_tail:.0f} tail against a floor of {self.ESS_FLOOR} — "
                "sample for longer"
            )
        return failures

    @property
    def healthy(self) -> bool:
        """Whether this fit is fit to publish from.

        Deliberately not raised on inside :func:`fit`. A diagnostic that aborts the run is one that
        cannot be looked at, and the first thing anyone wants when a fit misbehaves is the fit.
        Issue #15 is where a projection decides what to do about an unhealthy posterior.
        """
        return not self.concerns()

    def describe(self) -> str:
        """One line, for a report or a log."""
        return (
            f"{self.chains} chains x {self.draws} draws (tune {self.tune}, seed {self.seed}): "
            f"{self.divergences} divergences ({self.divergence_rate:.1%}), "
            f"max r_hat {self.max_r_hat:.4f}, "
            f"min ess {self.min_ess_bulk:.0f} bulk / {self.min_ess_tail:.0f} tail"
        )


@dataclass(frozen=True)
class Posterior:
    """Draws of the same :class:`~epl.models.likelihood.Strengths` the MLE path returns one of.

    Held as arrays rather than as a list of :class:`~epl.models.likelihood.Strengths` because the
    Season Projection wants 10,000 simulated Seasons and will index this per simulation;
    :meth:`draw` is what turns one row back into the shared type.

    Every draw is in the gauge a fitted ``Strengths`` is in — attack averaging zero — so a draw and
    an MLE can be read side by side. That is enforced by the model rather than repaired afterwards;
    see the module docstring.
    """

    clubs: tuple[str, ...]
    #: ``(draws, clubs)`` — one row per posterior draw, chains already concatenated.
    attack: npt.NDArray[np.float64]
    defence: npt.NDArray[np.float64]
    #: ``(draws,)``.
    home_advantage: npt.NDArray[np.float64]
    correction: npt.NDArray[np.float64]
    diagnostics: Diagnostics

    def __len__(self) -> int:
        return len(self.home_advantage)

    def draw(self, index: int) -> Strengths:
        """One draw, as the type every other part of this project already understands."""
        return Strengths(
            clubs=self.clubs,
            attack=self.attack[index],
            defence=self.defence[index],
            home_advantage=float(self.home_advantage[index]),
            correction=float(self.correction[index]),
        )

    def mean(self) -> Strengths:
        """The posterior mean, as a single :class:`~epl.models.likelihood.Strengths`.

        What ADR 0007 compares against the MLE, and **not** what a Season Projection should be run
        from: collapsing the draws to their mean throws away precisely the parameter uncertainty
        that the expensive fit was run to capture.
        """
        return Strengths(
            clubs=self.clubs,
            attack=self.attack.mean(axis=0),
            defence=self.defence.mean(axis=0),
            home_advantage=float(self.home_advantage.mean()),
            correction=float(self.correction.mean()),
        ).centred()


def _pymc() -> Any:
    """PyMC, imported on first use rather than at module import.

    Importing PyMC costs seconds and pulls in PyTensor, a BLAS provider and a C toolchain — a
    stack that this project has already twice found to be where a free version choice breaks the
    build rather than drifting (``environment.yml``). ``epl.simulate`` is imported by the layout
    test in a fresh interpreter and, in time, by anything that reads a projection; none of that
    should pay for a sampler it is not going to run.
    """
    import pymc

    return pymc


def log_likelihood_at(
    free: npt.NDArray[np.float64], sample: Sample
) -> tuple[float, npt.NDArray[np.float64]]:
    """The shared likelihood, negated back into a log-density, and made safe to evaluate anywhere.

    The negation is bookkeeping: :func:`epl.models.likelihood.negative_log_likelihood` is written
    for an optimiser, which minimises, and a sampler wants the log-density itself.

    The support check is not bookkeeping, and it is the one thing the Bayesian path needs that the
    MLE path gets for free. **That likelihood is only safe inside a box**, and
    :func:`epl.models.likelihood.bounds` is what keeps L-BFGS-B inside one. A sampler has no box,
    and two things go wrong outside it.

    The first is ordinary: far enough out, ``exp(attack - defence)`` overflows and both the value
    and the gradient come back ``nan``.

    The second is the dangerous one, because nothing about it looks wrong. Dixon-Coles' low-score
    correction multiplies four Scorelines by factors that go **negative** once the correction is
    large enough — at ``rho`` beyond about 0.35 on a typical sample, which is only three and a half
    prior standard deviations out. There the model is not a probability distribution at all, and
    :data:`epl.models.likelihood.CORRECTION_FLOOR` clamps the factor at 1e-12 before its logarithm
    so the optimiser's line search cannot fall off the edge. That clamp keeps the *value* smooth and
    finite — and puts ``1 / 1e-12`` into the gradient. Measured on a 600-match sample, the largest
    gradient component goes from 229 at ``rho = 0.30`` to **9.3e12** at ``rho = 0.35``, while the
    log-density moves by less than a hundred.

    A sampler handed that takes one leapfrog step against a gradient of 1e12, throws the position to
    infinity, overflows, and produces a ``nan`` — which is worse than an error, because NUTS tests
    for divergence with a comparison and every comparison against ``nan`` is false. The trajectory
    is never rejected: it doubles until it hits the tree-depth ceiling, on every draw, and the chain
    both crawls and converges somewhere wrong. Measured, that is 1,023 leapfrog steps per draw
    against 7, and a posterior mean a long way from the MLE.

    So the region where any correction factor is non-positive is refused outright. That is not a
    patch on the shared likelihood; it is the model's actual support written down. A Scoreline
    probability cannot be negative, so those parameter values have zero likelihood, and ``-inf``
    with a zero gradient is how a black-box density says so in terms NUTS already understands: the
    point is rejected as divergent, the step size adapts down, and the sampler stays where the model
    is defined. The clamp remains right for the optimiser, whose line search only accepts an
    improving step and so never lingers there.

    The check is built from :func:`epl.models.likelihood.low_score_factor` — the same function the
    likelihood corrects with and the Scoreline grid is built from — so this asks the model where it
    is valid rather than restating any part of it.
    """
    # `errstate` because the overflow is expected here and handled below. The MLE path never
    # silences it and must not start: there, a warning means the optimiser has left its box, which
    # is a real defect. Here it means tuning proposed something absurd, which is tuning working.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        strengths = unpack(np.asarray(free, dtype=np.float64), sample.clubs)
        home_rate, away_rate = strengths.rates(sample.home, sample.away)
        outside = not (np.all(np.isfinite(home_rate)) and np.all(np.isfinite(away_rate)))
        if not outside and len(sample):
            factor = low_score_factor(
                sample.home_goals, sample.away_goals, home_rate, away_rate, strengths.correction
            )
            outside = not np.all(factor > 0.0)
        if outside:
            return -np.inf, np.zeros(2 * sample.club_count + 2, dtype=np.float64)

        value, gradient = _negative_log_likelihood(free, sample)
        gradient = -np.asarray(gradient, dtype=np.float64)
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            return -np.inf, np.zeros_like(gradient)
        return -float(value), gradient


def _log_likelihood_op() -> Any:
    """The shared likelihood as one PyTensor ``Op``, defined on first use.

    Inside a function because subclassing ``Op`` requires PyTensor at class-definition time, and
    :func:`_pymc` explains why that is not paid at import.
    """
    import pytensor.tensor as pt
    from pytensor.graph.op import Op

    class LogLikelihoodGrad(Op):  # type: ignore[misc]
        """The gradient of the shared likelihood, which the shared likelihood already computes."""

        # PyTensor declares these on `Op` itself, so they are plain class attributes rather
        # than ClassVar: typing them as one is what mypy objects to.
        itypes = [pt.dvector]  # noqa: RUF012
        otypes = [pt.dvector]  # noqa: RUF012

        def __init__(self, sample: Sample) -> None:
            self.sample = sample

        def perform(self, node: Any, inputs: Sequence[Any], outputs: Sequence[Any]) -> None:
            _, gradient = log_likelihood_at(inputs[0], self.sample)
            outputs[0][0] = gradient

    class LogLikelihood(Op):  # type: ignore[misc]
        """:func:`epl.models.likelihood.negative_log_likelihood`, negated, as a PyTensor node.

        Opaque on purpose. The sampler sees one node it cannot look inside, evaluates it in numpy
        and differentiates it with the analytic gradient the same call returns — so the arithmetic
        being sampled is the arithmetic being optimised, rather than a faithful copy of it.
        """

        itypes = [pt.dvector]  # noqa: RUF012
        otypes = [pt.dscalar]  # noqa: RUF012

        def __init__(self, sample: Sample) -> None:
            self.sample = sample
            self._gradient = LogLikelihoodGrad(sample)

        def perform(self, node: Any, inputs: Sequence[Any], outputs: Sequence[Any]) -> None:
            value, _ = log_likelihood_at(inputs[0], self.sample)
            outputs[0][0] = np.asarray(value)

        def grad(self, inputs: Sequence[Any], output_gradients: Sequence[Any]) -> list[Any]:
            return [output_gradients[0] * self._gradient(inputs[0])]

    return LogLikelihood


def model_for(sample: Sample, priors: Priors = PRIORS) -> pm.Model:
    """The PyMC model over one weighted sample: the gauge, the scaffolding, and the likelihood.

    Public because it is the receipt for the claim the module docstring makes. A reader who wants
    to know whether the posterior really is the same model as the MLE can build this and look at
    what is in it: four prior terms and one opaque node, with no rates, no Poisson and no low-score
    correction restated anywhere.

    **Attack and defence are built the same way, and the shape of this model is that fact.** Both
    are zero-sum deviations; the overall scoring rate they are deviations *from* is
    ``scoring_level``, one parameter with a wide prior. The obvious alternative — a plain Normal on
    each defence — is what the first version did, and it is subtly wrong: a Normal centred at zero
    constrains not only the *spread* of defence but its *mean*, and that mean is the pyramid's goal
    rate, which the likelihood determines from every Scoreline in the sample. The MLE never has an
    opinion about it, so neither may this. Measured, that asymmetry left defence at 0.86 times the
    MLE's while attack had already recovered to 1.005.

    The gauge is the other half of the same idea, and it is not optional. The likelihood is flat
    along "add a constant to every attack and every defence"
    (:meth:`epl.models.likelihood.Strengths.centred`), and the two fits cannot treat that the same
    way: an optimiser walks to an arbitrary point on the ridge and is put in a gauge on the way out,
    while a sampler handed the same ridge has an improper posterior and wanders it forever. Making
    attack sum to zero *is* that gauge, expressed where a sampler can see it.
    """
    import pytensor.tensor as pt

    pm = _pymc()
    log_likelihood = _log_likelihood_op()(sample)

    with pm.Model() as model:
        attack = pm.ZeroSumNormal("attack", sigma=priors.strength_sigma, shape=sample.club_count)
        spread = pm.ZeroSumNormal(
            "defence_spread", sigma=priors.strength_sigma, shape=sample.club_count
        )
        scoring_level = pm.Normal("scoring_level", mu=0.0, sigma=priors.level_sigma)
        # Recorded as a Deterministic so the draws carry defence itself rather than its two
        # halves: `Posterior` hands out `Strengths`, and a Strengths has a defence per Club.
        defence = pm.Deterministic("defence", scoring_level + spread)
        home_advantage = pm.Normal(
            "home_advantage", mu=0.0, sigma=priors.home_advantage_sigma
        )
        correction = pm.Normal("correction", mu=0.0, sigma=priors.correction_sigma)
        pm.Potential(
            "dixon_coles",
            log_likelihood(
                pt.concatenate([attack, defence, pt.stack([home_advantage, correction])])
            ),
        )
    return model


def fit(
    sample: Sample,
    *,
    priors: Priors = PRIORS,
    sampling: Sampling = SAMPLING,
) -> Posterior:
    """Draws from the posterior over one weighted sample.

    The Bayesian counterpart of :func:`epl.models.dixon_coles.fit`, taking the same
    :class:`~epl.models.likelihood.Sample` and returning draws of the same
    :class:`~epl.models.likelihood.Strengths`.

    Sampled with nutpie (ADR 0009's conda-forge stack), because the alternative on this machine is
    PyMC's own NUTS over a PyTensor graph with no working C compiler.
    """
    if not sample.club_count:
        raise ModelError("a posterior needs at least one Club; this sample has none")

    pm = _pymc()
    with model_for(sample, priors):
        trace = pm.sample(
            draws=sampling.draws,
            tune=sampling.tune,
            chains=sampling.chains,
            cores=sampling.cores if sampling.cores is not None else sampling.chains,
            target_accept=sampling.target_accept,
            nuts_sampler="nutpie",
            random_seed=sampling.seed,
            progressbar=False,
        )

    return _posterior_from(trace, sample.clubs, sampling)


#: The model's four parameter blocks, in the order :func:`epl.models.likelihood.pack` lays them out.
PARAMETERS = ("attack", "defence", "home_advantage", "correction")


def _posterior_from(trace: Any, clubs: tuple[str, ...], sampling: Sampling) -> Posterior:
    """The sampler's draws, with the chains concatenated and the gauge re-asserted."""
    drawn = trace.posterior

    def flat(name: str) -> npt.NDArray[np.float64]:
        stacked = drawn[name].stack(sample=("chain", "draw"))
        values = np.asarray(stacked.to_numpy(), dtype=np.float64)
        return values.T if values.ndim > 1 else values

    attack = flat("attack")
    # Re-asserted rather than trusted. The gauge is the model's job (`ZeroSumNormal`), and this is
    # the one line that makes a change to the model's parameterisation fail loudly here instead of
    # quietly producing draws that no longer line up with a fitted `Strengths`.
    shift = attack.mean(axis=1, keepdims=True)
    return Posterior(
        clubs=clubs,
        attack=attack - shift,
        defence=flat("defence") - shift,
        home_advantage=flat("home_advantage"),
        correction=flat("correction"),
        diagnostics=_diagnostics_from(trace, sampling),
    )


def _diagnostics_from(trace: Any, sampling: Sampling) -> Diagnostics:
    """R-hat, ESS and divergences, taken over every parameter and reduced to the worst of each.

    The worst rather than the average, because a posterior is only as trustworthy as its least
    converged parameter and one Club's attack failing to mix is enough to make a table of title
    probabilities wrong.
    """
    import arviz as az

    summary = az.summary(trace, var_names=list(PARAMETERS), round_to=None)
    divergences = trace.sample_stats["diverging"]
    return Diagnostics(
        divergences=int(divergences.sum()),
        max_r_hat=float(summary["r_hat"].max()),
        min_ess_bulk=float(summary["ess_bulk"].min()),
        min_ess_tail=float(summary["ess_tail"].min()),
        draws=sampling.draws,
        tune=sampling.tune,
        chains=sampling.chains,
        seed=sampling.seed,
    )


__all__ = [
    "Diagnostics",
    "Posterior",
    "Priors",
    "Sampling",
    "fit",
    "log_likelihood_at",
    "model_for",
]
