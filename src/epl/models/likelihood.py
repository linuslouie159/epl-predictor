"""One Dixon-Coles likelihood, written once for both fits.

ADR 0007 fits this model two ways — maximum likelihood at every Prediction Round, a full posterior
only where a Season Projection is produced — and says in as many words that "both paths share one
likelihood function, so the models cannot drift apart". This module is that function, and the
weighted sample it is taken over. :mod:`epl.models.dixon_coles` is the MLE half; :mod:`epl.simulate`
(issue #14) will be the Bayesian half, and it imports from here rather than restating anything.

Nothing here fits, predicts, or knows what a Predictor is. It is the model written down::

    lambda = exp(attack[home] - defence[away] + home_advantage)      goals the home Club scores
    mu     = exp(attack[away] - defence[home])                       goals the away Club scores

two independent Poissons, corrected on the four low-scoring Scorelines by
:func:`low_score_factor` — the one part of Dixon-Coles that is not a Poisson, and the reason 0-0
and 1-1 are more common and 1-0 and 0-1 less common than independence predicts.

Three properties of that arithmetic are worth stating before anyone changes it:

* **The correction moves probability without creating any.** Summed over every Scoreline the four
  adjustments cancel exactly, so a corrected Scoreline distribution still sums to one and
  :func:`outcomes` needs no rescaling beyond the truncation at :data:`MAX_GOALS`.
* **The parameters are identified only up to a shift.** Adding a constant to every attack *and*
  every defence leaves every rate unchanged, so a fit has one flat direction and a fitted
  :class:`Strengths` is meaningless until it is put in a gauge. :meth:`Strengths.centred` is that
  gauge — attack averaging zero — and it is applied once, at the end of a fit.
* **A Club with no matches in the sample keeps the neutral strengths it started at.** That is not a
  guess dressed as a fit: it is what zero weight means, and the alternative — dropping the Club —
  would make a Fixture unpredictable rather than uncertain. Over the corpus this reaches only the
  clubs entering League Two from outside the Football League, four tiers from anything scored.

The time decay is here rather than in the fit because both fits weight the same way (ADR 0007) and
because a sample is not a sample until it is weighted. Its half-life is fitted in the Burn-In Window
like every other hyperparameter (ADR 0008, :mod:`epl.models.burn_in`) and frozen in
:mod:`epl.models.dixon_coles`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from epl.metrics import OUTCOMES
from epl.models.ordered_logit import ModelError
from epl.rounds import kickoff_instants

#: The largest Scoreline :func:`scorelines` enumerates, per Club. Sixteen values each way is 256
#: Scorelines per Fixture, and the mass past it is below 1e-9 for any rate this model produces —
#: far inside :data:`epl.metrics.SUM_TOLERANCE`. It is a truncation, not a modelling claim, and
#: :func:`outcomes` renormalises over what is enumerated so the truncated tail cannot reach the
#: ledger as a Prediction that fails to sum to one.
MAX_GOALS = 15

#: How far a strength may travel from neutral, in log-goals. A numerical guard rather than a fitted
#: bound: exp(5) is 148 goals a match, so no sample this project holds can push a Club against it —
#: but a Club with one match and no goals has a likelihood that improves forever as its attack
#: falls, and an unbounded search would follow it to negative infinity and take the optimiser's
#: line search with it.
STRENGTH_BOUND = 5.0

#: How far the low-score correction may travel. Dixon and Coles fitted about -0.13 on English
#: league football, and the sign matters: negative is what lifts 0-0 and 1-1 and lowers 1-0 and 0-1.
#: The bound is wide enough that the fitted value sits well inside it and narrow enough that the
#: correction cannot turn a Scoreline probability negative at any rate the model plausibly quotes.
CORRECTION_BOUND = 0.5

#: The four Scorelines Dixon-Coles corrects, and the only ones :func:`low_score_factor` leaves
#: changed. Named once because two places enumerate them — the likelihood, over the matches that
#: finished that way, and :func:`scorelines`, over the four cells of every Fixture's grid.
LOW_SCORES: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))

#: What :func:`low_score_factor` is floored at before a logarithm is taken.
#:
#: The correction multiplies four Scorelines and is positive at every parameter value a fit
#: settles on, because the log-likelihood of an observed 0-0 goes to minus infinity as its own
#: correction approaches zero — the likelihood defends the region it needs. The floor is for the
#: search's way *past* that region, in the same spirit as the overflow guard in
#: :func:`epl.models.ordered_logit.sigmoid`: a numerical warning that is merely harmless today is
#: what hides the one that is not tomorrow. ``tests/models/test_dixon_coles_over_the_corpus.py``
#: checks that no fit over the real corpus ever comes near it.
CORRECTION_FLOOR = 1e-12


@dataclass(frozen=True)
class Decay:
    """How much less a match counts for every day it recedes into the past.

    Dixon-Coles' one time-dependent idea: a match played last month says more about a Club than one
    played three years ago, so the likelihood is weighted rather than the sample truncated by hand.

    Stated as a **half-life in days** rather than as the exponent it becomes, because a half-life is
    a number a reader can argue with — "a match counts half as much a year later" is a claim about
    football, where "xi = 0.0019" is a claim about nothing. It is fitted in the Burn-In Window and
    frozen (ADR 0008); :data:`epl.models.dixon_coles.FROZEN_DECAY` is where the answer lives.

    ``floor`` is the weight below which a match is dropped from the sample entirely. That is a
    tolerance, not a hyperparameter: the dropped matches are the ones the weighting has already
    decided count for almost nothing, and what it buys is a fit over three years of football rather
    than over twenty-six. :attr:`horizon` is where it bites.
    """

    half_life_days: float
    floor: float = 0.01

    def __post_init__(self) -> None:
        if not self.half_life_days > 0:
            raise ModelError(
                f"a half-life must be positive; got {self.half_life_days}. A half-life of zero is "
                "a model that has never seen a match"
            )
        if not 0.0 < self.floor < 1.0:
            raise ModelError(
                f"the weight floor must sit strictly between 0 and 1; got {self.floor}"
            )

    @property
    def horizon(self) -> float:
        """How many days back the sample reaches: where a match's weight falls to :attr:`floor`."""
        return float(self.half_life_days * np.log2(1.0 / self.floor))

    def weights(self, days_before: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """What each match counts for, given how many days before the As-Of Instant it was played.

        One at the instant itself, a half after one half-life, and so on. Days are taken as a float
        so a match played this morning is not rounded to the same weight as one played last night.
        """
        days = np.asarray(days_before, dtype=np.float64).ravel()
        return np.asarray(0.5 ** (days / self.half_life_days), dtype=np.float64)


@dataclass(frozen=True)
class Sample:
    """The weighted matches one fit is taken over, with Clubs already reduced to indices.

    Built by :meth:`of`, which is where the decay and the Club indexing happen. Holding indices
    rather than names is what lets the gradient be four :func:`numpy.bincount` calls instead of a
    loop over matches — this is evaluated a few hundred times per fit and a fit happens at every
    one of the Evaluation Window's 952 Prediction Rounds.

    ``clubs`` is the index: position *i* in :attr:`Strengths.attack` is ``clubs[i]``. It carries
    every Club in the sample **and** every Club named in the Fixtures being predicted, so a
    Prediction never has to ask about a Club the parameter vector has no slot for.
    """

    clubs: tuple[str, ...]
    #: Where each Club sits in the parameter vector — the inverse of :attr:`clubs`, built once
    #: with it rather than rebuilt per lookup. :meth:`index_of` is asked twice per Prediction
    #: Round and the sample holds a hundred Clubs, so "once" is worth stating.
    position: dict[str, int]
    home: npt.NDArray[np.intp]
    away: npt.NDArray[np.intp]
    home_goals: npt.NDArray[np.float64]
    away_goals: npt.NDArray[np.float64]
    weight: npt.NDArray[np.float64]

    @classmethod
    def of(
        cls,
        matches: pd.DataFrame,
        as_of: pd.Timestamp,
        decay: Decay,
        *,
        also: Iterable[str] = (),
    ) -> Sample:
        """The matches within ``decay``'s horizon of ``as_of``, weighted by how long ago they were.

        ``matches`` is what :class:`epl.predictors.Evidence` handed over — already cut at the
        As-Of Instant, so this does not re-apply the project's one rule so much as refuse to be the
        place it is broken: a match at or after ``as_of`` would take a weight above one and is an
        error here rather than a quietly over-counted row.

        ``also`` names Clubs that must have a slot in the parameter vector whether or not they
        appear in the sample — the Clubs of the Fixtures about to be predicted.
        """
        instant = pd.Timestamp(as_of)
        kickoffs = (
            kickoff_instants(matches).to_numpy()
            if len(matches)
            else np.empty(0, dtype="datetime64[ns]")
        )
        days = (instant.to_numpy() - kickoffs) / np.timedelta64(1, "D")
        if len(days) and days.min() <= 0:
            raise ModelError(
                f"{int((days <= 0).sum())} of {len(days)} matches kicked off at or after "
                f"{instant}, which is the As-Of Instant they are being weighted against. Evidence "
                "is what cuts the corpus (epl.predictors); a sample should never have to"
            )

        kept = days <= decay.horizon
        within = matches.loc[kept]
        clubs, position, home, away = _index(
            within["home_club"].to_numpy(dtype=object),
            within["away_club"].to_numpy(dtype=object),
            also,
        )
        return cls(
            clubs=clubs,
            position=position,
            home=home,
            away=away,
            home_goals=within["home_goals"].to_numpy(dtype=np.float64),
            away_goals=within["away_goals"].to_numpy(dtype=np.float64),
            weight=decay.weights(days[kept]),
        )

    def __len__(self) -> int:
        return len(self.home)

    @property
    def club_count(self) -> int:
        return len(self.clubs)

    def index_of(self, clubs: Sequence[str] | npt.NDArray[np.object_]) -> npt.NDArray[np.intp]:
        """Where each named Club sits in the parameter vector, refusing one that has no slot."""
        unknown = sorted({club for club in clubs if club not in self.position})
        if unknown:
            raise ModelError(
                f"{len(unknown)} Clubs have no slot in this sample: {unknown[:5]}. Name them in "
                "`also` when the Sample is built, so a Fixture is uncertain and not unanswerable"
            )
        return np.asarray([self.position[club] for club in clubs], dtype=np.intp)


def _index(
    home: npt.NDArray[np.object_], away: npt.NDArray[np.object_], also: Iterable[str]
) -> tuple[tuple[str, ...], dict[str, int], npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Every Club in the sample and in ``also``, in name order, with the two match columns coded.

    Name order rather than order of appearance, so the same matches always produce the same
    parameter vector and a rebuilt backtest is byte-identical to the last one (ADR 0005).

    The returned mapping is the only Club-to-position table in this module: the bulk coding below
    and :meth:`Sample.index_of` both read it, so a Fixture's Club and a match's Club cannot come to
    mean different slots.
    """
    clubs = tuple(sorted(set(home.tolist()) | set(away.tolist()) | set(also)))
    position = {club: index for index, club in enumerate(clubs)}
    if not clubs:
        empty = np.empty(0, dtype=np.intp)
        return clubs, position, empty, empty

    positions = pd.Series(position, dtype=np.intp)
    return (
        clubs,
        position,
        np.asarray(positions.reindex(home).to_numpy(), dtype=np.intp),
        np.asarray(positions.reindex(away).to_numpy(), dtype=np.intp),
    )


@dataclass(frozen=True)
class Strengths:
    """One attack and one defence per Club, what playing at home is worth, and the correction.

    All four in log-goals except :attr:`correction`, which is the Dixon-Coles low-score parameter
    and unitless. Frozen for the reason :class:`epl.models.elo.Settings` is: a fitted parameter that
    can be reassigned after the fact is one that can be tuned outside the Burn-In Window without
    anything looking wrong (ADR 0008).

    Meaningless until it is put in a gauge — see :meth:`centred` and the module docstring.
    """

    clubs: tuple[str, ...]
    attack: npt.NDArray[np.float64]
    defence: npt.NDArray[np.float64]
    home_advantage: float
    correction: float

    def __post_init__(self) -> None:
        if not len(self.attack) == len(self.defence) == len(self.clubs):
            raise ModelError(
                f"{len(self.clubs)} Clubs, {len(self.attack)} attacks and "
                f"{len(self.defence)} defences; the three are one per Club"
            )

    def centred(self) -> Strengths:
        """The same model, with attack averaging zero.

        Adding a constant to every attack *and* every defence leaves every rate untouched, so a fit
        returns one arbitrary point on a flat line. This picks the point on it that can be read: an
        attack of zero is an average Club, and a defence of zero is one that concedes what an
        average Club scores against an average defence.
        """
        if not len(self.clubs):
            return self
        shift = float(self.attack.mean())
        return Strengths(
            clubs=self.clubs,
            attack=self.attack - shift,
            defence=self.defence - shift,
            home_advantage=self.home_advantage,
            correction=self.correction,
        )

    def rates(
        self, home: npt.NDArray[np.intp], away: npt.NDArray[np.intp]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The goals each Club is expected to score, given who is playing and who is at home."""
        home_rate = np.exp(self.attack[home] - self.defence[away] + self.home_advantage)
        away_rate = np.exp(self.attack[away] - self.defence[home])
        return home_rate, away_rate

    def outcomes_for(
        self, home: npt.NDArray[np.intp], away: npt.NDArray[np.intp]
    ) -> npt.NDArray[np.float64]:
        """(Home, Draw, Away) per Fixture, from these strengths — rates, Scorelines, collapsed.

        The three steps in one place because they are always taken together and always in this
        order, and because a caller that assembles them by hand is a caller that can assemble them
        differently. :meth:`rates` and :func:`scorelines` remain public for the paths that want the
        goals rather than the Outcome — the Season Projection needs a Scoreline grid, not this.
        """
        home_rate, away_rate = self.rates(home, away)
        return outcomes(scorelines(home_rate, away_rate, self.correction))

    def table(self) -> pd.DataFrame:
        """One row per Club, strongest attack first. The model's own view of the pyramid."""
        frame = pd.DataFrame(
            {"club": list(self.clubs), "attack": self.attack, "defence": self.defence}
        )
        return frame.sort_values("attack", ascending=False, kind="stable").reset_index(drop=True)


def start(sample: Sample) -> Strengths:
    """Where a fit begins: every Club average, at the sample's own scoring rates.

    Not zeros. With every strength at zero the away rate is one goal a match, so the search spends
    its first iterations discovering the mean of the very column it was handed. Setting the level
    from the sample costs one weighted mean and is deterministic given the sample, so it does not
    make a rebuild differ from the last one (ADR 0005).

    The correction starts at zero — two independent Poissons — which is the value that makes every
    Scoreline probability positive whatever the rates are. A line search that only accepts a step
    that improves the likelihood therefore starts inside the feasible region and stays there.
    """
    neutral = np.zeros(sample.club_count, dtype=np.float64)
    if not len(sample) or sample.weight.sum() <= 0:
        return Strengths(sample.clubs, neutral, neutral.copy(), 0.0, 0.0)

    total = float(sample.weight.sum())
    home_goals = max(float((sample.weight * sample.home_goals).sum()) / total, 1e-3)
    away_goals = max(float((sample.weight * sample.away_goals).sum()) / total, 1e-3)
    level = 0.5 * float(np.log(away_goals))
    return Strengths(
        clubs=sample.clubs,
        attack=neutral + level,
        defence=neutral - level,
        home_advantage=float(np.log(home_goals / away_goals)),
        correction=0.0,
    )


def low_score_factor(
    home_goals: npt.NDArray[np.float64],
    away_goals: npt.NDArray[np.float64],
    home_rate: npt.NDArray[np.float64],
    away_rate: npt.NDArray[np.float64],
    rho: float,
) -> npt.NDArray[np.float64]:
    """Dixon-Coles' low-score correction, tau — the factor that is not a Poisson.

    Two independent Poissons get the tail of the Scoreline distribution right and the bottom-left
    corner wrong: real football produces more 0-0 and 1-1 draws and fewer 1-0 and 0-1 wins than
    independence allows. Four Scorelines are therefore multiplied by::

        tau(0, 0) = 1 - lambda * mu * rho     tau(0, 1) = 1 + lambda * rho
        tau(1, 0) = 1 + mu * rho              tau(1, 1) = 1 - rho

    and every other Scoreline by one. A negative ``rho`` is what lifts the two draws and lowers the
    two one-goal wins, which is the direction the data has always given.

    Written once and called from both places it is needed — the likelihood and the Scoreline grid —
    so the fit and what is predicted from it cannot come to mean different things.
    """
    factor, _, _, _ = _correction_and_slopes(home_goals, away_goals, home_rate, away_rate, rho)
    return factor


def _correction_and_slopes(
    home_goals: npt.NDArray[np.float64],
    away_goals: npt.NDArray[np.float64],
    home_rate: npt.NDArray[np.float64],
    away_rate: npt.NDArray[np.float64],
    rho: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """:func:`low_score_factor`, and its derivatives with respect to log lambda, log mu and rho.

    Together rather than in two functions, because the four cases are the thing most likely to be
    edited and a derivative that had drifted from its own function is a bug the fit would absorb
    silently — a slightly wrong gradient converges slightly wrong.
    """
    factor = np.ones_like(home_rate)
    d_log_home = np.zeros_like(home_rate)
    d_log_away = np.zeros_like(home_rate)
    d_rho = np.zeros_like(home_rate)

    nil_nil = (home_goals == 0) & (away_goals == 0)
    product = home_rate[nil_nil] * away_rate[nil_nil]
    factor[nil_nil] = 1.0 - product * rho
    d_log_home[nil_nil] = -product * rho
    d_log_away[nil_nil] = -product * rho
    d_rho[nil_nil] = -product

    nil_one = (home_goals == 0) & (away_goals == 1)
    factor[nil_one] = 1.0 + home_rate[nil_one] * rho
    d_log_home[nil_one] = home_rate[nil_one] * rho
    d_rho[nil_one] = home_rate[nil_one]

    one_nil = (home_goals == 1) & (away_goals == 0)
    factor[one_nil] = 1.0 + away_rate[one_nil] * rho
    d_log_away[one_nil] = away_rate[one_nil] * rho
    d_rho[one_nil] = away_rate[one_nil]

    one_one = (home_goals == 1) & (away_goals == 1)
    factor[one_one] = 1.0 - rho
    d_rho[one_one] = -1.0

    return factor, d_log_home, d_log_away, d_rho


def pack(strengths: Strengths) -> npt.NDArray[np.float64]:
    """A :class:`Strengths` as the flat vector an optimiser searches over."""
    return np.concatenate(
        [
            strengths.attack,
            strengths.defence,
            [strengths.home_advantage, strengths.correction],
        ]
    )


def unpack(free: npt.NDArray[np.float64], clubs: tuple[str, ...]) -> Strengths:
    """The inverse of :func:`pack`."""
    count = len(clubs)
    return Strengths(
        clubs=clubs,
        attack=np.asarray(free[:count], dtype=np.float64),
        defence=np.asarray(free[count : 2 * count], dtype=np.float64),
        home_advantage=float(free[2 * count]),
        correction=float(free[2 * count + 1]),
    )


def bounds(club_count: int, *, correction: float | None = None) -> list[tuple[float, float]]:
    """The box the search runs in — :data:`STRENGTH_BOUND` and :data:`CORRECTION_BOUND`.

    Home advantage shares the strengths' bound: it is the same kind of quantity, log-goals, and the
    fitted value is about 0.25.

    ``correction`` pins the low-score parameter instead of fitting it, by giving it a box of zero
    width. That is how "what is Dixon-Coles' correction actually worth on this corpus?" is asked —
    fit with it held at zero, which is two independent Poissons, and compare. It is a measurement
    rather than a mode: nothing predicts with a pinned correction.
    """
    return [
        *[(-STRENGTH_BOUND, STRENGTH_BOUND)] * (2 * club_count),
        (-STRENGTH_BOUND, STRENGTH_BOUND),
        (-CORRECTION_BOUND, CORRECTION_BOUND) if correction is None else (correction, correction),
    ]


def negative_log_likelihood(
    free: npt.NDArray[np.float64], sample: Sample
) -> tuple[float, npt.NDArray[np.float64]]:
    """What the optimiser minimises, and its gradient — the weighted Dixon-Coles log-likelihood.

    Negative because optimisers minimise, weighted because recent matches say more (:class:`Decay`),
    and dropped of the factorial terms because they do not depend on any parameter and a fit is not
    a likelihood-ratio test. **Do not compare the value against another implementation's without
    adding them back.**

    The gradient is analytic rather than differenced. Two hundred parameters differenced is two
    hundred extra evaluations per step, which is the difference between a backtest that takes
    minutes and one that takes a day — the acceptance criterion issue #13 states in those words.
    ``tests/models/test_likelihood.py`` checks it against central differences, which is the only
    honest way to keep an analytic gradient true.
    """
    strengths = unpack(free, sample.clubs)
    home_rate, away_rate = strengths.rates(sample.home, sample.away)

    factor, d_log_home, d_log_away, d_rho = _correction_and_slopes(
        sample.home_goals, sample.away_goals, home_rate, away_rate, strengths.correction
    )
    safe = np.maximum(factor, CORRECTION_FLOOR)

    log_home = np.log(home_rate)
    log_away = np.log(away_rate)
    per_match = (
        sample.home_goals * log_home
        - home_rate
        + sample.away_goals * log_away
        - away_rate
        + np.log(safe)
    )
    value = -float((sample.weight * per_match).sum())

    inverse = 1.0 / safe
    by_home_rate = sample.weight * ((sample.home_goals - home_rate) + inverse * d_log_home)
    by_away_rate = sample.weight * ((sample.away_goals - away_rate) + inverse * d_log_away)
    by_correction = float((sample.weight * inverse * d_rho).sum())

    count = sample.club_count
    gradient = np.empty(2 * count + 2, dtype=np.float64)
    # A Club's attack appears in the home rate of its home matches and the away rate of its away
    # ones; its defence appears, negated, in the other two. `bincount` is the sum over each Club's
    # matches without ever grouping the frame.
    gradient[:count] = np.bincount(sample.home, by_home_rate, count) + np.bincount(
        sample.away, by_away_rate, count
    )
    gradient[count : 2 * count] = -np.bincount(sample.away, by_home_rate, count) - np.bincount(
        sample.home, by_away_rate, count
    )
    gradient[2 * count] = by_home_rate.sum()
    gradient[2 * count + 1] = by_correction
    return value, -gradient


def scorelines(
    home_rate: npt.NDArray[np.float64],
    away_rate: npt.NDArray[np.float64],
    rho: float,
    *,
    max_goals: int = MAX_GOALS,
) -> npt.NDArray[np.float64]:
    """The probability of every Scoreline, as an ``(n, max_goals + 1, max_goals + 1)`` array.

    Indexed ``[fixture, home goals, away goals]``, so ``grid[0, 2, 1]`` is the chance the first
    Fixture finishes 2-1. This is what the Season Projection needs and the Outcome does not carry:
    "A Scoreline implies an Outcome; an Outcome does not imply a Scoreline" (CONTEXT.md).

    Normalised over the truncated grid. The correction itself moves no mass — its four adjustments
    cancel exactly — so what is being renormalised away is only the tail past :data:`MAX_GOALS`.
    """
    counts = np.arange(max_goals + 1, dtype=np.float64)
    log_factorial = np.cumsum(np.concatenate([[0.0], np.log(counts[1:])]))

    def poisson(rate: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        log_rate = np.log(rate)[:, None]
        return np.exp(-rate[:, None] + counts[None, :] * log_rate - log_factorial[None, :])

    grid = poisson(home_rate)[:, :, None] * poisson(away_rate)[:, None, :]
    for goals_home, goals_away in LOW_SCORES:
        if goals_home > max_goals or goals_away > max_goals:
            continue  # pragma: no cover - only reachable at a grid too small to be a Scoreline
        grid[:, goals_home, goals_away] *= low_score_factor(
            np.full(len(home_rate), float(goals_home)),
            np.full(len(home_rate), float(goals_away)),
            home_rate,
            away_rate,
            rho,
        )

    # Clipped before normalising rather than trusted: `rho` is bounded but the rates are not, so a
    # Fixture between a very strong attack and a very weak defence could in principle drive
    # `1 + lambda * rho` below zero. It does not happen over this corpus — a test says so — and a
    # negative probability reaching the ledger would be a worse way to find out.
    grid = np.maximum(grid, 0.0)
    return np.asarray(grid / grid.sum(axis=(1, 2), keepdims=True), dtype=np.float64)


def outcomes(grid: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """A Scoreline distribution collapsed onto (Home, Draw, Away).

    The Outcome is a partition of the Scorelines — above the diagonal, on it, below it — so this is
    a sum and never a second model. Issue #13 asks for exactly this: Scoreline probabilities
    produced, then collapsed, so the goals model reaches the same scoreboard as everything else.
    """
    if not len(grid):
        return np.empty((0, len(OUTCOMES)), dtype=np.float64)
    size = grid.shape[1]
    home_goals = np.arange(size)[:, None]
    away_goals = np.arange(size)[None, :]
    return np.column_stack(
        [
            (grid * (home_goals > away_goals)).sum(axis=(1, 2)),
            (grid * (home_goals == away_goals)).sum(axis=(1, 2)),
            (grid * (home_goals < away_goals)).sum(axis=(1, 2)),
        ]
    )
