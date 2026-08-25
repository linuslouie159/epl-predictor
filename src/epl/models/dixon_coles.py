"""Dixon-Coles by maximum likelihood, refitted at every Prediction Round.

The second real model, and the first that predicts *goals*. Elo turns a Club's history into one
number and an ordered logit turns that number into three probabilities; this turns the same history
into an attack and a defence per Club and produces the probability of every Scoreline, which
collapses onto the same three (:func:`epl.models.likelihood.outcomes`).

The likelihood itself is not here. It is in :mod:`epl.models.likelihood`, because ADR 0007 fits this
model two ways — maximum likelihood at all 952 scored Prediction Rounds, a full posterior only where
a Season Projection is produced (issue #14) — and one shared likelihood is what stops the two paths
from drifting into two models. What is here is the *fit*: the optimiser, the frozen decay, and the
Predictor that wraps them.

**It rates the whole pyramid, for the reason Elo does** (ADR 0004): a Club promoted into the Premier
League must arrive with an attack and a defence it earned, and a Premier-League-only fit has nothing
to say about a Club that has not played in it. There is one difference from Elo worth understanding.
Elo is zero-sum, so a rating carried across a promotion is comparable by construction. This model
has no such guarantee — no Club ever plays outside its own tier, so the four tiers are connected
*only* by the Clubs that changed tier inside the decay horizon. Three promoted and three relegated
Clubs a Season at every boundary is what makes attack and defence comparable across tiers at all,
and it is why the horizon has to reach back years rather than months.

**It refits from cold at every round rather than folding a fit forward**, exactly as
:class:`epl.models.elo.Elo` does and for the same reason: a fit carried between calls would have to
judge whether the Evidence it has just been handed extends the one it fitted last, and getting that
wrong is invisible — the strengths would be built from the wrong matches while every stored row
still audited clean. A fit takes about 300 ms — roughly 230 parameters over 12,000
weighted matches — so a full walk over the Evaluation Window's 952 rounds costs five minutes, which
is the acceptance criterion issue #13 states as "minutes rather than hours".

Two things that look like defects and are not:

* **The low-score correction has all but vanished, and is still fitted.** Dixon and Coles found
  about -0.13 in four Seasons of one division in the early 1990s; over this corpus it wanders
  around zero — -0.058 at the first scored round, +0.003 in 2015/16, -0.010 in 2025/26 — and
  pinning it at zero, which is two independent Poissons, costs 0.00011 RPS on the Burn-In Window.
  It stays because it is a parameter of the likelihood the Bayesian fit shares rather than a
  hyperparameter, so dropping it would change the model rather than simplify the code, and because
  0.00011 in the right direction is a measurement rather than nothing. What it is *not* is the
  reason this model beats Elo. See docs/DECISIONS.md.
* **Nothing here regresses a Club to the mean between Seasons, and nothing carries a prior.** The
  time decay is the only thing that forgets, which is Dixon-Coles as published; a Club that stopped
  playing simply keeps the strengths its last matches earned, with less and less weight behind them.

    python -m epl.models fit          re-derive the frozen half-life on the Burn-In Window
    python -m epl.models strengths    the attack and defence table at a Season's first round
    python -m epl.models sequential   what predicting per Fixture instead of per round would buy
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize

from epl.models.likelihood import (
    MAX_GOALS,
    Decay,
    Sample,
    Strengths,
    bounds,
    negative_log_likelihood,
    outcomes,
    pack,
    scorelines,
    start,
    unpack,
)
from epl.models.ordered_logit import ModelError
from epl.predictors import Evidence, register

#: The tiers folded into the fit. All four, for the reason ADR 0004 gives — see the module
#: docstring, and note that the argument is not Elo's. Measured on the Burn-In Window: fitting the
#: Premier League alone scores worse there, and the gap is the promoted Clubs (docs/DECISIONS.md).
FITTED_DIVISIONS: tuple[str, ...] = ("E0", "E1", "E2", "E3")

#: Fitted on Seasons 2001/02-2004/05 and frozen (ADR 0008) by ``python -m epl.models fit``, on the
#: same protocol as :data:`epl.models.elo.FROZEN_SETTINGS`: 2000/01 warms the sample up and is not
#: fitted on, and the search is a grid because a half-life in days is a quantity a reader can argue
#: with.
#:
#: A literal rather than a fit at import, so a rebuilt backtest is byte-identical to the last one
#: (ADR 0005) and no Prediction depends on whether ``data/raw/`` happens to be populated.
#: ``tests/models/test_dixon_coles_over_the_corpus.py`` re-runs the fit and checks this is still
#: what it finds.
#:
#: Ten and a half months. The curve around it is nearly flat — anything from 270 to 480 days scores
#: within 0.0001 RPS on the Burn-In Window — so this is a well-determined *region* rather than a
#: well-determined number, which is the honest way to read any half-life. What is sharply excluded
#: is the short end: 60 days costs 0.007 RPS, because a fit that remembers two months is fitting
#: form rather than strength.
FROZEN_DECAY = Decay(half_life_days=322.5)

#: L-BFGS-B's convergence tolerances. Stated rather than defaulted for the reason
#: :data:`epl.models.burn_in.LOGIT_TOLERANCE` is: a scipy release that moved a default would move
#: every number this project publishes, quietly. Both are far tighter than the five decimal places
#: anything here is reported to, and measured over 136 sampled rounds the *objective* tolerance is
#: what terminates every fit — loosening the gradient tolerance from 1e-7 to 1e-6 does not change a
#: single iteration count.
GRADIENT_TOLERANCE = 1e-7
OBJECTIVE_TOLERANCE = 1e-12

#: The ceiling on iterations, and headroom rather than a target.
#:
#: Set from measurement: over 136 sampled Prediction Rounds a fit takes a mean of 251 iterations
#: and a worst of 1,454, and the tail is long enough that one of the diagnostic's 3,130 per-kickoff
#: cuts needed more than 2,000. A ceiling near the observed worst case turns a slow fit into a
#: failed run, so this sits several times above it — and a fit that genuinely cannot converge still
#: raises rather than returning what the optimiser had reached.
MAX_ITERATIONS = 10_000


def fit(
    sample: Sample,
    *,
    correction: float | None = None,
    max_iterations: int = MAX_ITERATIONS,
) -> Strengths:
    """The maximum-likelihood :class:`~epl.models.likelihood.Strengths` for one weighted sample.

    L-BFGS-B on the analytic gradient. Quasi-Newton because the problem is smooth and has a few
    hundred parameters, bounded because a Club with one match in the sample has a likelihood that
    improves forever as its attack falls (:data:`epl.models.likelihood.STRENGTH_BOUND`), and started
    from :func:`epl.models.likelihood.start` — the sample's own scoring rates with every Club
    average — so the search begins somewhere the correction is guaranteed feasible.

    Returns the fit in a gauge (:meth:`~epl.models.likelihood.Strengths.centred`). Without that the
    numbers are still a valid model and are not comparable to any other fit's, because the
    likelihood is flat along the direction that adds a constant to every attack and every defence.

    ``correction`` holds Dixon-Coles' low-score parameter at a stated value instead of fitting it,
    which is how what the correction is worth gets measured rather than assumed — hold it at zero
    and the model is two independent Poissons. Nothing predicts with it set; see
    :func:`epl.models.likelihood.bounds`.

    A fit that does not converge raises rather than returning what the optimiser had reached. A
    silently unconverged fit is a Prediction built from the wrong strengths, and it would look
    exactly like a converged one on the scoreboard.
    """
    if not sample.club_count:
        return start(sample)

    opening = start(sample)
    if correction is not None:
        opening = Strengths(
            opening.clubs,
            opening.attack,
            opening.defence,
            opening.home_advantage,
            correction,
        )

    found = minimize(
        negative_log_likelihood,
        pack(opening),
        args=(sample,),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds(sample.club_count, correction=correction),
        options={
            "maxiter": max_iterations,
            "maxfun": max_iterations * 2,
            "ftol": OBJECTIVE_TOLERANCE,
            "gtol": GRADIENT_TOLERANCE,
        },
    )
    if not found.success:
        raise ModelError(
            f"the Dixon-Coles fit did not converge over {len(sample)} matches and "
            f"{sample.club_count} Clubs: {found.message}"
        )
    return unpack(np.asarray(found.x, dtype=np.float64), sample.clubs).centred()


class DixonColes:
    """The goals model, registered as a Predictor.

    Handed one Prediction Round's Fixtures and the Evidence visible at its As-Of Instant, it fits
    attack and defence for every Club in :attr:`divisions` over everything that Evidence holds
    within the decay horizon, produces the probability of every Scoreline up to
    :data:`epl.models.likelihood.MAX_GOALS`, and collapses those onto (Home, Draw, Away).

    Stateless between rounds, and one instance serves the whole scoreboard — see the module
    docstring on why the fit is not carried forward.
    """

    def __init__(
        self,
        decay: Decay = FROZEN_DECAY,
        divisions: Sequence[str] = FITTED_DIVISIONS,
        *,
        name: str = "dixon_coles",
        max_goals: int = MAX_GOALS,
        correction: float | None = None,
    ) -> None:
        self.name = name
        self.decay = decay
        self.divisions = tuple(divisions)
        self.max_goals = max_goals
        #: Pins the low-score parameter instead of fitting it — ``0.0`` is two independent
        #: Poissons, which is Dixon-Coles with the Dixon-Coles taken out. The registered instance
        #: leaves it ``None``; it exists so "what is the correction worth on this corpus?" can be
        #: answered by walking the comparison rather than by reasoning about the fitted value.
        self.correction = correction

    def sample_at(self, evidence: Evidence, *, also: Iterable[str] = ()) -> Sample:
        """The weighted matches this model would fit on at this Evidence's As-Of Instant.

        Public because it is the receipt for the two claims the fit rests on: which tiers reached
        the sample, and how far back the decay let it look. Asked through Evidence rather than the
        corpus, so even a diagnostic stays on the right side of the project's one rule.
        """
        return Sample.of(
            evidence.matches(divisions=self.divisions),
            evidence.as_of,
            self.decay,
            also=also,
        )

    def strengths_at(self, evidence: Evidence, *, also: Iterable[str] = ()) -> Strengths:
        """The fitted attack and defence per Club as they stood at this As-Of Instant.

        The model's own view of the pyramid, and the counterpart of
        :meth:`epl.models.elo.Elo.ratings_at`. ``python -m epl.models strengths`` prints it.
        """
        return fit(self.sample_at(evidence, also=also), correction=self.correction)

    def scorelines(self, fixtures: pd.DataFrame, evidence: Evidence) -> npt.NDArray[np.float64]:
        """The probability of every Scoreline, per Fixture: ``(n, max_goals + 1, max_goals + 1)``.

        The Season Projection needs this and the Outcome does not carry it — 24 of 26 Seasons had a
        points tie that goal difference broke (ADR 0007), so a table cannot be simulated from three
        probabilities. :meth:`predict` is this, summed.
        """
        if not len(fixtures):
            size = self.max_goals + 1
            return np.empty((0, size, size), dtype=np.float64)

        home_clubs = fixtures["home_club"].to_numpy(dtype=object)
        away_clubs = fixtures["away_club"].to_numpy(dtype=object)
        sample = self.sample_at(evidence, also=[*home_clubs, *away_clubs])
        strengths = fit(sample, correction=self.correction)
        home_rate, away_rate = strengths.rates(
            sample.index_of(home_clubs), sample.index_of(away_clubs)
        )
        return scorelines(
            home_rate, away_rate, strengths.correction, max_goals=self.max_goals
        )

    def predict(
        self, fixtures: pd.DataFrame, evidence: Evidence
    ) -> npt.NDArray[np.float64]:
        """One Prediction per Fixture, from strengths fitted strictly before the As-Of Instant."""
        return outcomes(self.scorelines(fixtures, evidence))


#: The registered instance, on the frozen decay and the whole pyramid. One instance serves the
#: whole scoreboard because it holds no state between rounds — see :class:`DixonColes`.
DIXON_COLES = register(DixonColes())
