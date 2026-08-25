"""Simulate: the Bayesian fit and the Monte Carlo Season Projection.

**The posterior is built** (issue #14): :mod:`epl.simulate.posterior` samples Dixon-Coles rather
than maximising it, and :mod:`epl.simulate.checkpoints` decides the handful of Prediction Rounds it
is allowed to run at. The Monte Carlo Season Projection over those draws is issue #15 and is not
built yet.

A Season Projection is a distribution over final league tables, produced by simulating every
remaining Fixture many times: 10,000 simulated Seasons per projection, with a deterministic seed
recorded in the output so any published number can be reproduced exactly.

**There is one likelihood, and #14 did not write a second one.** ADR 0007's whole argument for
fitting one model two ways is that "both paths share one likelihood function, so the models cannot
drift apart", and stage 9 put that function in :mod:`epl.models.likelihood` — the rates, the
low-score correction, the weighted sample and the Scoreline grid, with nothing in it that knows what
an optimiser or a Predictor is. The posterior is fitted over the same
:class:`epl.models.likelihood.Sample` and returns draws of the same
:class:`epl.models.likelihood.Strengths`; :func:`epl.models.dixon_coles.fit` is the MLE path beside
it, and the two are readable side by side. What makes that structural rather than a promise is that
the sampler is handed the numpy function itself as one opaque node, gradient and all, rather than a
PyTensor re-expression of it — see :mod:`epl.simulate.posterior`.

Team strengths are **drawn from the posterior rather than fixed at point estimates** (ADR 0007).
With ~10,000 observations and ~50 parameters, parameter uncertainty barely moves any single
Fixture's probability — but it compounds across 380 simulated Fixtures into a final table, and
ignoring it is what makes naive season simulators report a title probability of 48% where the honest
answer is 34%.

Within-Season strength drift is deliberately **not** modelled: across 520 club-Seasons, observed
first-half to second-half variation was indistinguishable from sampling noise.

The projection needs goals, not just Outcomes, which is why it comes after Dixon-Coles: points ties
occurred in 24 of 26 Seasons. The full tiebreaker chain runs points, goal difference, goals scored,
head-to-head points, head-to-head away goals; the neutral-ground play-off is a coin flip.

**Match probabilities and Season Projections come from formally different fits of the same model**,
and issue #14's last acceptance criterion is that any published comparison between them says so.
Nothing in :mod:`epl.ledger` is fitted this way: every Prediction on the scoreboard is the MLE path,
and only a projection is the posterior.
"""

from epl.simulate.checkpoints import CHECKPOINTS_PER_SEASON, CheckpointError, projection_rounds
from epl.simulate.posterior import Diagnostics, Posterior, Priors, Sampling, fit

__all__ = [
    "CHECKPOINTS_PER_SEASON",
    "CheckpointError",
    "Diagnostics",
    "Posterior",
    "Priors",
    "Sampling",
    "fit",
    "projection_rounds",
]
