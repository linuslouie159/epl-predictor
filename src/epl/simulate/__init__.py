"""Simulate: the Bayesian fit and the Monte Carlo Season Projection.

Built by issues #14 (posterior) and #15 (projection).

A Season Projection is a distribution over final league tables, produced by simulating every
remaining Fixture many times: 10,000 simulated Seasons per projection, with a deterministic seed
recorded in the output so any published number can be reproduced exactly.

**The likelihood is already written, and #14 must not write a second one.** ADR 0007's whole
argument for fitting one model two ways is that "both paths share one likelihood function, so the
models cannot drift apart", and stage 9 put that function in :mod:`epl.models.likelihood` — the
rates, the low-score correction, the weighted sample and the Scoreline grid, with nothing in it that
knows what an optimiser or a Predictor is. The posterior is fitted over the same
:class:`epl.models.likelihood.Sample` and returns draws of the same
:class:`epl.models.likelihood.Strengths`; :func:`epl.models.dixon_coles.fit` is the MLE path beside
it, and the two should be readable side by side.

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
"""

__all__: list[str] = []
