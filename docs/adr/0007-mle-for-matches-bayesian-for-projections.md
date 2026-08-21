# Fit Dixon-Coles by maximum likelihood for matches, and Bayesian only for Season Projections

Fitting one model two ways invites a future reader to "simplify" it, so the split needs justifying.
Walk-forward prediction requires a refit at every Prediction Round: 1,189 of them across 2000/01–
2025/26. Full Bayesian sampling at each would make a single backtest an overnight-to-two-day job,
which is fatal during development when the backtest reruns after every change.

The split works because the two uses need different things. With ~10,000 observations and ~50
parameters, parameter uncertainty barely moves any single Fixture's probability — posterior mean and
MLE agree closely. But that same uncertainty compounds across 380 simulated Fixtures into a final
table, and ignoring it is what makes naive season simulators report a title probability of 48% where
the honest answer is 34%. So the expensive tool goes where it does real work.

Maximum likelihood therefore serves all 1,189 rounds; a full posterior is fitted only where a Season
Projection is produced — weekly during the live season, and at roughly six checkpoints per historical
Season for validation.

## Consequences

Both paths share one likelihood function, so the models cannot drift apart. Match probabilities and
Season Projections come from formally different fits of the same model, and any published comparison
between them must say so. Within-Season strength drift is deliberately not modelled: across 520
club-Seasons, observed first-half to second-half variation was indistinguishable from sampling noise.
