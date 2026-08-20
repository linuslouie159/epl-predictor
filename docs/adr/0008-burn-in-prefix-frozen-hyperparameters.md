# Reserve 2000/01–2004/05 for burn-in and tuning, and never score it

A reader will see 26 Seasons ingested but only 21 scored and assume data was lost. It is deliberate.
Tuning hyperparameters — Elo's K-factor and home-advantage constant, the ordered logit's cutpoints,
Dixon-Coles' time decay — against the full history and then running a walk-forward backtest over that
same history contaminates every result. The machinery still looks rigorous and still reports a number,
but the number is too good, because each Prediction came from a model whose settings were chosen
knowing how the whole period turned out.

Seasons 2000/01–2004/05 are therefore a Burn-In Window: ratings warm up there, all hyperparameters are
fitted there, and then frozen. Scoring begins at 2005/06.

The cost is zero in practice. The Market Line first exists in 2005/06, so those five Seasons could
never have been compared against the market anyway. The Evaluation Window ends up exactly matching the
market-benchmark window: 2005/06–2025/26, 7,980 Fixtures.

## Consequences

Hyperparameters are fitted once and do not adapt to the structural drift measured in the data — home
win rate has fallen from ~47% in the 2000s to ~43% since 2020, and draw rates have drifted too. Frozen
settings will be visibly suboptimal in the late Evaluation Window. That is accepted in exchange for a
protocol that is trivially verifiable as leak-free; periodic re-tuning was considered and rejected as
harder to explain and audit.
