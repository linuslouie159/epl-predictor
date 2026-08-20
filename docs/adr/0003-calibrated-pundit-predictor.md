# Score Pundits via a calibrated distribution, and publish the as-stated number beside it

A Pundit publishes a Scoreline, not a distribution. Taking it at face value as `[1, 0, 0]` and
scoring it with RPS charges 1.00 for calling Home when Away happens — punishing a claim of
certainty the Pundit never made. On illustrative rates it puts a pundit at ~0.36 RPS against a
market at ~0.19, a gap that mostly measures the format of the question rather than the quality of
the answer. That would fail our own honesty bar.

We therefore register two distinct Predictors. **Pundit** is the raw Scoreline, scored as-stated.
**Calibrated Pundit** maps the Scoreline — bucketed by predicted goal margin, since a 3-0 call is a
stronger claim than 2-1 — onto the empirical Outcome frequencies that call has historically
produced, fitted walk-forward on past calls only. The headline three-way comparison uses the
calibrated form; the as-stated number is published beside it as the "cost of stating certainty".

## Consequences

Calibrated Pundit is a one-feature model, not a person. It may beat our own models, which is a real
finding about the information in pundit calls — but it must never be presented as "Sutton beat the
model". The naming keeps that distinction visible in code and in output.
