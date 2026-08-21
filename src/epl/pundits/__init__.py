"""Pundits: backfill, grading, and the Calibrated Pundit.

Built by issues #11 (backfill and as-stated scoring), #12 (Calibrated Pundit and the three-way
scoreboard) and #16 (the BBC live spike).

A Pundit publishes a Scoreline, not a distribution. Scoring that as `[1, 0, 0]` charges 1.00 RPS for
calling Home when Away happens — punishing a claim of certainty the Pundit never made, and putting
them at ~0.36 against a market at ~0.19 in a gap that mostly measures the format of the question.
That would fail this project's own honesty bar.

So two distinct Predictors are registered (ADR 0003). **Pundit** is the raw Scoreline scored
as-stated. **Calibrated Pundit** maps the Scoreline — bucketed by predicted goal margin, since a 3-0
call is a stronger claim than 2-1 — onto the Outcome frequencies that call has historically
produced, fitted walk-forward on past calls only. The headline three-way comparison uses the
calibrated form; the as-stated number is published beside it as the cost of stating certainty.

**Calibrated Pundit is a one-feature model, not a person.** It may beat our own models, which is a
real finding about the information in pundit calls — but it must never be presented as "Sutton beat
the model". The naming keeps that distinction visible in code and in output.

Store only the facts — Fixture, predicted Scoreline, Predictor, date — never the prose, and
attribute BBC as the origin.
"""

__all__: list[str] = []
