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

## Measured at stage 7 (issue #11, 23 Aug 2026)

The illustrative rates above were close. Over the 1,896 and 1,512 Fixtures the two Pundits called,
with every other Predictor cut to the same slate:

| Over | Market Line | Elo | Naive Baseline | Pundit as-stated |
|---|---|---|---|---|
| Lawrenson's 1,896, RPS | 0.1946 | 0.2019 | 0.2356 | **0.3341** |
| Lawrenson's 1,896, accuracy | 0.5530 | 0.5385 | 0.4388 | **0.5095** |
| Sutton's 1,512, RPS | 0.1968 | 0.2028 | 0.2319 | **0.3343** |
| Sutton's 1,512, accuracy | 0.5453 | 0.5351 | 0.4431 | **0.4921** |

Those two rows are the argument this ADR makes, in numbers. On RPS a Pundit is 0.14 behind the
market and a tenth of a point *below the floor* — worse than a Predictor that does not know which
Clubs are playing. On accuracy, which asks only who they picked and not how sure they claimed to
be, the same calls are four points behind the market and seven ahead of the floor. Nothing about the
Pundit changes between those two lines; only the question does.

The as-stated number is published with that caveat attached to it, as each Pundit's `note` on the
scoreboard. **Two named Pundits are registered rather than one pundit slot** — Mark Lawrenson worked
2017/18–2021/22, Chris Sutton 2022/23–2025/26 — because a Pundit is "a named public forecaster"
(CONTEXT.md) and a line averaging two people is a Predictor nobody can be held to. The naming rule
in the Consequences above applies with the same force to that.

The lay pair beside it, from `python -m epl.pundits grades`: Lawrenson called the exact score 11.0%
of the time and the Outcome 50.9%; Sutton 9.1% and 49.2%.

One thing worth knowing before issue #12. The **shared** calibration layer of ADR 0006 — which is
not the Calibrated Pundit and buckets by nothing — already takes an as-stated Pundit from 0.3341 to
0.2374 and from 0.3343 to 0.2473, a gain of about 0.09 RPS where it cost every other Predictor
about 0.001. That is the bar the Calibrated Pundit has to clear, and it is evidence that most of
the as-stated gap really is the format of the question rather than the quality of the answer.
