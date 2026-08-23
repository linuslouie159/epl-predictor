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

## Measured at stage 8 (issue #12, 23 Aug 2026)

The Calibrated Pundit is built (`epl.pundits.margin`, `epl.pundits.calibrated`) and it clears that
bar. Over the Fixtures every Predictor in the comparison reached — 1,856 of Lawrenson's 1,896 and
1,472 of Sutton's 1,512, the difference being the opening calls a map has no sample for:

| Over | Market Line | Elo | Calibrated Pundit | Naive Baseline | Pundit as-stated |
|---|---|---|---|---|---|
| Lawrenson's 1,856, RPS | 0.1943 | 0.2016 | **0.2127** | 0.2356 | 0.3335 |
| Sutton's 1,472, RPS | 0.1968 | 0.2031 | **0.2111** | 0.2322 | 0.3346 |

**The cost of stating certainty is 0.1209 RPS for Lawrenson and 0.1235 for Sutton.** That is the
number this ADR exists to produce: what being asked for a scoreline instead of a probability
charged each forecaster. It is published as `outputs/certainty.csv` with the gap in a column of
that name, beside a three-way board at `outputs/three_way.csv`.

Three findings behind those two rows.

**Read fairly, the same calls beat the floor they were a tenth of a point below.** As stated a
Pundit scores worse than a Predictor that does not know which Clubs are playing; through their own
margin map they beat it by 0.02. Nothing about the Pundit changed — only the reading did, which is
the argument of this ADR in its strongest form.

**Accuracy barely moves: 0.5102 → 0.5116 and 0.4925 → 0.4993.** The two readings pick almost the
same Outcomes, so the 0.12 RPS really is the format of the question rather than a different set of
opinions. The map only changes the top pick on draw calls, where no bucket has the Draw as its mode.

**The margin is doing the work the shared layer could not do.** The generic layer gets to 0.2374
and 0.2473 without seeing a Scoreline at all; bucketing by predicted goal margin gets to 0.2127 and
0.2111. A +1 call goes Home 42% and 48% of the time and a +3 call 83% and 81% — which the shared
layer cannot see, because a Pundit's stored Prediction is one-hot and has no Scoreline left in it.

**Neither Calibrated Pundit beats Elo**, and this ADR's Consequences section anticipated that it
might. It sits between Elo and the floor. Had it landed the other way the naming rule would have
been carrying real weight rather than standing by, which is why it is in the code regardless: the
Predictors are registered as `margin_map_lawrenson` and `margin_map_sutton`, and each `note` says
in as many words that it is a one-feature model and not the person.

One more, and it confirms ADR 0006 rather than qualifying it. Put the margin map in front of the
shared isotonic layer and the layer's 0.09 RPS gain **disappears**: it costs 0.0014 and 0.0015,
exactly what it costs Elo and the market. The layer was never broken; it only ever had something to
find because an as-stated Pundit was the most miscalibrated Prediction on the board, and the margin
map has already fixed that. Ten-bin calibration error falls from 0.327 and 0.338 to 0.019 and
0.020.
