# Map ratings to Outcomes with an ordered logit, behind a shared calibration layer

Elo yields one rating difference; three probabilities are needed. A fixed draw probability would be
badly wrong at both ends: measured across 7,980 matches, draws run at 32.3% between evenly matched
Clubs and 13.4% at the widest Supremacy — a 2.4x range, falling monotonically.

That range is the *observed* draw rate with Fixtures bucketed by the **market's** Supremacy, which
is the only ordering that existed when this was written. It is evidence that the taper is real and
worth modelling; it is not a target a model is held to. Bucketing by a model's own Supremacy is a
different measurement — a noisier ordering puts less evenly matched Fixtures in its most-even
bucket, and those draw less often. Elo measures 27.6% to 13.8% observed over its own buckets
(issue #9). What a Predictor is judged on is the monotone fall and its predicted curve against its
observed one, both of which this design delivers.

An ordered logit reproduces that curve for free. A latent match margin slides along a line with two
fitted cutpoints; the draw band has fixed width, so the share of the distribution falling inside it
narrows automatically as Supremacy grows. Three parameters, no hand-coded taper, and it respects the
ordinal structure that RPS scores. Davidson's ties model and an empirical lookup table were both
rejected — the first fixes the taper's shape by construction, the second has no answer for rating
gaps it has never seen, which a pyramid-wide model produces constantly.

Every Predictor's raw output then passes through one shared isotonic calibration step, fitted
walk-forward on out-of-sample Predictions only. Calibration is infrastructure, not a per-model detail,
so Elo, Dixon-Coles, the Market Line and the Pundits all receive identical treatment.

## Consequences

A calibration layer can mask a broken model by correcting its symptoms. To keep that visible rather
than hidden, every metric is reported both pre-calibration and post-calibration; a large correction is
then a warning sign about the underlying model rather than a silent fix.

### Measured at stage 6 (issue #10): the warning pointed at the layer, not at a model

The decision above stands and the layer is built (`epl.calibration`). What it does on this corpus is
not what this ADR expected, and the double reporting is the only reason anyone would know.

**The layer makes every Predictor worse.** Over the Evaluation Window, walk-forward:

| Predictor | RPS | calibrated RPS | ten-bin error | calibrated | mass moved |
|---|---|---|---|---|---|
| Market Line | 0.19362 | 0.19450 | 0.0061 | 0.0124 | 0.034 |
| Ceiling Line | 0.19676 | 0.19800 | 0.0060 | 0.0084 | 0.033 |
| Elo | 0.19943 | 0.20037 | 0.0055 | 0.0097 | 0.031 |
| Naive Baseline | 0.22938 | 0.23087 | 0.0061 | 0.0161 | 0.046 |

It moves 3% to 5% of each Prediction's probability mass and charges 0.0009 to 0.0015 RPS for it. Two
things are behind that, and the second does not cancel the first:

**Knot resolution.** An isotonic map gets a knot per distinct quote, and market odds and Elo edges
are nearly continuous — 7,909 distinct Home quotes across 7,980 Fixtures — so most knots rest on a
single Fixture and the map fits noise. Cutting the knots at ten equal-width probability bands
instead recovers most of the loss: Elo 0.20037 → 0.19968, the Market Line 0.19450 → 0.19404. That
variant is *not* shipped, because the band count would be a hyperparameter and ADR 0008 wants those
fitted inside a Burn-In Window that holds no stored Prediction. It is measured and pinned by
`tests/test_calibration_over_the_corpus.py` so the claim above does not rest on prose.

**The corpus.** Even coarse, both stay worse than raw. All four sit at a pooled ten-bin error around
0.006 before the layer touches them, so there is little real miscalibration left for a monotone map
to find. A clean split-half over the market's 7,980 Fixtures shows the same without any of the
walk's machinery: fitted on the older half by kickoff, the map improves that half by 0.0017 RPS and
costs the later half 0.0005.

At full resolution the layer also leaves every Predictor *less* calibrated than it found it, which
rules out "the correction is right and RPS is judging it unfairly".

It is not that the correction points the wrong way. Elo's draw quote at even Supremacy moves 30.2%
→ 29.3% against 27.6% observed, exactly the defect issue #9 handed this layer as its reason to
exist. The correction is real and small; the noise around it is larger.

Three things follow, and none of them changes the decision:

- **The headline numbers stay pre-calibration.** The README's ≤0.1986 target is measured against a
  Predictor's own output, because that is the better of the two and calling the worse one the answer
  would be a fiction in the opposite direction from the usual one.
- **Both columns are published anyway.** A 0.001 RPS tax applied silently to every Predictor is
  precisely what this section was written to prevent, and it took the pre-calibration column to see
  it. The layer earning its place by reporting a null result is still the layer earning its place.
- **It is worth re-reading when a Predictor arrives that needs it.** A Pundit scored as-stated
  (ADR 0003) publishes `[1, 0, 0]`, which is the most miscalibrated Prediction there is. Issue #11
  is where this layer has something real to correct, and where these numbers should be measured
  again rather than assumed to hold.

## Re-measured at stage 7 (issue #11, 23 Aug 2026)

The layer met a Predictor that needed it, and none of the numbers above survived contact:

| Predictor | Predictions | corrected | RPS | calibrated RPS | ten-bin error | calibrated |
|---|---|---|---|---|---|---|
| Lawrenson, as-stated | 1,896 | 1,508 | 0.3341 | **0.2374** | 0.3270 | 0.0792 |
| Sutton, as-stated | 1,512 | 1,130 | 0.3343 | **0.2473** | 0.3386 | 0.0988 |

A gain of about 0.09 RPS, where the four Predictors above paid about 0.001. On the Fixtures a
fitted map actually reached, Lawrenson goes 0.3385 → 0.2169 against a Naive Baseline of 0.2381 over
the same 1,508 — so the layer takes a Predictor that is far below the floor as stated and lifts it
above the floor.

**This confirms the diagnosis rather than overturning it.** The layer was never broken; it had
nothing to find. All four earlier Predictors arrive at a ten-bin error of about 0.006, and a
monotone map fitted on that finds noise. A Pundit arrives at 0.33 and the same unchanged layer
recovers most of it. Nothing above changes: the headline numbers stay pre-calibration, and both
columns keep being published — which is now what makes this row visible as well as that tax.

None of this is the Calibrated Pundit. That map is bucketed by predicted goal margin and is fitted
on a Pundit's own past calls (ADR 0003, issue #12); this one sees a one-hot input with no Scoreline
in it. They must not be collapsed into each other.
