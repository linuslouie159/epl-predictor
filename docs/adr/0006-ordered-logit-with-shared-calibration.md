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
