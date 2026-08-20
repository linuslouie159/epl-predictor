# Predict in weekly Prediction Rounds, not per fixture

Predicting each Fixture using every match played strictly before its own kickoff is leak-free and
more accurate, so a future reader will wonder why we deliberately use less information. The reason
is comparability: it would let the model know Saturday's results when calling Monday night's game,
while the Market Line (sampled Friday) and the Pundits (published Thursday/Friday) would not. Every
three-way comparison we publish would silently overstate the model.

Instead, a Fixture's As-Of Instant is the most recent Tuesday or Friday preceding kickoff, mirroring
the market's own odds-sampling convention. Fixtures sharing an As-Of Instant form one Prediction
Round and are predicted as a batch from data strictly preceding it.

## Consequences

Source data has no matchweek or round column — 2024/25's 380 fixtures spanned 109 distinct dates —
so Prediction Rounds are derived from kickoff dates, not from any upstream notion of a gameweek.
Per-fixture sequential prediction is still run as a **diagnostic**, to quantify what the withheld
information was worth, but never as the headline number.
