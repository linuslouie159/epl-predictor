"""Models: Elo, the ordered logit, Dixon-Coles, and the shared calibration layer.

Built by issues #9 (pyramid-wide Elo through an ordered logit), #10 (shared isotonic calibration),
#13 (Dixon-Coles by MLE) and #14 (the Bayesian posterior).

Three decisions constrain what goes here:

* **Elo spans all four tiers, not just the Premier League** (ADR 0004). A promoted Club arrives with
  a rating it earned; relegated Clubs keep updating instead of freezing; yo-yo Clubs need no special
  case at all.
* **An ordered logit turns one rating difference into three probabilities** (ADR 0006). The draw
  band has fixed width, so its share narrows automatically as Supremacy grows — matching the
  measured fall from 32.3% to 13.4% — with no hand-coded taper.
* **Dixon-Coles is fitted two ways from one likelihood** (ADR 0007): maximum likelihood at all 1,189
  Prediction Rounds, a full posterior only where a Season Projection is produced. The split is
  deliberate and must not be "simplified" away.

Hyperparameters — Elo's K-factor and home-advantage constant, the logit's cutpoints, Dixon-Coles'
time decay — are fitted inside the Burn-In Window and then frozen (ADR 0008). See `epl.windows`.
"""

__all__: list[str] = []
