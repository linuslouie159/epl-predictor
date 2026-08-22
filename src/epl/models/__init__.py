"""Models: Elo, the ordered logit, Dixon-Coles, and the shared calibration layer.

Elo through an ordered logit is built (issue #9). The shared isotonic calibration (#10),
Dixon-Coles by MLE (#13) and the Bayesian posterior (#14) are not.

    from epl.models import ELO

    ELO.predict(fixtures, evidence)   # -> an (n, 3) array over (Home, Draw, Away)

**Elo scores 0.19943 RPS** over the Evaluation Window's 7,980 Fixtures, against the Naive
Baseline's 0.22938 and the Market Line's 0.19362. It takes 0.030 of the 0.036 RPS the market takes
out of the floor — 84% of the available edge, from ratings and nothing else, before any calibration
or any goal model. It is not expected to reach the market, and the 0.0008 it still sits above the
README's ≤0.1986 target is what issues #10 and #13 are for.

Three decisions constrain what goes here:

* **Elo spans all four tiers, not just the Premier League** (ADR 0004, :mod:`epl.models.elo`). A
  promoted Club arrives with a rating it earned; relegated Clubs keep updating instead of freezing;
  yo-yo Clubs need no special case at all. Measured: the three Clubs promoted for 2005/06 arrive on
  1679.8, 1623.9 and 1565.6, each from more than 200 matches.
* **An ordered logit turns one rating difference into three probabilities** (ADR 0006,
  :mod:`epl.models.ordered_logit`). The draw band has fixed width, so its share narrows
  automatically as Supremacy grows, with no hand-coded taper. Measured: Elo's predicted draw rate
  falls 30.2% -> 14.5% across ten Supremacy deciles, monotonically, and the observed rate over the
  same buckets falls 27.6% -> 13.8%.
* **Dixon-Coles is fitted two ways from one likelihood** (ADR 0007): maximum likelihood at all 1,189
  Prediction Rounds, a full posterior only where a Season Projection is produced. The split is
  deliberate and must not be "simplified" away.

Hyperparameters — Elo's K-factor and home-advantage constant, the logit's cutpoints, Dixon-Coles'
time decay — are fitted inside the Burn-In Window and then frozen (ADR 0008).
:mod:`epl.models.burn_in` is the only place that fits anything, it cuts the corpus to
2000/01-2004/05 before it walks a single match, and what it found is written into
:data:`epl.models.elo.FROZEN_SETTINGS` as literals.

Two things here look like bugs and are not:

* **The fitted draw band is symmetric and the fit is not allowed to move it.** An edge already
  carries what playing at home is worth, so an edge of zero is an even contest and Home and Away
  must come out equal. Freeing the band's centre makes it a second home-advantage parameter
  pointing the other way; searched end to end it buys 0.00004 RPS and drags the home-advantage
  constant from 80 rating points to 155. See :func:`epl.models.burn_in.fit_logit`.
* **Elo rebuilds its rating pool at every Prediction Round** instead of folding one forward. That is
  a minute per backfill, bought deliberately: a pool carried between calls has to judge whether the
  Evidence it was just handed extends the one it folded last, and getting that wrong is invisible —
  the ratings would be built from the wrong matches while every stored row still audited clean.

    python -m epl.models fit        re-derive the frozen hyperparameters on the Burn-In Window
    python -m epl.models draws      the draw rate against Supremacy, predicted and observed
    python -m epl.models ratings    the pool at a Season's first Prediction Round
"""

from epl.models import burn_in, elo, ordered_logit
from epl.models.elo import (
    DECADE,
    ELO,
    FROZEN_LOGIT,
    FROZEN_SETTINGS,
    HOME_SCORE,
    START_RATING,
    Elo,
    Ratings,
    Settings,
)
from epl.models.ordered_logit import (
    DRAW_BUCKETS,
    DRAW_CURVE_COLUMNS,
    ModelError,
    OrderedLogit,
    draw_curve,
    sigmoid,
    supremacy,
)

__all__ = [
    "DECADE",
    "DRAW_BUCKETS",
    "DRAW_CURVE_COLUMNS",
    "ELO",
    "FROZEN_LOGIT",
    "FROZEN_SETTINGS",
    "HOME_SCORE",
    "START_RATING",
    "Elo",
    "ModelError",
    "OrderedLogit",
    "Ratings",
    "Settings",
    "burn_in",
    "draw_curve",
    "elo",
    "ordered_logit",
    "sigmoid",
    "supremacy",
]
