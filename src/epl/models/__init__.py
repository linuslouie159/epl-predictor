"""Models: Elo, the ordered logit, and Dixon-Coles.

Elo through an ordered logit is built (issue #9) and Dixon-Coles by maximum likelihood is built
(#13). The Bayesian posterior over the same likelihood (#14) is not, and belongs in
:mod:`epl.simulate` when it arrives (ADR 0007).

The shared isotonic calibration layer (#10) is built and does **not** live here. It takes
Predictions and gives back Predictions, it wraps the Market Line and the Pundits as readily as it
wraps a model, and it is applied at scoring time — so it sits in :mod:`epl.calibration`, beside
the Predictor contract it wraps. Nothing in this package calls it.

    from epl.models import ELO

    ELO.predict(fixtures, evidence)   # -> an (n, 3) array over (Home, Draw, Away)

**Elo scores 0.19943 RPS** over the Evaluation Window's 7,980 Fixtures, against the Naive
Baseline's 0.22938 and the Market Line's 0.19362. It takes 0.030 of the 0.036 RPS the market takes
out of the floor — 84% of the available edge, from ratings and nothing else, before any calibration
or any goal model. It is not expected to reach the market, and the 0.0008 it still sits above the
README's ≤0.1986 target now rests entirely on issue #13: the shared calibration layer measured at
stage 6 costs Elo 0.0009 RPS rather than buying any, because Elo is already well calibrated and
what a monotone map finds in it is sampling noise (docs/DECISIONS.md, "Measured at stage 6").

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
* **Dixon-Coles is fitted two ways from one likelihood** (ADR 0007): maximum likelihood at every
  Prediction Round — all 1,189 in the corpus, the 952 of them that are scored — and a full
  posterior only where a Season Projection is produced. The split is
  deliberate and must not be "simplified" away. The likelihood both share is
  :mod:`epl.models.likelihood`, and it deliberately knows nothing about optimisers, Evidence or
  Predictors — :mod:`epl.models.dixon_coles` is the MLE half, and issue #14 is the other.
* **Dixon-Coles rates the whole pyramid too, and that was measured rather than inherited**
  (:mod:`epl.models.dixon_coles`). ADR 0004 is an argument about Elo, which is zero-sum and so
  carries a rating across a promotion by construction; this model has no such guarantee, and the
  tiers are joined only by the Clubs that changed tier inside the decay horizon. On the Burn-In
  Window fitting all four tiers scores 0.20165 against a Premier-League-only 0.20382, so the
  argument holds for the goals model as well — but for a different reason and by measurement.

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

    python -m epl.models fit          re-derive the frozen hyperparameters on the Burn-In Window
    python -m epl.models draws        the draw rate against Supremacy, predicted and observed
    python -m epl.models ratings      the pool at a Season's first Prediction Round
    python -m epl.models strengths    Dixon-Coles' attack and defence at a Season's first round
    python -m epl.models sequential   what predicting per Fixture instead of per round would buy
"""

from epl.models import burn_in, dixon_coles, elo, likelihood, ordered_logit
from epl.models.dixon_coles import (
    DIXON_COLES,
    FITTED_DIVISIONS,
    FROZEN_DECAY,
    DixonColes,
)
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
from epl.models.likelihood import (
    MAX_GOALS,
    Decay,
    Sample,
    Strengths,
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
    "DIXON_COLES",
    "DRAW_BUCKETS",
    "DRAW_CURVE_COLUMNS",
    "ELO",
    "FITTED_DIVISIONS",
    "FROZEN_DECAY",
    "FROZEN_LOGIT",
    "FROZEN_SETTINGS",
    "HOME_SCORE",
    "MAX_GOALS",
    "START_RATING",
    "Decay",
    "DixonColes",
    "Elo",
    "ModelError",
    "OrderedLogit",
    "Ratings",
    "Sample",
    "Settings",
    "Strengths",
    "burn_in",
    "dixon_coles",
    "draw_curve",
    "elo",
    "likelihood",
    "ordered_logit",
    "sigmoid",
    "supremacy",
]
