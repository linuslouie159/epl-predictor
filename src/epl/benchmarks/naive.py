"""The Naive Baseline — the floor every Predictor has to clear.

Base-rate Outcome frequencies with no knowledge of which Clubs are playing. It scores ~0.2292 RPS
over the Evaluation Window, and a Predictor that cannot beat that has no value (CONTEXT.md).

Its one subtlety is that the rates are estimated **walk-forward**: at each Prediction Round it
counts only the matches its Evidence holds, which is everything played strictly before that round's
As-Of Instant. Fitting the floor on the whole history would be the same leak this project spends
its architecture avoiding, and it would move in the flattering direction — a floor that knew how
the next twenty Seasons turned out is a slightly better floor, and every Predictor measured against
it would look slightly worse than it is.

That is also why the published whole-window rates (Home 45.6% / Draw 24.3% / Away 30.1%) are not
what this Predictor quotes. It quotes what was knowable at the time, so the two land close but not
equal, and the difference is the leak that is being refused.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from epl.metrics import OUTCOMES
from epl.predictors import Evidence, register

#: What it says before it has seen a single match: no information, stated honestly.
UNINFORMED: tuple[float, ...] = tuple(1 / len(OUTCOMES) for _ in OUTCOMES)


class NaiveBaseline:
    """Outcome base rates, counted from the tiers being predicted and from nothing else.

    Restricting the count to the Fixtures' own tiers is deliberate: the pyramid is rated as one for
    Elo because Clubs move between tiers (ADR 0004), but a base rate has no Club in it to move, and
    League Two's home advantage is not evidence about the Premier League.
    """

    name = "naive_baseline"

    def predict(self, fixtures: pd.DataFrame, evidence: Evidence) -> npt.NDArray[np.float64]:
        seen = evidence.matches(divisions=tuple(pd.unique(fixtures["division"])))
        return np.tile(_base_rates(seen["outcome"]), (len(fixtures), 1))


def _base_rates(outcomes: pd.Series) -> npt.NDArray[np.float64]:
    """How often each Outcome happened, in the ordinal (Home, Draw, Away) order."""
    counts = outcomes.value_counts()
    total = float(counts.sum())
    if total == 0:
        return np.asarray(UNINFORMED, dtype=np.float64)
    rates = np.array([float(counts.get(outcome, 0)) for outcome in OUTCOMES]) / total
    return np.asarray(rates / rates.sum(), dtype=np.float64)


#: The registered instance. It is stateless — every rate it quotes comes from the Evidence it was
#: handed — so one instance serves the whole scoreboard.
NAIVE_BASELINE = register(NaiveBaseline())
