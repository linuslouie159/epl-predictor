"""One number in, three probabilities out — and the draw band that narrows by itself.

Elo yields a single rating difference; a Prediction is three probabilities over an *ordinal*
Outcome, with Draw sitting between Home and Away. An ordered logit is the mapping that respects
that structure (ADR 0006).

A latent match margin slides along a line. Two cutpoints divide the line into three intervals, and
a Prediction is how much of the logistic distribution falls into each::

    P(Away) = sigma(lower - margin)
    P(Draw) = sigma(upper - margin) - sigma(lower - margin)
    P(Home) = 1 - sigma(upper - margin)

The draw band between the cutpoints has *fixed width*, so as the margin slides the band's share of
the distribution shrinks on its own. That is the whole reason for choosing this over a fixed draw
probability or a hand-coded taper, and none of the fall is coded anywhere here: it comes out of
three parameters — one scale and two cutpoints.

The fall it has to reproduce was measured across 7,980 matches at 32.3% between evenly matched
Clubs down to 13.4% at the widest Supremacy. Read that figure carefully before comparing anything
to it: it is the *observed* draw rate bucketed by **market** Supremacy. A model with a noisier
Supremacy sorts the same Fixtures less sharply, so its most-even bucket is a genuinely less even
set and draws less often — Elo spans 30.2% to 14.5% over its own buckets, a 2.08x fall against the
market's 2.4x. :func:`draw_curve` is how either is measured; docs/DECISIONS.md records both.

The parameters are *fitted*, in the Burn-In Window and nowhere else (:mod:`epl.models.burn_in`,
ADR 0008). Nothing in this module fits anything: it is the mapping, and it is deliberately unaware
of Elo, of matches and of Seasons, so that Dixon-Coles or anything else with a latent margin can be
read through the same three numbers.

Calibration is a separate step and belongs to issue #10. What comes out of here is a Predictor's
*raw* output, which every metric is reported against as well as post-calibration, so that a large
correction reads as a warning about the model rather than as a silent fix (ADR 0006).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from epl.metrics import OUTCOMES


class ModelError(Exception):
    """A model, or one of its fitted parameters, was not something that could produce a
    Prediction."""


def sigmoid(margin: object) -> npt.NDArray[np.float64]:
    """The logistic function, written so that a large margin cannot overflow.

    ``exp`` of a few hundred is ``inf`` and ``inf/inf`` is ``nan``, which would reach the ledger as
    a Prediction that does not sum to one. Rating gaps that big do not arise between two Premier
    League Clubs, but the Burn-In fit searches over ``exp(log scale)`` and can put a candidate
    scale close to zero on its way past — and a numerical warning that is merely harmless today is
    what hides the one that is not tomorrow.

    Each branch is evaluated only where it applies. Writing this as a ``where`` over both would
    overflow anyway, because ``where`` computes both sides and then chooses.
    """
    array = np.asarray(margin, dtype=np.float64)
    computed = np.empty(array.shape, dtype=np.float64)

    high = array >= 0
    computed[high] = 1.0 / (1.0 + np.exp(-array[high]))
    exponential = np.exp(array[~high])
    computed[~high] = exponential / (1.0 + exponential)
    return computed


def supremacy(predictions: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """How far apart two Clubs are judged to be: the gap between Home and Away (CONTEXT.md).

    Read off the Prediction rather than off the rating difference, because that is what the term
    means and what the measured draw curve is stated against. It is also the only form comparable
    across Predictors — the Market Line has no ratings to take a difference of.
    """
    array = np.asarray(predictions, dtype=np.float64)
    if array.size == 0:
        return np.empty(0, dtype=np.float64)
    return np.asarray(array[:, 0] - array[:, 2], dtype=np.float64)


#: How many Supremacy buckets :func:`draw_curve` uses. Ten, matching :data:`epl.metrics.BINS`, so
#: the two diagrams a Predictor is read through are cut the same way.
DRAW_BUCKETS = 10

#: Canonical column order for :func:`draw_curve`.
DRAW_CURVE_COLUMNS: tuple[str, ...] = (
    "bucket",
    "fixtures",
    "mean_supremacy",
    "lowest_supremacy",
    "highest_supremacy",
    "predicted_draw",
    "observed_draw",
)


def draw_curve(
    predictions: npt.NDArray[np.float64],
    outcomes: object,
    *,
    buckets: int = DRAW_BUCKETS,
) -> pd.DataFrame:
    """How the draw rate moves with Supremacy — as predicted, and as it turned out.

    ADR 0006's claim measured rather than asserted. Fixtures are sorted into equal-sized buckets by
    **how far apart** the two Clubs were judged to be, so bucket 0 is the closest calls and the
    last is the widest mismatches, and each reports the draw probability the Predictor quoted
    beside the share that were actually drawn.

    Both columns, because they answer different questions. A taper that is predicted and not
    observed is a miscalibration; one that is observed and not predicted is a model with nothing to
    say about draws at all. Neither shows up in a single number.

    Bucketed on the *magnitude* of Supremacy, which is signed: a strong away Club is as far from an
    even contest as an equally strong home one, and the draw band does not care which way round.

    Takes probabilities and Outcomes rather than a Predictor, so it reads the Market Line and a
    Pundit exactly as it reads Elo.
    """
    array = np.asarray(predictions, dtype=np.float64)
    if array.size == 0:
        return pd.DataFrame(columns=list(DRAW_CURVE_COLUMNS))

    gap = np.abs(supremacy(array))
    frame = pd.DataFrame(
        {
            "bucket": _buckets(gap, buckets),
            "supremacy": gap,
            "predicted": array[:, 1],
            "drawn": (np.asarray(outcomes, dtype=object).ravel() == "D").astype(np.float64),
        }
    )
    summary = (
        frame.groupby("bucket", sort=True)
        .agg(
            fixtures=("supremacy", "size"),
            mean_supremacy=("supremacy", "mean"),
            lowest_supremacy=("supremacy", "min"),
            highest_supremacy=("supremacy", "max"),
            predicted_draw=("predicted", "mean"),
            observed_draw=("drawn", "mean"),
        )
        .reset_index()
    )
    return summary[list(DRAW_CURVE_COLUMNS)]


def _buckets(gap: npt.NDArray[np.float64], buckets: int) -> npt.NDArray[np.intp]:
    """Which Supremacy bucket each Fixture falls in, as equal-sized groups.

    A Predictor with no Supremacy to speak of gets one bucket. The Naive Baseline says the same
    thing about every Fixture, so there is nothing to sort it by — one bucket is the honest answer,
    where quantiles over a constant would place every Fixture in no bucket at all and report an
    empty curve as though the question had not been asked.
    """
    if len(gap) == 0 or gap.min() == gap.max():
        return np.zeros(len(gap), dtype=np.intp)
    return np.asarray(
        pd.qcut(gap, buckets, labels=False, duplicates="drop"), dtype=np.intp
    )


@dataclass(frozen=True)
class OrderedLogit:
    """The mapping from one latent edge to three probabilities.

    ``scale`` is how many units of edge make one unit of latent margin — for Elo, how many rating
    points it takes to move the line by one logit. ``cutpoints`` are where the draw band starts and
    ends. Three parameters, all fitted (ADR 0006).

    Frozen, because a fitted parameter that could be reassigned after the fact is a hyperparameter
    that could be tuned outside the Burn-In Window without anything looking wrong (ADR 0008).
    """

    scale: float
    cutpoints: tuple[float, float]

    def __post_init__(self) -> None:
        lower, upper = self.cutpoints
        if not self.scale > 0:
            raise ModelError(f"scale must be positive; got {self.scale}")
        if not lower < upper:
            raise ModelError(
                f"cutpoints must be ordered, lower before upper; got {self.cutpoints}. "
                "A reversed pair gives a negative draw probability, which is an ordinal model "
                "that has lost the order RPS depends on"
            )

    @property
    def band(self) -> float:
        """The width of the draw band, in logits. What sets the draw rate at even Supremacy."""
        lower, upper = self.cutpoints
        return upper - lower

    def probabilities(self, edges: object) -> npt.NDArray[np.float64]:
        """One Prediction per edge, as an ``(n, 3)`` array over (Home, Draw, Away).

        Built from the two tails and subtraction rather than from three independent expressions,
        so the three always sum to one exactly as far as float arithmetic allows — the ledger
        rejects a Prediction that does not (:func:`epl.metrics.as_predictions`).
        """
        margin = np.asarray(edges, dtype=np.float64).ravel() / self.scale
        if margin.size == 0:
            return np.empty((0, len(OUTCOMES)), dtype=np.float64)

        lower, upper = self.cutpoints
        away = sigmoid(lower - margin)
        not_home = sigmoid(upper - margin)
        return np.column_stack([1.0 - not_home, not_home - away, away])
