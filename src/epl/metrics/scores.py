"""The scores themselves, and the coercion that keeps them comparable.

Nothing here knows which Predictor produced a Prediction. A metric that could tell an Elo output
from a Market Line output could be tuned, however unintentionally, to flatter one of them; every
comparison this project publishes depends on that being impossible (spec, user story 16).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

#: The three Outcomes, in the ordinal order RPS depends on: Draw sits between Home and Away.
OUTCOMES: tuple[str, ...] = ("H", "D", "A")

#: How far a Prediction's probabilities may stray from summing to one before it is rejected.
#: Loose enough for the float noise of vig removal, tight enough to catch a genuine mistake.
SUM_TOLERANCE = 1e-6

#: Canonical column order for the per-Fixture score table.
PER_PREDICTION_COLUMNS: tuple[str, ...] = (
    "outcome",
    "top_pick",
    "hit",
    "rps",
    "brier",
    "log_loss",
)

#: Probabilities are floored here before log loss takes their log. A Predictor that says 0.0 and
#: is wrong would otherwise score infinity and make every mean it appears in infinite too.
LOG_LOSS_FLOOR = 1e-15


class MetricsError(Exception):
    """A Prediction or an Outcome was not something that can be scored."""


def as_predictions(predictions: object) -> npt.NDArray[np.float64]:
    """Coerce to an ``(n, 3)`` array of probabilities over (Home, Draw, Away), or raise.

    A single Prediction may be passed as a bare triple. The validation is deliberately strict:
    a Prediction whose probabilities do not sum to one is not a badly calibrated Predictor, it is
    a bug, and letting it through would put a meaningless number on the scoreboard.
    """
    array = np.asarray(predictions, dtype=np.float64)
    if array.size == 0:
        array = array.reshape(0, len(OUTCOMES))
    elif array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != len(OUTCOMES):
        raise MetricsError(
            f"a Prediction is {len(OUTCOMES)} probabilities over {OUTCOMES}; got shape "
            f"{np.asarray(predictions).shape}"
        )
    if not np.isfinite(array).all():
        raise MetricsError("a Prediction may not hold NaN or infinity")
    if (array < 0).any():
        raise MetricsError("a Prediction may not hold a negative probability")
    drift = np.abs(array.sum(axis=1) - 1.0)
    if (drift > SUM_TOLERANCE).any():
        worst = int(np.argmax(drift))
        raise MetricsError(
            f"a Prediction must sum to 1; row {worst} sums to {array[worst].sum():.9f}"
        )
    return array


def as_outcomes(outcomes: str | Sequence[str] | object) -> npt.NDArray[np.intp]:
    """Coerce Outcome labels to their ordinal indices, or raise.

    A single Outcome may be passed as a bare ``"H"``, ``"D"`` or ``"A"``.
    """
    labels = np.asarray([outcomes] if isinstance(outcomes, str) else outcomes, dtype=object)
    if labels.ndim != 1:
        raise MetricsError(f"Outcomes must be a flat sequence; got shape {labels.shape}")

    index = {outcome: position for position, outcome in enumerate(OUTCOMES)}
    try:
        return np.array([index[str(label)] for label in labels], dtype=np.intp)
    except KeyError as unknown:
        raise MetricsError(
            f"{unknown.args[0]!r} is not an Outcome; expected one of {OUTCOMES}"
        ) from None


def _aligned(
    predictions: object, outcomes: str | Sequence[str] | object
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]:
    probabilities = as_predictions(predictions)
    observed = as_outcomes(outcomes)
    if len(probabilities) != len(observed):
        raise MetricsError(
            f"{len(probabilities)} Predictions against {len(observed)} Outcomes"
        )
    return probabilities, observed


def _mean(scores: npt.NDArray[np.float64]) -> float:
    """The mean of a slate's scores, refusing an empty one.

    Averaging nothing is undefined, and a NaN printed on a scoreboard reads as a real number.
    """
    if len(scores) == 0:
        raise MetricsError("no Fixtures to score")
    return float(scores.mean())


def _one_hot(observed: npt.NDArray[np.intp]) -> npt.NDArray[np.float64]:
    """What actually happened, as a probability distribution that is certain after the fact."""
    hot = np.zeros((len(observed), len(OUTCOMES)), dtype=np.float64)
    hot[np.arange(len(observed)), observed] = 1.0
    return hot


def rps_per_prediction(
    predictions: object, outcomes: str | Sequence[str] | object
) -> npt.NDArray[np.float64]:
    """The Ranked Probability Score of each Prediction, unaveraged.

    The mean squared error between the cumulative Prediction and the cumulative Outcome, over the
    first two categories — dividing by two is what puts the worst possible score at exactly 1.00.

    Cumulating is the whole point. It is what makes calling Home when Away happens cost more than
    calling Home when a Draw happens, which a metric treating the three Outcomes as unordered
    labels cannot do.
    """
    probabilities, observed = _aligned(predictions, outcomes)
    steps = len(OUTCOMES) - 1
    forecast = np.cumsum(probabilities, axis=1)[:, :steps]
    happened = np.cumsum(_one_hot(observed), axis=1)[:, :steps]
    return np.asarray(((forecast - happened) ** 2).sum(axis=1) / steps, dtype=np.float64)


def rps(predictions: object, outcomes: str | Sequence[str] | object) -> float:
    """Mean Ranked Probability Score — the project's primary metric.

    Lower is better. The floor is 0.00 for a certain Prediction that is right; the ceiling is 1.00
    for a certain Prediction that is wrong in the worst direction.

    >>> rps([1.0, 0.0, 0.0], "A")
    1.0
    >>> rps([1.0, 0.0, 0.0], "D")
    0.5
    """
    return _mean(rps_per_prediction(predictions, outcomes))


def brier_per_prediction(
    predictions: object, outcomes: str | Sequence[str] | object
) -> npt.NDArray[np.float64]:
    """The multi-category Brier score of each Prediction, unaveraged."""
    probabilities, observed = _aligned(predictions, outcomes)
    return np.asarray(((probabilities - _one_hot(observed)) ** 2).sum(axis=1), dtype=np.float64)


def brier(predictions: object, outcomes: str | Sequence[str] | object) -> float:
    """Mean multi-category Brier score. Lower is better; the worst possible score is 2.00.

    Summed over the three Outcomes rather than averaged, which is Brier's original multi-category
    definition. Reported as a cross-check on RPS, never as the headline: Brier treats the three
    Outcomes as unordered labels, so it charges the same for calling Home when a Draw happens as
    for calling Home when an Away win happens.

    >>> brier([1.0, 0.0, 0.0], "A")
    2.0
    """
    return _mean(brier_per_prediction(predictions, outcomes))


def log_loss_per_prediction(
    predictions: object, outcomes: str | Sequence[str] | object
) -> npt.NDArray[np.float64]:
    """The log loss of each Prediction, unaveraged."""
    probabilities, observed = _aligned(predictions, outcomes)
    given = probabilities[np.arange(len(observed)), observed]
    return np.asarray(-np.log(np.clip(given, LOG_LOSS_FLOOR, None)), dtype=np.float64)


def log_loss(predictions: object, outcomes: str | Sequence[str] | object) -> float:
    """Mean log loss — the negative log of the probability placed on what happened.

    Probabilities are floored at :data:`LOG_LOSS_FLOOR` before the log is taken. Unclipped, a
    single Predictor that said 0.0 and was wrong would return infinity and take the entire
    scoreboard with it — and a Pundit's published Scoreline is exactly such a Prediction, which is
    part of why the headline metric is RPS (ADR 0003).

    >>> log_loss([1.0, 0.0, 0.0], "H")
    0.0
    """
    return _mean(log_loss_per_prediction(predictions, outcomes))


def top_pick(predictions: object) -> npt.NDArray[np.str_]:
    """The most likely Outcome of each Prediction, for display.

    Ties are broken toward the earlier Outcome in Home-Draw-Away order. That tie-break is
    arbitrary and vanishingly rare, and :func:`accuracy` deliberately does not follow it — see
    :func:`hits`.
    """
    probabilities = as_predictions(predictions)
    return np.asarray(OUTCOMES, dtype=np.str_)[probabilities.argmax(axis=1)]


def hits(predictions: object, outcomes: str | Sequence[str] | object) -> npt.NDArray[np.float64]:
    """Whether each Prediction's top pick came in: 1.0, 0.0, or a share of 1.0 for a tie.

    A tie is not a pick. Awarding the whole hit to whichever Outcome happens to be listed first
    would hand a Predictor that never picked anything a Home-win-shaped hit rate, so ``k`` tied
    leaders each earn ``1/k`` — exactly what breaking the tie by coin flip would yield on average,
    without depending on the order the Outcomes happen to be written in.
    """
    probabilities, observed = _aligned(predictions, outcomes)
    leaders = probabilities == probabilities.max(axis=1, keepdims=True)
    picked = leaders[np.arange(len(observed)), observed]
    return np.asarray(picked / leaders.sum(axis=1), dtype=np.float64)


def accuracy(predictions: object, outcomes: str | Sequence[str] | object) -> float:
    """The top-pick hit rate: how often the most likely Outcome was the one that happened.

    Reported for lay explanation only. Accuracy cannot tell a 0.99 Home call from a 0.34 Home
    call, so it is never this project's headline metric (CLAUDE.md).

    >>> accuracy([0.5, 0.3, 0.2], "H")
    1.0
    """
    return _mean(hits(predictions, outcomes))


def per_prediction(
    predictions: object, outcomes: str | Sequence[str] | object
) -> pd.DataFrame:
    """Every score for every Fixture, one row each, in the order they were handed in.

    This is what makes a Predictor's best and worst calls findable — the spec asks for a Pundit's
    calls surfaced by calibration miss, and that is a sort over this table.

    ``top_pick`` and ``hit`` disagree on a tie, deliberately: a row can read ``top_pick="H"`` with
    ``hit=0.5`` because :func:`top_pick` has to name one Outcome for display while :func:`hits`
    refuses to award a whole hit to a Predictor that never picked anything. See :func:`hits`.
    """
    probabilities, observed = _aligned(predictions, outcomes)
    return pd.DataFrame(
        {
            "outcome": np.asarray(OUTCOMES, dtype=np.str_)[observed],
            "top_pick": top_pick(probabilities),
            "hit": hits(probabilities, outcomes),
            "rps": rps_per_prediction(probabilities, outcomes),
            "brier": brier_per_prediction(probabilities, outcomes),
            "log_loss": log_loss_per_prediction(probabilities, outcomes),
        }
    )[list(PER_PREDICTION_COLUMNS)]


@dataclass(frozen=True)
class Scorecard:
    """Every headline number for one Predictor over one slate of Fixtures.

    Carries no Predictor identity by design. A Scorecard is the same object whether it came from
    Elo, the Market Line, a Pundit or the Naive Baseline, which is what makes the three-way
    comparison structurally incapable of being apples-to-oranges (spec, user story 16).
    """

    fixtures: int
    rps: float
    brier: float
    log_loss: float
    accuracy: float


def score(predictions: object, outcomes: str | Sequence[str] | object) -> Scorecard:
    """Score a slate of Predictions against what happened.

    >>> score([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], ["H", "A"]).rps
    0.5
    """
    probabilities, _ = _aligned(predictions, outcomes)
    return Scorecard(
        fixtures=len(probabilities),
        rps=rps(probabilities, outcomes),
        brier=brier(probabilities, outcomes),
        log_loss=log_loss(probabilities, outcomes),
        accuracy=accuracy(probabilities, outcomes),
    )
