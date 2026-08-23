"""Metrics: RPS, Brier, log loss, accuracy, calibration.

Built by issue #6, and built **before any model**, because every comparison in this project rests
on it. Expected values are worked out by hand and written as literals — a metric verified against
the code that produces it only proves the code agrees with itself.

RPS is primary. Outcomes are ordinal (Draw sits between Home and Away), and RPS is the metric that
knows it: calling Home when Away happens costs more than calling Home when a Draw happens.
Accuracy is reported for lay explanation only and is never the headline (CLAUDE.md).

The anchors these functions hit, from ADR 0006 and the spec:

* a certain Prediction wrong in the worst direction scores exactly 1.00 RPS
* wrong by one ordinal step scores 0.50
* correct scores 0.00

Nothing here knows which Predictor produced a Prediction. Every function takes probabilities and
Outcomes and nothing else, and ``tests/metrics/test_module_contract.py`` asserts structurally that
the package imports no module capable of producing a Prediction. That is what makes the three-way
scoreboard incapable of being apples-to-oranges rather than merely intended not to be.

:mod:`epl.metrics.calibration` *measures* calibration. Correcting it produces Predictions, so it
lives outside this package in :mod:`epl.calibration` (issue #10) — which is both what keeps the
promise above and what lets every metric be emitted twice, pre- and post-calibration, so a large
correction reads as a warning rather than a silent fix. Functions here take Predictions, of course;
what none of them does is make one.

    >>> from epl import metrics
    >>> metrics.rps([0.5, 0.3, 0.2], "H")
    0.145
    >>> metrics.score([[0.5, 0.3, 0.2]], ["H"]).accuracy
    1.0
"""


from epl.metrics.calibration import (
    BINS,
    EDGE_PRECISION,
    RELIABILITY_COLUMNS,
    expected_calibration_error,
    reliability,
)
from epl.metrics.scores import (
    LOG_LOSS_FLOOR,
    OUTCOMES,
    PER_PREDICTION_COLUMNS,
    SUM_TOLERANCE,
    MetricsError,
    Scorecard,
    accuracy,
    as_outcomes,
    as_predictions,
    brier,
    brier_per_prediction,
    hits,
    log_loss,
    log_loss_per_prediction,
    per_prediction,
    rps,
    rps_per_prediction,
    score,
    top_pick,
)

__all__ = [
    "BINS",
    "EDGE_PRECISION",
    "LOG_LOSS_FLOOR",
    "OUTCOMES",
    "PER_PREDICTION_COLUMNS",
    "RELIABILITY_COLUMNS",
    "SUM_TOLERANCE",
    "MetricsError",
    "Scorecard",
    "accuracy",
    "as_outcomes",
    "as_predictions",
    "brier",
    "brier_per_prediction",
    "expected_calibration_error",
    "hits",
    "log_loss",
    "log_loss_per_prediction",
    "per_prediction",
    "reliability",
    "rps",
    "rps_per_prediction",
    "score",
    "top_pick",
]
