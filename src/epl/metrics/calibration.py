"""Calibration measured: the reliability diagram and its one-number summary.

This module *measures* calibration. It does not correct it — the shared isotonic layer every
Predictor's output passes through produces Predictions, so it sits outside this package entirely
(:mod:`epl.calibration`, issue #10). Keeping the two apart is what lets every metric be reported
twice, pre-calibration and post-calibration, so a large correction reads as a warning rather than
a silent fix (spec, user story 22). It is also what keeps the promise this package rests on: no
function in here *produces* a Prediction, and the package imports nothing that can, which
``tests/metrics/test_module_contract.py`` asserts structurally.

The diagram pools all three Outcomes of every Prediction. A Prediction of (0.5, 0.3, 0.2)
contributes three points, and the question the pooled diagram answers is the one worth asking:
when this Predictor says 60%, does it happen 60% of the time?
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from epl.metrics.scores import MetricsError, _aligned, _one_hot

#: Bins in a reliability diagram. Ten, per the spec — fixed so that diagrams from different
#: Predictors are directly comparable rather than differently resolved.
BINS = 10

#: Decimal places a probability is rounded to before it is placed in a bin, so that a value one
#: float-step below an edge still counts as being on it.
EDGE_PRECISION = 9

#: Canonical column order for a reliability diagram.
RELIABILITY_COLUMNS: tuple[str, ...] = (
    "lower",
    "upper",
    "predictions",
    "mean_predicted",
    "observed",
    "gap",
)


def reliability(predictions: object, outcomes: str | Sequence[str] | object) -> pd.DataFrame:
    """A reliability diagram: what was promised in each probability band against what happened.

    One row per bin, always :data:`BINS` of them. An unoccupied bin keeps its row with a count
    of zero and no ``observed`` or ``gap`` — reporting a zero gap where nothing was predicted
    would read as perfect calibration in a band the Predictor never used.

    ``gap`` is observed minus promised, so a negative gap is overconfidence.
    """
    probabilities, observed = _aligned(predictions, outcomes)
    promised = probabilities.reshape(-1)
    happened = _one_hot(observed).reshape(-1)

    edges = np.linspace(0.0, 1.0, BINS + 1)
    # Bins are right-open, so a probability lands in the bin it opens and 1.0 folds back into the
    # top one. The rounding is not cosmetic: 0.3 is stored as slightly less than 0.3, so without
    # it a Prediction of exactly 0.3 would fall into [0.2, 0.3) instead of the bin it opens.
    index = np.clip(np.floor(np.round(promised * BINS, EDGE_PRECISION)), 0, BINS - 1).astype(int)

    counts = np.bincount(index, minlength=BINS)
    occupied = counts > 0
    with np.errstate(invalid="ignore"):
        mean_predicted = np.where(
            occupied, np.bincount(index, weights=promised, minlength=BINS) / counts, np.nan
        )
        frequency = np.where(
            occupied, np.bincount(index, weights=happened, minlength=BINS) / counts, np.nan
        )

    return pd.DataFrame(
        {
            "lower": edges[:-1],
            "upper": edges[1:],
            "predictions": counts,
            "mean_predicted": mean_predicted,
            "observed": frequency,
            "gap": frequency - mean_predicted,
        }
    )[list(RELIABILITY_COLUMNS)]


def expected_calibration_error(
    predictions: object, outcomes: str | Sequence[str] | object
) -> float:
    """The reliability diagram as one number: the count-weighted mean absolute gap.

    Empty bins are ignored rather than counted as perfectly calibrated.

    Four Predictions of (0.5, 0.3, 0.2) on Outcomes H, H, D, A give gaps of 0.05, 0.05 and 0.00
    over twelve pooled points, so the error is 0.4 / 12:

    >>> error = expected_calibration_error([[0.5, 0.3, 0.2]] * 4, ["H", "H", "D", "A"])
    >>> abs(error - 0.4 / 12) < 1e-12
    True
    """
    table = reliability(predictions, outcomes)
    weights = table["predictions"].to_numpy(dtype=np.float64)
    if weights.sum() == 0:
        raise MetricsError("no Fixtures to score")
    gaps = np.abs(table["gap"].to_numpy(dtype=np.float64))
    return float(np.nansum(gaps * weights) / weights.sum())
