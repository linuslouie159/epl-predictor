"""The shared calibration layer — one isotonic step, wrapped around every Predictor identically.

Calibration is infrastructure, not a per-model detail (ADR 0006). Elo, the Market Line, the Ceiling
Line, the Naive Baseline and the Pundits all receive the same treatment here, and a new Predictor
gets it by being registered — there is no calibration code anywhere else and nowhere for a
per-Predictor branch to be added.

It exists because #9 measured a reason rather than a principle. Elo quotes draws too often in all
ten Supremacy buckets — 30.2% against 27.6% observed at the even end — and the Market Line
*under*-quotes them there, 28.6% against 32.0%. Both are corrections a monotone map can make.

**And on this corpus it makes every Predictor worse**, by 0.0009 to 0.0015 RPS, moving 3% to 5% of
every Prediction's probability mass to do it. Two things are behind that and neither cancels the
other:

* **Resolution.** A map gets a knot per distinct quote, and market odds and Elo edges are nearly
  continuous — 7,909 distinct Home quotes over 7,980 Fixtures — so most knots rest on a single
  Fixture and the map fits noise. Fitting instead over ten probability bands recovers most of the
  loss: Elo 0.20037 to 0.19968, the Market Line 0.19450 to 0.19404. Not taken, because the band
  count is a hyperparameter and ADR 0008 wants those fitted in a Burn-In Window that holds no
  stored Prediction; `tests/test_calibration_over_the_corpus.py` pins the coarse numbers anyway.
* **The corpus.** Even coarse, both stay worse than raw. These four sit at a pooled ten-bin error
  of about 0.006 before the layer touches them, so there is little real miscalibration left for a
  monotone map to find.

The correction it does make points the right way — Elo's even-Supremacy draw quote moves 30.2% to
29.3% against 27.6% observed. So the scoreboard's headline numbers are the pre-calibration ones, and
the post-calibration column is published beside them because a 0.001 RPS tax charged silently to
every Predictor is exactly what ADR 0006's double reporting exists to catch. Read docs/DECISIONS.md,
"Measured at stage 6", before concluding this module is broken; and measure again at issue #11,
where a Pundit scored as-stated is the first input it has something real to fix.

**Nothing here calibrates a forecast.** The layer runs at scoring time over Predictions whose
Outcomes are known, so a Prediction sealed for an unplayed Fixture is published raw and gains a
calibrated form only once its round has been scored. That follows from fitting on Outcomes and is
the reason the spec puts this "as a shared pipeline step rather than inside any model". The live
loop honours it by construction: :mod:`epl.live` seals raw Predictions, and `epl.live score` prints
only the pre-calibration board, because a Season in progress has no track record for a map to
be fitted on and a calibrated column there would be the raw one under another name.

**What it is.** One isotonic regression per Outcome, fitted one-versus-rest: "when this Predictor
said 20% Draw, how often was it a Draw?", answered as a monotone step function, then the three
answers renormalised into a Prediction. Per Outcome rather than pooled over all three, because the
correction the layer was built for is Outcome-specific — a pooled map cannot lower the draw quotes
without also lowering every Home and Away quote in the same probability band.

**Where it is fitted.** Walk-forward, on out-of-sample Predictions only, and never on the Prediction
it is correcting. The pool a Prediction Round may be corrected against is the Predictions whose
Fixtures kicked off *strictly before* that round's As-Of Instant — an Outcome is not knowable until
its Fixture has been played, so this is the same cut :class:`epl.predictors.Evidence` applies to the
corpus and it is applied for the same reason.

That is also why nothing here is frozen the way :mod:`epl.models.burn_in`'s hyperparameters are
(ADR 0008). A frozen map would be a map fitted once, on some window, and applied to Predictions
outside it; the walk refits at every round and so has nothing to freeze. :data:`MINIMUM_SAMPLE` is
the one number chosen rather than fitted, and it is chosen structurally — see its note.

**Why it does not live in `epl.models`.** It is not a model: it takes Predictions and gives back
Predictions, and it is applied to the Market Line and to Pundits, neither of which is a model at
all. Putting it under `epl.models` would also make `epl.models` and `epl.ledger` import each other,
since the ledger is where the Predictions it walks over are stored.

**Why it does not live in `epl.metrics`.** :mod:`epl.metrics.calibration` *measures* calibration and
must keep doing only that: no function in that package may **produce** a Prediction, and the package
imports nothing that can — which is what makes the three-way scoreboard structurally incapable of
being apples-to-oranges (spec, user story 16). This module produces Predictions and imports
`scikit-learn` to do it, so it sits outside.

**Why it stores nothing.** A calibrated Prediction is a function of a stored Prediction and of
Outcomes that happened after it, so it is derived at scoring time rather than written to the ledger.
No row in either store knows an Outcome (ADR 0005), and a store whose rows were built from Outcomes
would lose the property that makes a leaked Prediction distinguishable from a recorded one.

    from epl import calibration

    walked = calibration.walk_forward(predictions, outcomes, as_of_instants, kickoffs)
    walked.predictions       # (n, 3), calibrated, in the order they were handed in
    walked.moved             # probability mass moved, per Prediction
    walked.correction        # the mean of that — the size of the correction, in one number

Arrays in, arrays out. The ledger's column names are the ledger's business
(:mod:`epl.ledger.scoreboard`), which is what lets this module be tested on twelve Predictions
worked by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.isotonic import isotonic_regression

from epl.metrics import OUTCOMES, MetricsError, as_outcomes, as_predictions

#: How many (Prediction, Outcome) pairs must be available before a map is fitted at all. Below it,
#: Predictions pass through untouched and are reported as uncorrected.
#:
#: One Season of Premier League Fixtures. Stated rather than fitted, and deliberately not compared
#: against an Evaluation Window score, which is the tuning ADR 0008 forbids. The reasoning is
#: structural: an isotonic map has as many knots as the Predictor has distinct quotes, so on a
#: sample smaller than a Season it can fit one match per knot and hand back the Outcome that
#: happened as though it were a probability.
#:
#: Its cost is visible rather than hidden: the first Prediction Rounds of the Evaluation Window are
#: uncorrected, and the scoreboard reports how many Predictions a map was actually applied to.
MINIMUM_SAMPLE = 380


class CalibrationError(Exception):
    """A calibration map could not be fitted, or could not be applied to what it was handed."""


@dataclass(frozen=True)
class Curve:
    """One Outcome's monotone map: what this Predictor's quoted probability turned out to be worth.

    ``quoted`` is ascending and ``happened`` is non-decreasing — that is what makes this isotonic,
    and it is what stops the layer from re-ordering a Predictor's own judgements. A map that could
    say 0.8 is worth less than 0.4 would be correcting the Predictor's ranking rather than its
    scale, and the ranking is the part a Predictor is actually good at.

    Frozen and knot-based rather than a fitted estimator object, so a map can be printed, compared
    and stored, and so applying one depends on nothing but :func:`numpy.interp`.
    """

    quoted: tuple[float, ...]
    happened: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.quoted) != len(self.happened) or not self.quoted:
            raise CalibrationError(
                f"a Curve is a knot per quoted probability; got {len(self.quoted)} quoted "
                f"against {len(self.happened)} observed"
            )

    def __call__(self, probabilities: object) -> npt.NDArray[np.float64]:
        """What this map says each quoted probability is worth.

        Linear between knots, and **clipped** outside them. A Predictor may quote a probability it
        has never quoted before; extending a monotone map past its own evidence would invent a
        correction, so the end knot is held instead.
        """
        return np.interp(
            np.asarray(probabilities, dtype=np.float64), self.quoted, self.happened
        )


@dataclass(frozen=True)
class Isotonic:
    """The three maps that make up one calibration step, and the sample they were fitted on.

    ``curves`` is one :class:`Curve` per Outcome in :data:`epl.metrics.OUTCOMES` order. ``sample``
    is the map's own receipt — a correction drawn from twelve Predictions and one drawn from seven
    thousand are not the same claim, and only one of them is worth acting on.
    """

    curves: tuple[Curve, ...]
    sample: int

    def __post_init__(self) -> None:
        if len(self.curves) != len(OUTCOMES):
            raise CalibrationError(
                f"a calibration step is one map per Outcome, {len(OUTCOMES)} of them; "
                f"got {len(self.curves)}"
            )

    def apply(self, predictions: object) -> npt.NDArray[np.float64]:
        """One calibrated Prediction per row, renormalised so each still sums to one.

        The three maps are fitted independently and have no reason to agree on a total, so the
        renormalisation is not a tidy-up — it is what turns three answers into a Prediction. The
        ledger refuses one that does not sum to one (:func:`epl.metrics.as_predictions`).
        """
        raw = as_predictions(predictions)
        if raw.size == 0:
            return raw

        mapped = np.column_stack(
            [curve(raw[:, index]) for index, curve in enumerate(self.curves)]
        )
        # Three monotone maps can all read zero at one point, and a row of zeros has no shape left
        # to renormalise. Handing back the raw Prediction is the only answer that is still one.
        flattened = mapped.sum(axis=1) <= 0
        mapped[flattened] = raw[flattened]
        return as_predictions(mapped / mapped.sum(axis=1, keepdims=True))


@dataclass(frozen=True)
class Calibrated:
    """A walk's output: what was quoted, what the layer made of it, and how much it moved.

    Carries the raw Predictions as well as the calibrated ones because every metric is reported
    twice, before and after (ADR 0006). Keeping both in one object is what stops the two from being
    computed over different slates and compared anyway.
    """

    raw: npt.NDArray[np.float64]
    predictions: npt.NDArray[np.float64]
    fitted: npt.NDArray[np.bool_]

    @property
    def corrected(self) -> int:
        """How many Predictions a fitted map was actually applied to.

        The rest passed through because no map had enough out-of-sample Predictions behind it yet.
        Reported beside :attr:`correction` so that a small mean correction can be read as "the layer
        found little to do" rather than mistaken for "the layer barely ran".
        """
        return int(self.fitted.sum())

    @property
    def moved(self) -> npt.NDArray[np.float64]:
        """How much probability mass the layer moved on each Prediction.

        Total variation: half the summed absolute change, so it reads as the share of the
        distribution that was relocated and sits on the same 0-to-1 scale for every Predictor.
        """
        if self.raw.size == 0:
            return np.empty(0, dtype=np.float64)
        return np.asarray(np.abs(self.predictions - self.raw).sum(axis=1) / 2.0, dtype=np.float64)

    @property
    def correction(self) -> float:
        """:attr:`moved`, averaged over the whole slate — "the size of the correction", in one
        number (issue #10).

        Over every Prediction rather than only the corrected ones, so it is comparable with the RPS
        and the Brier score beside it on the scoreboard, which are also over every Prediction.
        :attr:`corrected` is what says how much of the slate the mean is diluted by.

        A large number here is a warning, and reading it takes the metrics on both sides of it: a
        large correction that *buys* something says the underlying Predictor was off, and a large
        correction that buys nothing says this layer is fitting noise. Over this project's four
        Predictors it is the second (ADR 0006, "Measured at stage 6"), which is exactly why both
        sides are published rather than only the better one.
        """
        return float(self.moved.mean()) if self.raw.size else 0.0


def fit(predictions: object, outcomes: object) -> Isotonic:
    """The calibration step these Predictions and Outcomes imply.

    One isotonic regression per Outcome, fitted one-versus-rest. Predictions quoting the same
    probability are pooled into one knot first: isotonic regression is defined on ordered inputs,
    and a Predictor that says the same thing about many Fixtures — the Naive Baseline says it about
    every Fixture in a round — must get one answer for that quote rather than an arbitrary
    ordering of the ties.
    """
    probabilities = as_predictions(predictions)
    observed = _outcomes(outcomes)
    if len(probabilities) != len(observed):
        raise CalibrationError(
            f"{len(probabilities)} Predictions against {len(observed)} Outcomes"
        )
    if len(probabilities) == 0:
        raise CalibrationError("nothing to fit a calibration map on")
    return _isotonic(probabilities, observed)


def walk_forward(
    predictions: object,
    outcomes: object,
    as_of: object,
    kickoff: object,
    *,
    minimum: int = MINIMUM_SAMPLE,
) -> Calibrated:
    """Calibrate a Predictor's whole track record without ever letting it see its own future.

    ``as_of`` is each Prediction's As-Of Instant and ``kickoff`` its Fixture's kickoff. Every
    Prediction sharing an As-Of Instant is corrected by one map — a Prediction Round is predicted as
    one batch (ADR 0002), so it is calibrated as one too — and that map is fitted on the Predictions
    whose Fixtures had kicked off **strictly before** that instant.

    Rows may arrive in any order and come back in the order they were handed in. The walk is over
    the distinct As-Of Instants rather than over the rows, so nothing depends on how the caller
    happened to sort them.

    Where fewer than ``minimum`` Predictions are available to fit on, the round passes through
    uncorrected and is reported as such in :attr:`Calibrated.fitted`. That is what "out-of-sample
    only" costs at the start of the Evaluation Window, and it is shown rather than papered over.
    """
    raw = as_predictions(predictions)
    observed = _outcomes(outcomes)
    instants = _instants(as_of, "As-Of Instants")
    kickoffs = _instants(kickoff, "kickoffs")
    if not len(raw) == len(observed) == len(instants) == len(kickoffs):
        raise CalibrationError(
            f"{len(raw)} Predictions against {len(observed)} Outcomes, {len(instants)} As-Of "
            f"Instants and {len(kickoffs)} kickoffs"
        )

    calibrated = raw.copy()
    fitted = np.zeros(len(raw), dtype=bool)
    for instant in np.unique(instants):
        # Everything already played at this instant, and so everything this round may be corrected
        # against. The comparison is the whole leak guard; see the docstring.
        pool = kickoffs < instant
        if int(pool.sum()) < minimum:
            continue
        due = instants == instant
        calibrated[due] = _isotonic(raw[pool], observed[pool]).apply(raw[due])
        fitted[due] = True
    return Calibrated(raw=raw, predictions=calibrated, fitted=fitted)


def _isotonic(
    probabilities: npt.NDArray[np.float64], observed: npt.NDArray[np.intp]
) -> Isotonic:
    """The fit itself, over already-validated inputs — one loop shared by both public entry
    points, so the walk cannot drift into fitting differently from :func:`fit`."""
    return Isotonic(
        curves=tuple(
            _curve(probabilities[:, index], (observed == index).astype(np.float64))
            for index in range(len(OUTCOMES))
        ),
        sample=len(probabilities),
    )


def _curve(quoted: npt.NDArray[np.float64], happened: npt.NDArray[np.float64]) -> Curve:
    """One Outcome's map: how often it happened at each probability this Predictor quoted for it.

    Ties are pooled before the monotone fit rather than after. ``numpy.unique`` returns the distinct
    quotes already ascending, which is both the pooling and the ordering the fit needs.
    """
    distinct, index, counts = np.unique(quoted, return_inverse=True, return_counts=True)
    rate = np.bincount(index, weights=happened, minlength=len(distinct)) / counts
    monotone = isotonic_regression(
        rate, sample_weight=counts.astype(np.float64), y_min=0.0, y_max=1.0, increasing=True
    )
    return Curve(
        quoted=tuple(float(value) for value in distinct),
        happened=tuple(float(value) for value in np.asarray(monotone, dtype=np.float64)),
    )


def _outcomes(outcomes: object) -> npt.NDArray[np.intp]:
    """Outcome labels as their ordinal indices, complained about in this module's own terms.

    Coerced through :mod:`epl.metrics` rather than here, so that what counts as an Outcome has one
    definition in the project and a calibration map cannot be fitted against a fourth one.
    """
    try:
        return as_outcomes(outcomes)
    except MetricsError as unreadable:
        raise CalibrationError(str(unreadable)) from unreadable


def _instants(values: object, called: str) -> npt.NDArray[np.datetime64]:
    """A column of instants, as datetimes that can be compared."""
    try:
        return np.asarray(values, dtype="datetime64[ns]").ravel()
    except (TypeError, ValueError) as unreadable:
        raise CalibrationError(f"the {called} are not instants: {unreadable}") from unreadable
