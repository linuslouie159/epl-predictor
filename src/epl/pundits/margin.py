"""The margin map: what a published Scoreline of a given goal margin has historically been worth.

This is the model behind the Calibrated Pundit (ADR 0003), and it has exactly one feature. A call
is reduced to its **predicted goal margin** — 3-0 and 4-1 are both +3, 1-1 is 0, 0-2 is -2 — and
the map answers "when this Pundit has said that before, how often was it Home, Draw and Away?".

**Why the margin and not the Outcome the Scoreline implies.** Reading a call as its Outcome throws
away the part of it the Pundit took a risk on: 1-0 and 3-0 are both "Home", and only one of them is
a claim that Home wins nine times in ten. Measured over the nine archived Seasons the difference is
enormous — a +1 call goes Home 45% of the time and a +3 call 83% — so a map that could not tell
them apart would be leaving most of the information in the call on the floor. That is issue #12's
first acceptance criterion and it is the reason this module exists rather than a second call to the
shared layer.

**This is not `epl.calibration`, and must not be collapsed into it.** The shared isotonic layer sees
a stored Prediction and nothing else, and a Pundit's stored Prediction is one-hot: it holds the
Outcome the Scoreline implied and has no Scoreline left in it. So the shared layer *cannot* see that
a 3-0 is a stronger claim than a 2-1, and no amount of refitting it would let it. It takes the two
Pundits from 0.334 to 0.237 and 0.247 all the same, which is the bar this map has to clear (ADR
0003), and both layers stay in the project because they are answering different questions.

## Buckets, and the one constant

A bucket is a predicted goal margin. Nothing is capped and no bucket boundary is chosen: the
margins the Pundit has actually called are the buckets, and the only rule is that **a bucket too
thin to carry a rate merges with its neighbour nearer zero**.

That rule is doing real work rather than tidying up. Over the full nine Seasons the two Pundits
called +3 or better 213 times and -3 or worse 50 times, so a fixed symmetric cap would either
throw away the +3 bucket the ticket asks for by name or invent a -3 rate out of two dozen calls.
Merging outward-to-inward settles it from the sample instead: +3 earns its own bucket, -3 is read
with the -2s, and neither is a number anybody chose.

:data:`MINIMUM_SAMPLE` is the one number here, and it is **stated rather than fitted** — because
it cannot be fitted. ADR 0008 permits a hyperparameter to be tuned only inside the Burn-In Window
(2000/01-2004/05), and no Pundit in this project published a single call before 2017/18. So there
is no honest fit available, and the reasoning is structural instead; see the constant's own note.

## Fitted on past calls only

Nothing in this module knows what "past" means — it is handed a sample and fits it. The
walk-forward cut is :mod:`epl.pundits.calibrated`'s job, applied through
:class:`epl.predictors.Evidence` like every other Predictor's, which is what makes it auditable off
the stored rows rather than asserted here. That split is deliberate: it lets every rate below be
worked by hand on a dozen calls.

## What it deliberately does not do

**It does not enforce monotonicity.** A bigger predicted margin ought to mean a higher Home rate,
and on this corpus it does at every step — but that is a measurement, re-derived by
`tests/pundits/test_calibrated_over_the_corpus.py`, and imposing it would turn a finding into an
assumption. The isotonic layer of ADR 0006 imposes monotonicity because it is correcting a
Predictor's *scale* and must not touch its ranking; here the ranking is the thing being measured.

**It does not smooth toward a base rate.** Shrinkage needs a strength, a strength is a
hyperparameter, and the paragraph above says why one cannot be fitted. :data:`MINIMUM_SAMPLE` and
the merge rule are the whole of the defence against a thin bucket, and they are enough because a
bucket below the minimum does not exist.

    from epl.pundits import margin

    fitted = margin.fit(margins, outcomes)
    fitted.quote([3, 0, -1])   # (3, 3) over (Home, Draw, Away)
    fitted.table()             # the same map as a plain frame, for publishing
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from epl.metrics import OUTCOMES, MetricsError, as_outcomes

#: How many past calls a bucket needs before it carries a rate of its own. Below it, the bucket
#: merges with its neighbour nearer zero, and a margin with no bucket at all is quoted the pooled
#: rate over every past call.
#:
#: Stated, not fitted, and it could not have been fitted: hyperparameters are tuned in the Burn-In
#: Window (ADR 0008) and no Pundit published a call within a dozen years of it. So the reasoning is
#: structural, and it is the same one :data:`epl.calibration.MINIMUM_SAMPLE` gives at a different
#: scale. A bucket is a claim about three Outcomes, and the rarest of them is the Draw at about a
#: quarter of Fixtures. Forty calls expect ten Draws; twenty expect five, which is a rate read off
#: single digits and one unlucky month away from quoting a Draw at zero.
#:
#: Its cost is visible rather than hidden. Each Pundit's first forty-odd calls have no map behind
#: them, so the Calibrated Pundit does not cover them at all — and the scoreboard's `fixtures`
#: column is where that shows up.
MINIMUM_SAMPLE = 40

#: What the pooled fallback is called in :meth:`MarginMap.table`. Not a margin, so it cannot be
#: confused with one.
POOLED = "pooled"

#: Canonical column order for a published map.
MAP_COLUMNS: tuple[str, ...] = ("margins", "calls", "prob_home", "prob_draw", "prob_away")


class MarginMapError(Exception):
    """A margin map could not be fitted on what it was handed."""


@dataclass(frozen=True)
class Bucket:
    """One group of predicted goal margins, and what such a call turned out to be worth.

    ``margins`` is usually one margin. It is more than one where the outermost margins were too
    thin to stand alone and merged inward — which is how a Pundit's rare 5-0 calls are read with
    their 3-0s rather than either discarded or believed.

    ``calls`` is the bucket's own receipt. A rate over 400 calls and a rate over 40 are not the
    same claim, and :meth:`MarginMap.table` publishes it beside the rate for exactly that reason.
    """

    margins: tuple[int, ...]
    calls: int
    rates: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.rates) != len(OUTCOMES):
            raise MarginMapError(
                f"a bucket quotes one rate per Outcome, {len(OUTCOMES)} of them; "
                f"got {len(self.rates)}"
            )


@dataclass(frozen=True)
class MarginMap:
    """A Pundit's margins mapped onto the Outcome frequencies their calls have produced.

    Frozen, and holding rates rather than a fitted estimator, so a map can be printed, published
    and compared — and so quoting one depends on nothing but a dictionary lookup.

    ``pooled`` is what a margin with no bucket behind it is quoted: the rate over every call the
    map was fitted on. It is reached by an away call from a Pundit who has barely made one, and by
    a draw call before there are enough of those to say anything, and it says exactly that.
    """

    buckets: tuple[Bucket, ...]
    pooled: tuple[float, ...]
    calls: int

    def __post_init__(self) -> None:
        if self.calls <= 0:
            raise MarginMapError("a margin map fitted on nothing")

    def bucket_for(self, predicted_margin: int) -> Bucket | None:
        """Which bucket quotes this margin, or ``None`` if the pooled rate does.

        A margin the Pundit has never called is read as the nearest one they have, **toward zero**
        — so a first 5-0 call is read with their 3-0s rather than as an average call, and a gap in
        the middle of the range is read from the side nearer zero. Toward zero rather than away
        from it because the middle is where the calls are: reading an unprecedented claim as a
        slightly weaker one that has evidence behind it is the conservative direction, and reading
        it as a bolder one would be inventing confidence out of a margin nobody has tested.

        **The walk stops before it crosses zero.** A home call never lands in a bucket of draw
        calls or of away calls: margin zero is not a weaker home call, it is a different claim, and
        a Pundit whose away calls cannot yet carry a rate is better told so — by the pooled rate —
        than quoted what their draw calls happened to do.
        """
        return self._quoting(self._by_margin(), predicted_margin)

    @staticmethod
    def _quoting(
        by_margin: dict[int, Bucket], predicted_margin: int
    ) -> Bucket | None:
        """:meth:`bucket_for`'s walk, over a lookup the caller already has.

        Split out so :meth:`quote` can build the lookup once for a whole round rather than once
        per Fixture in it — a Prediction Round is predicted as one batch (ADR 0002).
        """
        if predicted_margin == 0:
            return by_margin.get(0)

        step = -1 if predicted_margin > 0 else 1
        for candidate in range(predicted_margin, 0, step):
            if candidate in by_margin:
                return by_margin[candidate]
        return None

    def quote(self, margins: object) -> npt.NDArray[np.float64]:
        """One Prediction per predicted margin, in the order they were handed in.

        Reshaped rather than left to numpy, so a Prediction Round this Pundit called nothing in
        comes back as ``(0, 3)`` rather than as ``(0,)``.
        """
        asked = np.asarray(margins, dtype=np.int64).ravel()
        by_margin = self._by_margin()
        rows = [
            bucket.rates if (bucket := self._quoting(by_margin, int(value))) else self.pooled
            for value in asked
        ]
        return np.asarray(rows, dtype=np.float64).reshape(-1, len(OUTCOMES))

    def table(self) -> pd.DataFrame:
        """The whole map as a plain frame — every bucket, its sample and its rates.

        Published rather than printed only, because "what a 3-0 call from this Pundit is worth" is
        the artifact a reader of this project actually wants, and issue #12 asks for the results as
        files with no presentation logic in them.

        The pooled fallback is always the last row, named :data:`POOLED`. Always, rather than only
        where some margin actually reaches it: it is this Pundit's own base rate over the sample
        the map was fitted on, which is the thing every bucket above it should be read against.
        """
        rows = [
            {
                "margins": ", ".join(str(value) for value in bucket.margins),
                "calls": bucket.calls,
                **dict(zip(_PROBABILITY_COLUMNS, bucket.rates, strict=True)),
            }
            for bucket in self.buckets
        ]
        rows.append(
            {
                "margins": POOLED,
                "calls": self.calls,
                **dict(zip(_PROBABILITY_COLUMNS, self.pooled, strict=True)),
            }
        )
        return pd.DataFrame(rows, columns=list(MAP_COLUMNS))

    def _by_margin(self) -> dict[int, Bucket]:
        return {value: bucket for bucket in self.buckets for value in bucket.margins}


#: The rate columns of a published map, in the ordinal (Home, Draw, Away) order.
_PROBABILITY_COLUMNS: tuple[str, ...] = ("prob_home", "prob_draw", "prob_away")


def fit(
    margins: object, outcomes: object, *, minimum: int = MINIMUM_SAMPLE
) -> MarginMap:
    """The map this Pundit's past calls imply.

    ``margins`` is one predicted goal margin per past call and ``outcomes`` is what that Fixture
    actually did. Both are plain sequences: the walk-forward cut that decides which calls are
    "past" belongs to the Predictor (:mod:`epl.pundits.calibrated`), so that every rate here can be
    worked by hand on a dozen calls.

    Refuses a sample below ``minimum`` rather than returning a map that quotes the pooled rate for
    everything. Below it there is no pooled rate worth having either, and a map fitted on a handful
    of calls hands back the Outcome that happened as though it were a probability — which is the
    one failure a Pundit's track record cannot survive. Keeping such a Fixture off the walk is
    ``covers``'s job, exactly as it is for a Fixture the Pundit never called.
    """
    predicted = np.asarray(margins, dtype=np.int64).ravel()
    observed = _outcomes(outcomes)
    if len(predicted) != len(observed):
        raise MarginMapError(
            f"{len(predicted)} margins against {len(observed)} Outcomes"
        )
    if len(predicted) < minimum:
        raise MarginMapError(
            f"a margin map needs {minimum} past calls behind it and has {len(predicted)}; a rate "
            "read off fewer hands back the Outcome that happened as though it were a probability"
        )

    counts = _counts(predicted, observed)
    return MarginMap(
        buckets=tuple(
            _bucket(group, counts) for group in _grouped(counts, minimum)
        ),
        pooled=_rates(sum(counts.values(), np.zeros(len(OUTCOMES), dtype=np.int64))),
        calls=len(predicted),
    )


def margins_of(home_goals: object, away_goals: object) -> npt.NDArray[np.int64]:
    """The predicted goal margin each Scoreline claims, home Club first.

    Beside the map rather than in :mod:`epl.pundits.dataset`, because the margin is this model's
    one feature rather than a fact about a call — the arrow the dataset owns is the Scoreline to
    the Outcome it implies (:func:`epl.pundits.dataset.outcomes_of`), and that is a different
    reduction of the same call.
    """
    return np.asarray(home_goals, dtype=np.int64) - np.asarray(away_goals, dtype=np.int64)


def _counts(
    predicted: npt.NDArray[np.int64], observed: npt.NDArray[np.intp]
) -> dict[int, npt.NDArray[np.int64]]:
    """How often each margin turned out each way, keyed by margin and ascending."""
    tally: dict[int, npt.NDArray[np.int64]] = {}
    for value, outcome in zip(predicted.tolist(), observed.tolist(), strict=True):
        tally.setdefault(int(value), np.zeros(len(OUTCOMES), dtype=np.int64))[outcome] += 1
    return {value: tally[value] for value in sorted(tally)}


def _grouped(
    counts: dict[int, npt.NDArray[np.int64]], minimum: int
) -> list[tuple[int, ...]]:
    """Which margins share a bucket, thin ones merging with their neighbour nearer zero.

    Each side is swept outward from zero, accumulating margins and sealing a bucket the moment it
    reaches ``minimum``. Whatever is left at the far end is by construction below the minimum, so
    it joins the last bucket sealed on its own side — which is the one just inside it. A side that
    never sealed a bucket at all has nothing to join, and its calls fall through to the pooled
    rate; so does margin zero, which has no neighbour nearer zero to merge with and no business
    picking a side.

    Sides are swept independently. A thin away side reaching across zero would put a Pundit's home
    calls and away calls in one bucket, which is a bucket that says nothing.
    """
    zero = [(0,)] if int(counts.get(0, np.zeros(1)).sum()) >= minimum else []
    return [
        *reversed(_side(sorted((v for v in counts if v < 0), reverse=True), counts, minimum)),
        *zero,
        *_side(sorted(v for v in counts if v > 0), counts, minimum),
    ]


def _side(
    outward: list[int], counts: dict[int, npt.NDArray[np.int64]], minimum: int
) -> list[tuple[int, ...]]:
    """One side's buckets, in the order it was swept — nearest zero first."""
    sealed: list[tuple[int, ...]] = []
    carried: list[int] = []
    for value in outward:
        carried.append(value)
        if sum(int(counts[held].sum()) for held in carried) >= minimum:
            sealed.append(tuple(sorted(carried)))
            carried = []
    if carried and sealed:
        sealed[-1] = tuple(sorted((*sealed[-1], *carried)))
    return sealed


def _bucket(
    group: tuple[int, ...], counts: dict[int, npt.NDArray[np.int64]]
) -> Bucket:
    tallied = sum(
        (counts[value] for value in group), np.zeros(len(OUTCOMES), dtype=np.int64)
    )
    return Bucket(margins=group, calls=int(tallied.sum()), rates=_rates(tallied))


def _rates(tallied: npt.NDArray[np.int64]) -> tuple[float, ...]:
    """A tally of Outcomes as the Prediction it implies."""
    total = int(tallied.sum())
    if total == 0:  # pragma: no cover - a bucket with no calls is never built
        raise MarginMapError("a bucket with no calls in it has no rate")
    return tuple(float(value) / total for value in tallied)


def _outcomes(outcomes: object) -> npt.NDArray[np.intp]:
    """Outcome labels as their ordinal indices, complained about in this module's own terms.

    Coerced through :mod:`epl.metrics` rather than here, so what counts as an Outcome has one
    definition in the project and a margin map cannot be fitted against a second one.
    """
    try:
        return as_outcomes(outcomes)
    except MetricsError as unreadable:
        raise MarginMapError(str(unreadable)) from unreadable

