"""Vig removal: turning a book of decimal odds into probabilities that sum to one.

A bookmaker's three prices imply probabilities that sum to more than one. The excess is the
**overround** — the margin — and stripping it out is what makes the Market Line a Prediction that
can be scored beside a model's (ADR 0001).

There is no single right way to strip it, because the margin is not observably spread across the
three Outcomes in any particular way. Three methods are implemented behind one interface:

* :func:`normalise` — divide the book out proportionally. Assumes the margin sits evenly on every
  price, which is the assumption favourite-longshot bias says is false.
* :func:`power` — raise every raw probability to a common exponent. Corrects the bias with a free
  parameter and no story about where it comes from.
* :func:`shin` — Shin's model, in which the margin exists because some bettors are better informed
  than the book. Corrects the same bias, less hard, from an explicit account of why it is there.

**Shin is the default** (DECISIONS.md). Over the Evaluation Window the three score 0.19379,
0.19362 and 0.19359 RPS — a spread of 0.0002, which is near-immaterial for benchmarking. The
choice is kept configurable anyway because power and Shin correct favourite-longshot bias and
normalisation does not, and that difference stops being immaterial the moment value betting is
explored rather than only benchmarking.

Both corrections point the same way and differ only in strength, which is worth knowing when
reading the three numbers above: on a typical Premier League book they move the favourite up and
the longshot down, normalisation least and power most, with Shin between them.

Nothing here is trusted on its own. :func:`overround` is public and reported per Season by
``python -m epl.benchmarks overround``, so the margin being removed can be looked at rather than
assumed — it runs from 9.4% in 2005/06 down to 4.1% in the early 2020s, and a method that silently
stopped removing it would show up there as a scoreboard that barely moved.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from epl.metrics import OUTCOMES

#: The default vig removal method (DECISIONS.md).
DEFAULT_METHOD = "shin"

#: How far an overround may fall below one before the book is called corrupt. A market average
#: rounded to two decimals can land a hair under a fair book; an arbitrage cannot.
FAIR_BOOK_TOLERANCE = 1e-9

#: Bisection steps used to solve for the power exponent and for Shin's insider share. Both roots
#: are bracketed and both functions are monotone, so this is the interval halved 80 times — far
#: past the point where a float can tell two candidates apart. Fixed rather than convergence-based
#: so that a rebuilt `outputs/backtest/` file is byte-identical to the last one (ADR 0005).
BISECTION_STEPS = 80

#: The largest exponent :func:`power` will consider. A book needing more than this is not a book.
MAX_EXPONENT = 100.0


class VigError(Exception):
    """A book of odds was not something the vig could be removed from."""


def shaped(odds: object) -> npt.NDArray[np.float64]:
    """Coerce to an ``(n, 3)`` array of decimal odds over (Home, Draw, Away), or raise on shape.

    A single book may be passed as a bare triple. Only the shape is checked here; whether the
    numbers are a book is :func:`is_book`'s question, and it has a per-row answer.
    """
    array = np.asarray(odds, dtype=np.float64)
    if array.size == 0:
        array = array.reshape(0, len(OUTCOMES))
    elif array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != len(OUTCOMES):
        raise VigError(
            f"a book is three decimal odds over {OUTCOMES}; got shape "
            f"{np.asarray(odds).shape}"
        )
    return array


def _faults(array: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.bool_]]:
    """Every way a row can fail to be a book, as a mask per reason.

    Stated once, so that the question :func:`is_book` answers per row and the question
    :func:`as_book` raises on are the same question. They must not drift: a Predictor whose
    ``covers`` said yes to a row ``as_book`` then refuses would stop a whole backfill on a row
    nobody meant to walk over.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        book = (1.0 / array).sum(axis=1)
    return {
        "missing": np.isnan(array).any(axis=1),
        "impossible": ~(array > 1.0).all(axis=1) & ~np.isnan(array).any(axis=1),
        "arbitrage": np.asarray(book < 1.0 - FAIR_BOOK_TOLERANCE),
    }


#: What to say about each fault, in the order they are reported. First match wins, so a row with a
#: hole in it is called missing rather than impossible.
_COMPLAINTS: tuple[tuple[str, str], ...] = (
    (
        "missing",
        "missing prices. A Season with no odds has no market comparison at all (ADR 0001), which "
        "is not the same as a book with a hole in it",
    ),
    (
        "impossible",
        "decimal odds must be greater than 1. At or below 1 a price implies certainty or better, "
        "which is corrupt data rather than a confident market",
    ),
    (
        "arbitrage",
        "a book's overround must be at least 1. Below one the book pays out more than it takes, "
        "which is an arbitrage rather than a market average",
    ),
)


def is_book(odds: object) -> npt.NDArray[np.bool_]:
    """Which rows are a book the vig can be removed from — one answer per row, never raising.

    This is what a Predictor asks before claiming to cover a Fixture. Four Fixtures in the corpus
    need it: 2025/26 League One closing books whose market average lands below an overround of one
    (as low as 0.955), which is not a price anyone was offered. No Premier League Fixture and no
    pre-match book anywhere is affected, so neither scored line loses a Fixture to it — but the
    per-tier overround report walks the whole pyramid, and a report that fell over on four rows of
    upstream noise would be a report nobody ran.
    """
    faults = _faults(shaped(odds))
    return ~np.logical_or.reduce(list(faults.values()))


def as_book(odds: object) -> npt.NDArray[np.float64]:
    """``odds`` as an ``(n, 3)`` array of decimal odds, or raise saying which row is not a book.

    Strict for the same reason :func:`epl.metrics.as_predictions` is: a malformed book is a bug in
    the ingest, not a market with an unusual opinion, and every method here would return
    plausible-looking numbers for one.
    """
    array = shaped(odds)
    faults = _faults(array)
    for reason, complaint in _COMPLAINTS:
        offending = faults[reason]
        if offending.any():
            row = int(np.argmax(offending))
            raise VigError(f"row {row} {list(array[row])}: {complaint}")
    return array


def array_overround(book: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The overround of already-shaped odds. Split out so the checks above can use it too."""
    return np.asarray((1.0 / book).sum(axis=1), dtype=np.float64)


def overround(odds: object) -> npt.NDArray[np.float64]:
    """The sum of a book's raw implied probabilities — one number per book.

    1.0562 means the market took 5.62% out, which is the whole-window mean over the Market Line
    (DECISIONS.md). Reported rather than assumed: it is the one number that shows the vig removal
    had something to remove.
    """
    return array_overround(as_book(odds))


def normalise(odds: object) -> npt.NDArray[np.float64]:
    """Divide the overround out proportionally, keeping every price's share of the book.

    The simplest method and the one every source of odds probabilities uses by default. Its
    defining property is that the ratio between any two prices survives untouched; its weakness is
    that this assumes the margin sits evenly on all three, which favourite-longshot bias denies.
    """
    raw = 1.0 / as_book(odds)
    return _renormalised(raw)


def power(odds: object) -> npt.NDArray[np.float64]:
    """Raise every raw implied probability to the common exponent that makes them sum to one.

    Solve ``sum((1/o) ** k) == 1`` for k. Since every raw probability is below one, the sum falls
    as k rises, so the root is unique and bisection finds it without a derivative.

    k exceeds one on any book with a margin, and raising to a power above one shrinks small
    probabilities proportionally harder than large ones — which is the favourite-longshot
    correction, arrived at without any account of why the bias exists.
    """
    book = as_book(odds)
    raw = 1.0 / book
    exponent = _bisect(
        lambda k: 1.0 - (raw ** k[:, None]).sum(axis=1),
        np.zeros(len(raw)),
        np.full(len(raw), MAX_EXPONENT),
    )
    return _renormalised(raw ** exponent[:, None])


def shin_z(odds: object) -> npt.NDArray[np.float64]:
    """Shin's insider share: the fraction of money the model attributes to informed bettors.

    Public because it is the sanity check on :func:`shin` that :func:`overround` is on the book. It
    is zero for a fair book and rises with the margin, and a z pinned at its bracket would mean the
    solver had failed rather than that the market was unusually exposed.
    """
    book = as_book(odds)
    raw = 1.0 / book
    return _bisect(
        lambda z: 1.0 - _shin_probabilities(raw, z).sum(axis=1),
        np.zeros(len(raw)),
        np.full(len(raw), 1.0 - 1e-9),
    )


def shin(odds: object) -> npt.NDArray[np.float64]:
    """Shin's method: the margin explained as protection against better-informed bettors.

    Shin (1993) models the book as facing a fraction ``z`` of insiders who know the Outcome. The
    price it must post to break even is then higher than the true probability, and by more on long
    prices than on short ones — which is the favourite-longshot bias, derived rather than fitted.
    Inverting that model gives, with ``B`` the overround::

        p = (sqrt(z**2 + 4 * (1 - z) * (1/o)**2 / B) - z) / (2 * (1 - z))

    with ``z`` chosen so the three sum to one. The default (DECISIONS.md), because it corrects the
    bias from a stated mechanism rather than from a free parameter, while landing between the other
    two rather than at an extreme.
    """
    book = as_book(odds)
    raw = 1.0 / book
    return _renormalised(_shin_probabilities(raw, shin_z(book)))


def _shin_probabilities(
    raw: npt.NDArray[np.float64], z: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Shin's inverse at one insider share per book, before the sum is forced to exactly one."""
    share = z[:, None]
    book = raw.sum(axis=1, keepdims=True)
    return (np.sqrt(share**2 + 4.0 * (1.0 - share) * raw**2 / book) - share) / (
        2.0 * (1.0 - share)
    )


def _renormalised(probabilities: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Divide each row by its sum.

    For :func:`normalise` this is the method. For the other two it is float housekeeping: both
    solve for a parameter that makes the row sum to one, and this removes the last bits of
    bisection residue so :func:`epl.metrics.as_predictions` never has to be told to be lenient.
    """
    return np.asarray(
        probabilities / probabilities.sum(axis=1, keepdims=True), dtype=np.float64
    )


def _bisect(
    residual: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    low: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Halve ``[low, high]`` until it closes on the root of a decreasing ``residual``, per book.

    Vectorised over books and run for a fixed number of steps rather than to a tolerance, so the
    answer depends on the odds alone and never on how quickly a particular book converged.
    """
    low, high = np.asarray(low, dtype=np.float64).copy(), np.asarray(high, dtype=np.float64).copy()
    for _ in range(BISECTION_STEPS):
        middle = 0.5 * (low + high)
        overshot = residual(middle) > 0
        high = np.where(overshot, middle, high)
        low = np.where(overshot, low, middle)
    return 0.5 * (low + high)


#: Every vig removal method, by name. The Market Line reads this rather than naming a function, so
#: comparing the three is a loop over a dict instead of a branch (issue #8).
METHODS: dict[str, Callable[[object], npt.NDArray[np.float64]]] = {
    "normalise": normalise,
    "power": power,
    "shin": shin,
}


def remove(odds: object, *, method: str = DEFAULT_METHOD) -> npt.NDArray[np.float64]:
    """Strip the margin out of a book by the named method, defaulting to Shin's.

    The one interface issue #8 asks for. Everything downstream calls this, so switching the whole
    project's vig removal is one argument rather than an edit in every Predictor.
    """
    try:
        removal = METHODS[method]
    except KeyError:
        known = ", ".join(sorted(METHODS))
        raise VigError(f"no vig removal method named {method!r}; known: {known}") from None
    return removal(odds)
