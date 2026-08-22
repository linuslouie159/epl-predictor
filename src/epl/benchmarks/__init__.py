"""Benchmarks: the Market Line, the Ceiling Line and the Naive Baseline.

All three are built: the Naive Baseline at issue #7 (`naive.py`), the two market lines at issue #8
(`market.py`, with the vig removal in `vig.py`).

The **Market Line** is the opponent: vig-removed implied probabilities from the market-average
*pre-match* odds, scoring ~0.1936 RPS. Pre-match rather than closing, against convention, because
closing odds absorb team news the model and the Pundits cannot have — scoring against them would
produce a loss that says nothing about model quality (ADR 0001).

The **Ceiling Line** is the same treatment of closing odds, available from 2019/20. Reference upper
bound only, always labelled as knowing more than we do. It is never the headline opponent.

The **Naive Baseline** is the floor: Outcome base rates with no knowledge of which Clubs are
playing, scoring **0.22938 RPS** walk-forward. Anything that fails to beat it has no value. Its
base rates are estimated only from Seasons already seen, so even the floor is leak-free — the
0.2292 quoted in the design is the whole-window figure, and the difference is the leak refused.

Vig removal implements normalisation, the power method and Shin's method behind one interface, with
**Shin as the default**. They differ by 0.0002 RPS, so the choice is near-immaterial for
benchmarking — but power and Shin correct favourite-longshot bias, which matters if value betting is
ever explored. The overround is reported alongside every Market Line so the removal can be
sanity-checked rather than trusted; it measures 1.0562 over the Evaluation Window.

Which Seasons carry a market at all is recorded by the ingest, not re-derived here — see
`epl.ingest.odds_availability`. A Season with no odds has no market comparison rather than a
market comparison of zero, so each line declares what it `covers` and the walk writes nothing for
the rest (ADR 0001).

Measured over the Evaluation Window: Market Line 0.19362 RPS over 7,980 Fixtures; Ceiling Line
0.19676 over the 2,660 from 2019/20 — which reads worse only because it is a different, harder
span. On the Fixtures they share, the Market Line scores 0.19810 and the Ceiling Line beats it by
0.0013 RPS. That caveat travels with the Ceiling Line onto the scoreboard as its `note`.

    python -m epl.benchmarks overround     the margin in each book, per Season and tier
    python -m epl.benchmarks methods       the three vig removals compared on one book
"""

from epl.benchmarks import market, vig
from epl.benchmarks.market import (
    CEILING_LINE,
    CLOSING_COLUMNS,
    MARKET_LINE,
    PREMATCH_COLUMNS,
    MarketError,
    OddsLine,
    overround_report,
)
from epl.benchmarks.naive import NAIVE_BASELINE, UNINFORMED, NaiveBaseline

__all__ = [
    "CEILING_LINE",
    "CLOSING_COLUMNS",
    "MARKET_LINE",
    "NAIVE_BASELINE",
    "PREMATCH_COLUMNS",
    "UNINFORMED",
    "MarketError",
    "NaiveBaseline",
    "OddsLine",
    "market",
    "overround_report",
    "vig",
]
