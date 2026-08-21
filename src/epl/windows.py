"""Season identity and the two Windows — defined once, imported everywhere.

A Season is identified by its start year: ``2019`` means 2019/20.

The split between the Windows is the project's leakage protocol, not a convenience. Ratings warm up
and every hyperparameter is fitted inside the **Burn-In Window**, which is then frozen and never
scored. Scoring happens only over the **Evaluation Window**. Tuning against the full history and
then walking a backtest over that same history still looks rigorous and still reports a number — it
is just too good, because each Prediction came from a model whose settings were chosen knowing how
the whole period turned out (ADR 0008).

That is why these live in one module rather than as literals near their use. A second definition
drifting out of step with this one would not raise; it would silently move the boundary and hand
back a better score.

The cost of the split is zero in practice: the Market Line first exists in 2005/06, so the Burn-In
Seasons could never have been compared against the market anyway. The Evaluation Window ends up
exactly matching the market-benchmark window.
"""

from __future__ import annotations

#: First Season ingested. Football-Data's match statistics begin here (decision 2).
FIRST_SEASON = 2000

#: Last Season ingested. 2025/26 closes the Evaluation Window.
LAST_SEASON = 2025

#: Seasons 2000/01-2004/05. Ratings warm up and hyperparameters are fitted here. Never scored.
BURN_IN_WINDOW = range(2000, 2005)

#: Seasons 2005/06-2025/26 — the span over which Predictors are scored against each other.
#: 21 Seasons, 7,980 Premier League Fixtures.
EVALUATION_WINDOW = range(2005, 2026)


def season_label(season: int) -> str:
    """How a Season is written for humans.

    >>> season_label(2005)
    '2005/06'
    >>> season_label(1999)
    '1999/00'
    """
    return f"{season}/{(season + 1) % 100:02d}"


def is_burn_in(season: int) -> bool:
    """Whether this Season is for warming up and tuning — and therefore never scored."""
    return season in BURN_IN_WINDOW


def is_evaluation(season: int) -> bool:
    """Whether this Season is scored."""
    return season in EVALUATION_WINDOW
