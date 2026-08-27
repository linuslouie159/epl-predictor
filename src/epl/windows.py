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

From stage 13 there is a **third** span, and it is the one that moves. The Live Season is
ingested — the live loop cannot predict it otherwise, and cannot score what it sealed — but it is
neither Burn-In nor Evaluation. It is never fitted on, because nothing outside the Burn-In Window
ever is (ADR 0008); and it is never *backfilled*, because a Backtest Prediction is regenerable and
regenerating a live one is precisely the fiction ADR 0005 exists to prevent. Its Predictions are
sealed and only sealed, and they are scored on their own board rather than folded into the
Evaluation Window's — a headline that grew by a handful of Fixtures every Saturday would not be
comparable with itself from one week to the next.
"""

from __future__ import annotations

#: First Season ingested. Football-Data's match statistics begin here (decision 2).
FIRST_SEASON = 2000

#: Last Season ingested — and, being the last, the Live Season. Moving it is what puts the
#: current campaign into the corpus, which is a deliberate act rather than a config bump: this is
#: the leakage protocol's own module.
LAST_SEASON = 2026

#: Seasons 2000/01-2004/05. Ratings warm up and hyperparameters are fitted here. Never scored.
BURN_IN_WINDOW = range(2000, 2005)

#: Seasons 2005/06-2025/26 — the span over which Predictors are scored against each other.
#: 21 Seasons, 7,980 Premier League Fixtures.
#:
#: **Deliberately not `LAST_SEASON`.** These are the *closed* Seasons, and closed is what makes the
#: number on the scoreboard mean the same thing this week as last. The Live Season is
#: :data:`LIVE_SEASON` and is scored apart from them.
EVALUATION_WINDOW = range(2005, 2026)

#: The Season predicted live: sealed before kickoff, scored retrospectively, never backfilled.
#:
#: Defined as the last Season ingested rather than as a second literal, so there is one place to
#: move at the start of a campaign and no way for the two to disagree. What stops it going stale
#: is a measurement rather than a promise — :func:`epl.live.upcoming.to_predict` refuses a Season
#: the corpus does not hold, and refuses one it holds a complete campaign of.
LIVE_SEASON = LAST_SEASON


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
