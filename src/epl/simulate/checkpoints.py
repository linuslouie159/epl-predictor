"""Where a Season Projection is taken — and, far more importantly, where it is not.

ADR 0007 splits Dixon-Coles into two fits for one reason: maximum likelihood is cheap enough to
run at every one of the 1,189 Prediction Rounds, and a full posterior is not. "Full Bayesian
sampling at each would make a single backtest an overnight-to-two-day job, which is fatal during
development when the backtest reruns after every change." The split only works if something
actually holds the expensive fit to a handful of points, and this module is that something.

The rule issue #14 states is two rules, because the two uses are different:

* **A live Season is projected weekly**, which here means at every Prediction Round. Those are not
  quite the same sentence and it is worth checking rather than assuming: 2015/16 has **42** Premier
  League Prediction Rounds across a 38-week campaign, so "every round" *is* about one a week. There
  is only one live Season at a time, so paying four minutes per round is affordable.
* **A historical Season is projected at roughly six checkpoints**, for validation — simulating from
  mid-Season across 21 Seasons and checking where the real champion landed (issue #15). Six times
  21 is 126 fits; every round times 21 would be about 20,000, which is the overnight job ADR 0007
  refuses by name.

The checkpoints are spread across the campaign rather than bunched, because what validation is
looking at is the distribution *tightening* as Fixtures are played — six projections from the same
fortnight in May would be measuring one thing six times. And the first one waits: a projection taken
before the Season has produced any Fixtures is a fit containing none of the Season, so it measures
the decay horizon rather than the campaign.

Nothing here fits anything or knows what a posterior is. It picks Prediction Rounds, so the As-Of
Instant rule (:mod:`epl.rounds`) still decides what any fit at one of them may see.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from epl.rounds import prediction_rounds
from epl.windows import season_label

#: How many posterior fits a historical Season is worth. Issue #14 says "roughly six checkpoints per
#: historical Season", and "roughly" is honest: what the number buys is a curve of projections
#: tightening over a campaign, and the shape of that curve is not sensitive to whether it has five
#: points or seven. It is stated rather than fitted — ADR 0008 allows fitting only in the Burn-In
#: Window, and this is a budget rather than a model parameter.
CHECKPOINTS_PER_SEASON = 6

#: How much of a Season must be behind a projection before the first checkpoint is taken. A tenth
#: of the campaign is about four Fixtures a Club — enough that the Season is saying something, and
#: early enough that the projection is still a forecast rather than a report.
FIRST_CHECKPOINT_AT = 0.1


class CheckpointError(Exception):
    """A Season Projection was asked for where no Season Projection can be taken."""


#: The tier a Season Projection is *of*. Dixon-Coles is fitted across all four (ADR 0004) and a
#: projection is still a distribution over one final league table — the Premier League's — so the
#: rounds a projection is taken at are that division's rounds. Handing this the whole pyramid
#: instead is not a harmless superset: 2015/16 has 46 Premier League Prediction Rounds and 70 across
#: the four tiers, and a checkpoint anchored to a midweek League Two round would project a Premier
#: League table from an instant at which no Premier League Fixture had moved.
PROJECTED_DIVISION = "E0"


def projection_rounds(
    matches: pd.DataFrame,
    season: int,
    *,
    live: bool = False,
    checkpoints: int = CHECKPOINTS_PER_SEASON,
    division: str = PROJECTED_DIVISION,
) -> pd.DataFrame:
    """The Prediction Rounds of one Season at which a Season Projection is produced.

    Returns rows of :data:`epl.rounds.ROUND_COLUMNS`, so a caller reads ``as_of_instant`` off them
    and builds its :class:`epl.predictors.Evidence` the same way every other stage does.

    ``live`` is the current Season, projected weekly. Otherwise this is a historical Season being
    validated and ``checkpoints`` fits are spread across it — see the module docstring on why the
    two differ, and why the difference is what makes ADR 0007's split affordable.

    ``division`` is what the projection is *of*, and defaults to the Premier League. It narrows
    which rounds are candidates, and narrows nothing about the fit taken at one: a posterior at any
    of these instants is fitted over the whole pyramid, exactly as the MLE is.
    """
    missing = {"season", "division"} - set(matches.columns)
    if missing:
        raise CheckpointError(
            f"a match table needs {sorted(missing)} to be cut into Seasons and tiers; "
            f"got {sorted(matches.columns)[:8]}"
        )

    within = matches.loc[(matches["division"] == division) & (matches["season"] == season)]
    if within.empty:
        raise CheckpointError(
            f"no {division} Fixtures in {season_label(season)}, so there is no Season to project. "
            f"The corpus holds {_seasons_in(matches)}"
        )

    rounds = prediction_rounds(within)
    if live:
        return rounds

    return rounds.iloc[_checkpoint_positions(len(rounds), checkpoints)].reset_index(drop=True)


def _checkpoint_positions(rounds: int, checkpoints: int) -> list[int]:
    """Which of a Season's rounds to project from: evenly spread, after the opening tenth.

    Positions rather than dates, because Prediction Rounds are not evenly spaced in time — a
    midweek round sits two days after the one before it — and what should be evenly spread is the
    amount of football behind each projection.
    """
    if checkpoints < 1:
        raise CheckpointError(f"a Season needs at least one checkpoint; got {checkpoints}")

    first = int(np.ceil(rounds * FIRST_CHECKPOINT_AT))
    available = rounds - first
    if available <= checkpoints:
        # Fewer rounds than checkpoints asked for: take what there is rather than raise. A short
        # Season is a real thing (a Season still being played is the obvious one) and it is not an
        # error, it is simply a Season with less to project from.
        return list(range(first, rounds)) or [rounds - 1]

    # `endpoint=False` so no checkpoint lands on the final round, where every Fixture has been
    # played and the "projection" is the finished table.
    return [first + int(step) for step in np.linspace(0, available, checkpoints, endpoint=False)]


def _seasons_in(matches: pd.DataFrame) -> str:
    """What the caller *does* have, for the error message. The column is checked before this."""
    if matches.empty:
        return "no Seasons at all"
    seasons = sorted(matches["season"].unique())
    return f"{season_label(int(seasons[0]))}-{season_label(int(seasons[-1]))}"


__all__ = [
    "CHECKPOINTS_PER_SEASON",
    "FIRST_CHECKPOINT_AT",
    "CheckpointError",
    "projection_rounds",
]
