"""Scoring the ledger: one line per Predictor over the Evaluation Window.

This is the module issue #7 is really about. It takes ledger rows and the match table and produces
a Scorecard per Predictor, and it is written once against the Predictor contract — there is no
branch in here for a model, for the Market Line or for a Pundit, and there is nowhere for one to be
added without it being obvious (spec, user story 16).

It also knows nothing about which store a row came from. A Backtest Prediction and a Sealed
Prediction share one schema, so both arrive here as the same frame, and a Predictor's live track
record and its backtest are scored by identical code.

Two rules the join encodes:

* **The ledger holds no Outcome**, so what happened is fetched from the match table by Fixture
  identity — the Club pairing within a Season and tier, not the date, because a postponed Fixture
  is still the Fixture that was predicted.
* **Only the latest Prediction for a Fixture counts.** Correcting a sealed Prediction means adding
  a superseding row at a new As-Of Instant (ADR 0005); scoring both would count the Fixture twice
  and average the mistake back in.

The scoreboard sits at `outputs/scoreboard.csv` rather than inside either store: it summarises
both, so it belongs to neither, and reading a store must never pick up a report as if it were a
file of Predictions. It is gitignored, because it is derived and regenerable — the same reasoning
ADR 0005 applies to the Backtest Predictions themselves. Only `outputs/live/` is evidence.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from epl import metrics
from epl.ledger import schema
from epl.paths import outputs_dir
from epl.windows import EVALUATION_WINDOW

#: Canonical column order for the scoreboard. RPS is primary; accuracy is for lay explanation only
#: and is never the headline (CLAUDE.md).
SCOREBOARD_COLUMNS: tuple[str, ...] = (
    "predictor",
    "fixtures",
    "rps",
    "brier",
    "log_loss",
    "accuracy",
)

#: The probability columns, in the ordinal (Home, Draw, Away) order the metrics expect.
PROBABILITY_COLUMNS: tuple[str, ...] = ("prob_home", "prob_draw", "prob_away")


def path() -> Path:
    """Where the scoreboard is written."""
    return outputs_dir() / "scoreboard.csv"


def scored_predictions(
    rows: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    seasons: Iterable[int] = EVALUATION_WINDOW,
) -> pd.DataFrame:
    """The ledger rows that can be scored, each with the Outcome that happened.

    Public because it is the table behind every finer-grained question the project asks later — a
    Predictor's best and worst calls, a Pundit's calibration misses — all of which are a sort or a
    group over this.
    """
    in_window = rows.loc[rows["season"].isin(list(seasons))]
    latest = (
        in_window.sort_values("as_of_instant", kind="stable")
        .drop_duplicates(subset=["predictor", *schema.FIXTURE_KEY], keep="last")
        .reset_index(drop=True)
    )
    outcomes = matches[[*schema.FIXTURE_KEY, "outcome"]].astype(
        {name: schema.DTYPES[name] for name in schema.FIXTURE_KEY}
    )
    # many-to-one: several Predictors call the same Fixture, but the match table must hold it once
    joined = latest.merge(outcomes, on=list(schema.FIXTURE_KEY), how="left", validate="m:1")
    return joined.loc[joined["outcome"].notna()].reset_index(drop=True)


def build(
    rows: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    seasons: Iterable[int] = EVALUATION_WINDOW,
) -> pd.DataFrame:
    """Every Predictor in ``rows``, scored over the window, best RPS first."""
    scored = scored_predictions(rows, matches, seasons=seasons)
    lines = [
        _line(str(name), group)
        for name, group in scored.groupby("predictor", sort=True)
        if len(group)
    ]
    board = pd.DataFrame(lines, columns=list(SCOREBOARD_COLUMNS))
    return board.sort_values("rps", kind="stable").reset_index(drop=True)


def _line(predictor: str, group: pd.DataFrame) -> dict[str, object]:
    """One Predictor's Scorecard as a scoreboard row.

    The Scorecard fields are spread rather than re-listed, so a metric added to
    :class:`epl.metrics.Scorecard` reaches the scoreboard by being named in
    :data:`SCOREBOARD_COLUMNS` and nowhere else.
    """
    card = metrics.score(
        group[list(PROBABILITY_COLUMNS)].to_numpy(float), group["outcome"].tolist()
    )
    return {"predictor": predictor} | asdict(card)


def write(board: pd.DataFrame) -> Path:
    """Write the scoreboard, and return where it went."""
    destination = path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(destination, index=False, float_format=schema.FLOAT_FORMAT, lineterminator="\n")
    return destination
