"""The regenerable store: a Predictor walked over history, one Prediction Round at a time.

A Backtest Prediction is reproducible — rerun the pipeline and it comes back identical — so this
store is a convenience rather than evidence, and `outputs/backtest/` is gitignored and deletable at
will (ADR 0005). Its value is in the aggregate score; no individual row here is worth anything.

Two properties earn it that status, and both are tested:

* the walk is leak-free at every round, because each round's Predictor sees only
  :class:`~epl.predictors.Evidence` cut at that round's own As-Of Instant
* a rebuild writes the same bytes, so "regenerable" cannot quietly come to mean "different every
  time" and a rebuild is never mistaken for a change

The Seasons predicted and the Seasons visible are different things. :func:`backfill` scores only
the window it is given, but every round sees the whole corpus up to its instant — which is what the
Burn-In Window is for: the first scored round already has a warmed-up Predictor behind it without a
single Burn-In Fixture being scored (ADR 0008).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from epl.ledger import schema
from epl.paths import backtest_dir
from epl.predictors import Corpus, Evidence, Predictor
from epl.rounds import assign_rounds
from epl.windows import EVALUATION_WINDOW

#: The tiers a Predictor is scored on. All four are ingested and rated (ADR 0004); only the
#: Premier League is predicted.
SCORED_DIVISIONS: tuple[str, ...] = ("E0",)

#: The order rows are written in, so the file does not depend on how the caller assembled it.
SORT_KEY: tuple[str, ...] = ("predictor", "as_of_instant", "kickoff", "division", "home_club")


def path(predictor: str) -> Path:
    """Where one Predictor's Backtest Predictions live."""
    return backtest_dir() / f"{predictor}.csv"


def backfill(
    predictor: Predictor,
    matches: pd.DataFrame,
    *,
    seasons: Iterable[int] = EVALUATION_WINDOW,
    divisions: tuple[str, ...] = SCORED_DIVISIONS,
) -> pd.DataFrame:
    """Walk ``predictor`` over every Prediction Round in the window and return its ledger rows.

    ``matches`` is the whole corpus, not the window: the window says what is *predicted*, while
    everything in ``matches`` that had kicked off by a round's As-Of Instant is what that round may
    *see*.
    """
    scoped = matches.loc[
        matches["season"].isin(list(seasons)) & matches["division"].isin(list(divisions))
    ]
    # Asked before rounds are assigned, so a round this Predictor covers nothing in never becomes
    # a round at all. Most Predictors cover everything and this is a no-op; the ones that do not
    # are the Ceiling Line and, later, the Pundits.
    scoped = scoped.loc[schema.covered(predictor, scoped)]
    if scoped.empty:
        return schema.empty()

    corpus = Corpus(matches)
    assigned = assign_rounds(scoped)
    rounds = [
        schema.predictions_for(
            predictor, fixtures, Evidence.before(corpus, fixtures["as_of_instant"].iloc[0])
        )
        for _, fixtures in assigned.groupby("prediction_round", sort=True)
    ]
    return schema.conform(pd.concat(rounds, ignore_index=True))


def write(rows: pd.DataFrame) -> list[Path]:
    """Write one file per Predictor in ``rows``, and return the paths in name order.

    Nothing is written until every row passes :func:`epl.ledger.schema.audit`, so a store that
    exists is a store that was auditable when it was made.
    """
    schema.check(rows)
    ordered = rows.sort_values(list(SORT_KEY), kind="stable")
    return [
        schema.write_csv(group.reset_index(drop=True), path(str(name)))
        for name, group in ordered.groupby("predictor", sort=True)
    ]


def read(predictor: str | None = None) -> pd.DataFrame:
    """Every Backtest Prediction in the store, or just one Predictor's."""
    files = sorted(backtest_dir().glob("*.csv")) if predictor is None else [path(predictor)]
    return schema.read_all(files)
