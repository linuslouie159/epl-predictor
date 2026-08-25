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

import numpy as np
import pandas as pd

from epl import metrics
from epl.ledger import schema
from epl.paths import backtest_dir
from epl.predictors import Corpus, Evidence, Predictor
from epl.rounds import assign_rounds, kickoff_instants
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
    corpus, rounds = _rounds_of(predictor, matches, seasons, divisions)
    if not rounds:
        return schema.empty()

    return schema.conform(
        pd.concat(
            [
                schema.predictions_for(
                    predictor, fixtures, Evidence.before(corpus, fixtures["as_of_instant"].iloc[0])
                )
                for fixtures in rounds
            ],
            ignore_index=True,
        )
    )


def _rounds_of(
    predictor: Predictor,
    matches: pd.DataFrame,
    seasons: Iterable[int],
    divisions: tuple[str, ...],
) -> tuple[Corpus, list[pd.DataFrame]]:
    """The corpus to cut against, and the Prediction Rounds this Predictor has anything to say in.

    Shared by :func:`backfill` and :func:`sequential` because both must scope, ask what the
    Predictor covers, and group into rounds in exactly the same way — the two differ only in what
    they do at each round, and a second copy of this would be the place they quietly stopped
    walking the same Fixtures.

    ``covered`` is asked *before* rounds are assigned, so a round this Predictor covers nothing in
    never becomes a round at all. Most Predictors cover everything and this is a no-op; the ones
    that do not are the Ceiling Line and the Pundits.
    """
    scoped = matches.loc[
        matches["season"].isin(list(seasons)) & matches["division"].isin(list(divisions))
    ]
    scoped = scoped.loc[schema.covered(predictor, scoped)]
    if scoped.empty:
        return Corpus(matches), []

    grouped = assign_rounds(scoped).groupby("prediction_round", sort=True)
    return Corpus(matches), [fixtures for _, fixtures in grouped]


#: What :func:`sequential` hands back. Deliberately **not** :data:`epl.ledger.schema.LEDGER_COLUMNS`
#: and deliberately carrying an ``outcome``, so that a diagnostic frame cannot be mistaken for
#: ledger rows or written into either store by anything that takes them.
SEQUENTIAL_COLUMNS: tuple[str, ...] = (
    "predictor",
    "season",
    "division",
    "prediction_round",
    "as_of_instant",
    "kickoff",
    "home_club",
    "away_club",
    "prob_home",
    "prob_draw",
    "prob_away",
    "sequential_prob_home",
    "sequential_prob_draw",
    "sequential_prob_away",
    "outcome",
)


def sequential(
    predictor: Predictor,
    matches: pd.DataFrame,
    *,
    seasons: Iterable[int] = EVALUATION_WINDOW,
    divisions: tuple[str, ...] = SCORED_DIVISIONS,
) -> pd.DataFrame:
    """Every Fixture predicted twice: once in its batch, and once from its own kickoff.

    **This is the diagnostic ADR 0002 promised, and it is never a headline number.** The project
    predicts in weekly Prediction Rounds and so deliberately withholds Saturday's results when
    calling Monday night's game, because the Market Line is sampled on the Friday and the Pundits
    publish on the Thursday — a per-Fixture model would silently outscore both. What the ADR owes
    the reader is the size of what it gave up, and that is a number rather than an argument.

    So each Fixture gets two Predictions from the same Predictor: ``prob_*`` from the round's As-Of
    Instant, exactly as :func:`backfill` produces and stores, and ``sequential_prob_*`` from an
    Evidence cut strictly before that Fixture's own kickoff. Both are leak-free; the second simply
    knows more.

    **Nothing here is written to either store**, and the frame is shaped so it cannot be: it carries
    the Outcome, which no ledger row may (ADR 0005), and it does not carry ``inputs_seen``. Its
    columns are :data:`SEQUENTIAL_COLUMNS`.

    Two limits worth reading before quoting the answer, and they point opposite ways.

    **The cut can only be as sharp as the corpus's timestamps.** Football-Data records no kickoff
    time before 2019/20, and such a Fixture sits at midnight on its own day — so across most of the
    window the sequential reading cannot see the earlier kickoffs of its own day either, and
    understates what a per-Fixture model would know. That is why the report prints the timed era
    separately.

    **Inside the timed era it reads generously.** :class:`~epl.predictors.Evidence` timestamps a
    match at its *kickoff*, not at its final whistle, which is exact for a Prediction Round because
    every As-Of Instant is a midnight and nothing is in play then — the guarantee
    :class:`~epl.predictors.Evidence` states. It is not exact here: a Fixture kicking off at 17:30
    is handed the 16:00 match, which was still being played. So the sequential column is an *upper
    bound* on what a per-Fixture model could honestly have known, and the gap this measures is an
    upper bound on what the weekly batch gives up. That is the useful direction for the thing
    ADR 0002 wants bounded, and it is the reason no attempt is made to sharpen it: a match-length
    constant subtracted from every kickoff would be a hyperparameter invented to flatter a
    diagnostic.

    Neither limit reaches a stored Prediction. Nothing here goes into either store.
    """
    corpus, rounds = _rounds_of(predictor, matches, seasons, divisions)
    if not rounds:
        return pd.DataFrame(columns=list(SEQUENTIAL_COLUMNS))

    return pd.concat(
        [_both_readings(predictor, corpus, fixtures) for fixtures in rounds],
        ignore_index=True,
    )[list(SEQUENTIAL_COLUMNS)]


def _both_readings(
    predictor: Predictor, corpus: Corpus, fixtures: pd.DataFrame
) -> pd.DataFrame:
    """One Prediction Round, predicted as a batch and then again Fixture by Fixture.

    Fixtures sharing a kickoff share a cut and are predicted together, which is not an optimisation
    so much as the only defensible reading: two Fixtures kicking off at three o'clock cannot inform
    each other whatever the model does.
    """
    as_of = pd.Timestamp(fixtures["as_of_instant"].iloc[0])
    # Validated through the same function the ledger validates a stored Prediction with, so a
    # diagnostic cannot quietly average in a row that does not sum to one.
    batch = metrics.as_predictions(
        predictor.predict(schema.visible(predictor, fixtures), Evidence.before(corpus, as_of))
    )

    kickoffs = kickoff_instants(fixtures)
    later = np.empty_like(batch)
    # Grouped by position rather than by index label: these Fixtures carry the match table's own
    # index, and a duplicated label there would silently predict one Fixture from another's cut.
    instants = kickoffs.to_numpy()
    for instant in np.unique(instants):
        at = np.flatnonzero(instants == instant)
        later[at] = metrics.as_predictions(
            predictor.predict(
                schema.visible(predictor, fixtures.iloc[at]),
                Evidence.before(corpus, pd.Timestamp(instant)),
            )
        )

    return pd.DataFrame(
        {
            "predictor": predictor.name,
            "season": fixtures["season"].to_numpy(),
            "division": fixtures["division"].to_numpy(),
            "prediction_round": fixtures["prediction_round"].to_numpy(),
            "as_of_instant": as_of,
            "kickoff": kickoffs.to_numpy(),
            "home_club": fixtures["home_club"].to_numpy(),
            "away_club": fixtures["away_club"].to_numpy(),
            "prob_home": batch[:, 0],
            "prob_draw": batch[:, 1],
            "prob_away": batch[:, 2],
            "sequential_prob_home": later[:, 0],
            "sequential_prob_draw": later[:, 1],
            "sequential_prob_away": later[:, 2],
            "outcome": fixtures["outcome"].to_numpy(),
        }
    )


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
