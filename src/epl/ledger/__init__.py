"""Ledger: the two Prediction stores, the row audit and the scoreboard over them.

Built at stage 3 (issue #7). `schema` holds the one row schema and its audits, `backtest` the
regenerable store, `live` the sealed one, and `scoreboard` the scoring over both.

At stage 6 (issue #10) `scoreboard` also became where the shared calibration layer is applied. It
belongs here rather than in any Predictor because it is fitted on Outcomes, and the Predictions it
is fitted on are what these stores hold — but nothing calibrated is ever *written* to a store. A
calibrated Prediction is a function of a stored Prediction and of Outcomes that happened after it,
and no row in either store may know an Outcome (see below).

Two stores for one kind of record looks like duplication, so the reason is worth restating here
(ADR 0005). A Backtest Prediction is reproducible — rerun the pipeline and it comes back identical —
so storing it is a convenience. A Sealed Prediction is evidence, and evidence that can be silently
regenerated is not evidence.

The failure mode is concrete: Football-Data's current-Season file grows weekly and backfills results
and odds into rows already published. Regenerating last Friday's live Prediction a month later can
feed it odds that did not exist when the Prediction was claimed to have been made, and nothing in
the code would flag it. The live accuracy log — the one artifact in this project that is worthless
if not trustworthy — would quietly become fiction.

So `outputs/backtest/` is regenerable, deterministic and gitignored, while `outputs/live/` is
append-only: one file per Prediction Round, written before that round's first kickoff and committed,
so git history is itself the proof of when each Prediction existed.

Both stores share one row schema, so scoring code never needs to know which it is reading. Two
checks belong here and matter more than any unit test:

* no Prediction consumed a row timestamped at or after its As-Of Instant
* no file in `outputs/live/` changed after its round's first kickoff

Correcting a genuine bug in a sealed round means adding a superseding row under a new As-Of Instant,
never editing history. The ingest already applies that same rule to the bytes a Prediction was made
from — see `epl.ingest.superseded_dir`.
"""

import importlib

import pandas as pd

from epl.ledger import backtest, live, schema, scoreboard
from epl.ledger.schema import (
    DTYPES,
    FIXTURE_KEY,
    LEDGER_COLUMNS,
    LedgerError,
    audit,
    check,
    predictions_for,
)
from epl.ledger.scoreboard import (
    RELIABILITY_REPORT_COLUMNS,
    SCOREBOARD_COLUMNS,
    calibrated_predictions,
    scored_predictions,
)

#: Importing these is what puts Predictors on the scoreboard. Each stage adds its own — the
#: benchmarks at issues #7 and #8, ``epl.models`` at #9, ``epl.pundits`` at #11 and #12.
#:
#: Here rather than in the command line because two entry points now need it: ``epl.ledger`` walks
#: and scores every registered Predictor, and ``epl.pundits`` reports over a subset of the same
#: board. A second hand-written list is how a Predictor comes to exist on one report and not the
#: other.
PREDICTOR_PACKAGES: tuple[str, ...] = ("epl.benchmarks", "epl.models", "epl.pundits")


def register_all() -> None:
    """Import every package that registers Predictors.

    Imported inside the call rather than at module scope, because ``epl.pundits`` reports over this
    package and importing it from here would close the loop.
    """
    for package in PREDICTOR_PACKAGES:
        importlib.import_module(package)


def stored() -> pd.DataFrame:
    """Every Prediction in both stores, as one frame.

    One row schema, so scoring never asks which store it is reading (ADR 0005) — and a caller
    cannot tell from the shape of what comes back either, which is the property that keeps a live
    track record and a backtest scored by identical code.
    """
    return pd.concat([backtest.read(), live.read()], ignore_index=True)


__all__ = [
    "DTYPES",
    "FIXTURE_KEY",
    "LEDGER_COLUMNS",
    "PREDICTOR_PACKAGES",
    "RELIABILITY_REPORT_COLUMNS",
    "SCOREBOARD_COLUMNS",
    "LedgerError",
    "audit",
    "backtest",
    "calibrated_predictions",
    "check",
    "live",
    "predictions_for",
    "register_all",
    "schema",
    "scoreboard",
    "scored_predictions",
    "stored",
]
