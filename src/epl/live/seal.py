"""Every registered Predictor run over the upcoming round, sealed before kickoff and committed.

This is the thin half of the live loop, and thin is the point. Choosing the round is
:mod:`epl.live.upcoming`; the row schema, the leak audit and the append-only store are
:mod:`epl.ledger`; what each Predictor says is each Predictor's own business. What is left here is
the walk — and it is the same walk :func:`epl.ledger.backtest.backfill` does, cut to one round and
pointed at the other store, because a Sealed Prediction and a Backtest Prediction are one record
written under two rules (ADR 0005).

**There is no branch per Predictor here and there must never be one.** Five of the nine registered
Predictors cannot speak to a Fixture that has not been played, and all five say so themselves: two
Pundits have published no call the frozen dataset holds, two Calibrated Pundits have none to read,
and the Ceiling Line has no closing odds because the match has not closed. Each answers
:func:`epl.ledger.schema.covered` with nothing, exactly as it does for a Season it never covered,
and this module records which Predictors were silent rather than knowing in advance which they
would be. Stage 12 measured *why* a Pundit is among them (:mod:`epl.pundits.live`); nothing here
had to be told.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from epl.ledger import backtest, live, schema
from epl.live.upcoming import PredictionRound
from epl.predictors import Corpus, Evidence, registered

#: A superseding Prediction is stamped at the moment it was made, and instants are stored to the
#: second (:data:`epl.ledger.schema.DATE_FORMAT`). Floored rather than rounded, so the stamp is
#: never later than the moment it names.
INSTANT_RESOLUTION = "s"


@dataclass(frozen=True)
class Sealed:
    """What one run of the live loop wrote, and what it could not.

    ``silent`` is not a failure. It is the Predictors that declared they cover none of this round's
    Fixtures, which for five of the nine is the permanent answer for anything unplayed.
    """

    path: Path
    rows: pd.DataFrame
    spoke: tuple[str, ...]
    silent: tuple[str, ...]
    superseded: bool
    commit: str | None

    def describe(self) -> str:
        """One line for the log: what was written, by whom, and whether it is proven yet."""
        proof = f"committed {self.commit[:8]}" if self.commit else "NOT COMMITTED"
        wrote = "superseded" if self.superseded else "sealed"
        return (
            f"{wrote} {len(self.rows)} Predictions from {len(self.spoke)} Predictors "
            f"-> {self.path.name} ({proof})"
        )


def sealed_predictions(
    upcoming: PredictionRound,
    corpus: Corpus,
    *,
    as_of: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    """Ledger rows for the round from every Predictor that covers any of it, and who was silent.

    Each Predictor is handed its own :class:`~epl.predictors.Evidence`, because what a stored row
    records about the inputs it saw is a fact about that Predictor rather than about the round.

    ``as_of`` is passed through to :func:`epl.ledger.schema.predictions_for` and is only ever set
    when superseding — see there for why a later stamp is honest rather than convenient.
    """
    instant = pd.Timestamp(as_of) if as_of is not None else upcoming.as_of
    rows: list[pd.DataFrame] = []
    spoke: list[str] = []
    silent: list[str] = []

    for predictor in registered():
        covers = schema.covered(predictor, upcoming.fixtures)
        if not covers.any():
            silent.append(predictor.name)
            continue
        rows.append(
            schema.predictions_for(
                predictor,
                upcoming.fixtures.loc[covers],
                Evidence.before(corpus, instant),
                as_of=as_of,
            )
        )
        spoke.append(predictor.name)

    if not rows:
        return schema.empty(), (), tuple(silent)
    combined = pd.concat(rows, ignore_index=True).sort_values(
        list(backtest.SORT_KEY), kind="stable"
    )
    return schema.conform(combined.reset_index(drop=True)), tuple(spoke), tuple(silent)


def run(
    upcoming: PredictionRound,
    matches: pd.DataFrame | Corpus,
    *,
    now: pd.Timestamp,
    supersede: bool = False,
    commit: bool = True,
) -> Sealed:
    """Predict the round, write it to the sealed store, and commit it.

    ``supersede`` corrects a round already in the store rather than sealing it for the first time
    (:func:`epl.ledger.live.supersede`). The correction is stamped at ``now`` rather than at the
    round's own midnight, because that is when it was made; everything downstream then prefers it,
    since scoring keeps the latest instant per Fixture.

    Committing is not optional in spirit — an uncommitted sealed file proves nothing, and
    :func:`epl.ledger.live.seal_violations` says so once the round has kicked off — but it is
    optional here, so that a caller can inspect what would be written without touching history.
    """
    corpus = matches if isinstance(matches, Corpus) else Corpus(matches)
    moment = pd.Timestamp(now)
    stamp = moment.floor(INSTANT_RESOLUTION) if supersede else None

    rows, spoke, silent = sealed_predictions(upcoming, corpus, as_of=stamp)
    if rows.empty:
        raise schema.LedgerError(
            f"no registered Predictor covers any of {upcoming.prediction_round}'s Fixtures, so "
            f"there is nothing to seal. Silent: {', '.join(silent) or 'nothing registered'}"
        )

    write = live.supersede if supersede else live.seal
    path = write(rows, now=moment)
    return Sealed(
        path=path,
        rows=rows,
        spoke=spoke,
        silent=silent,
        superseded=supersede,
        commit=live.commit([path], message=_message(upcoming, rows, spoke, supersede))
        if commit
        else None,
    )


def _message(
    upcoming: PredictionRound, rows: pd.DataFrame, spoke: tuple[str, ...], superseded: bool
) -> str:
    """The commit message. It is the store's index in git log, so it says what a reader would grep
    for: the round, how much of it, and from whom."""
    verb = "Supersede" if superseded else "Seal"
    return (
        f"{verb} {upcoming.prediction_round}: {len(rows)} Predictions on "
        f"{len(upcoming.fixtures)} Fixtures from {', '.join(spoke)}"
    )


__all__ = ["INSTANT_RESOLUTION", "Sealed", "run", "sealed_predictions"]
