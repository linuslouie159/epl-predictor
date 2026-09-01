"""The fresh look an hour before kickoff: which Fixtures are due, and what the models say now.

The sealed half of this loop runs twice a week and forecasts a whole round at once, hours before any
of it is played. This runs every half hour and forecasts **one Fixture at a time, shortly before it
starts**, and the difference is not cosmetic: a round anchored to Friday is played Friday to Monday,
so a Reading taken at four o'clock on Sunday has seen Friday's and Saturday's results and the sealed
Prediction had not. That is a genuinely larger information set, and it is the whole reason the
message this feeds is worth sending.

**What it does not do is change the record.** The Reading goes to :mod:`epl.ledger.readings`, a
third store that the scoreboard does not read; the Sealed Prediction stands and is what gets scored.
Nothing here imports :func:`epl.ledger.live.seal` or :func:`epl.ledger.live.supersede`, and the
reason is in `readings`' own docstring: in the sealed store a later stamp for the same Fixture means
a correction, and a track record that quietly swapped in a better-informed forecast every Sunday
would be worthless in exactly the way ADR 0005 exists to prevent.

**Cheap first, expensive only when there is something to do.** The schedule fires around forty times
on a matchday and most fires have nothing due. :func:`due` answers that from the sealed store alone
— a file read, no network — and the caller stops there. Only a fire with a Fixture in its window
pays for a fetch of the Live Season's results, a rebuild of the match table and a fit of every
Predictor that will speak.

**The window is wide enough to be hit and narrow enough to mean "soon".**
:data:`WINDOW_OPENS` to :data:`WINDOW_SHUTS` against a fire every half hour means every kickoff on a
quarter-hour is inside at least one fire's window, and most are inside two — which is what
:func:`epl.ledger.readings.already_read` is for. Widening it would start reading Fixtures whose
earlier-in-the-round results have not been played yet, which is the information this exists to use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from epl.ledger import readings, schema
from epl.ledger.live import commit, push
from epl.predictors import Corpus, Evidence, registered

#: How long before kickoff a Fixture becomes due, and when it stops being due.
#:
#: Forty-five to seventy-five minutes, against a schedule firing on the hour and the half hour.
#: Every Premier League kickoff falls on a quarter-hour, so each is inside a fire's window:
#: 20:00 is caught at 19:00, 17:30 at 16:30, 12:30 at 11:30, 16:30 at 15:30. A midweek 19:45 is
#: caught twice, which the store's own dedupe absorbs.
WINDOW_OPENS = pd.Timedelta(minutes=75)
WINDOW_SHUTS = pd.Timedelta(minutes=45)

#: How a fire with nothing kicking off soon says so, named rather than only written.
#:
#: The same job as :data:`epl.live.upcoming.NO_FIXTURE_TO_PREDICT` and for the same reason: this is
#: ordinary prose to a person reading a log and it is the whole of what :mod:`epl.bot` has to tell a
#: quiet fire from one that did something. A constant so that rewording the message cannot silently
#: blind the reader — and so the coupling is visible from here rather than only from the far end.
#:
#: It is most fires. Forty a matchday and ten Fixtures a round means the overwhelming majority of
#: this schedule's output is this line, which is exactly why the bot must not announce it.
NOTHING_DUE = "nothing kicks off within the hour"

#: How a fire says a due Fixture has dropped off the rolling file, named for the same reason.
#:
#: A different silence from :data:`NOTHING_DUE` and worth telling apart: this one means a Fixture
#: *is* about to kick off and the loop could not read it, because upstream's file no longer carries
#: it. Both exit 0 — there is nothing anybody can do about either — but only this one is evidence
#: about open risk 7, so the bot must be able to see which it got.
NOT_IN_THE_ROLLING_FILE = "the rolling file no longer carries what is kicking off"


class PrematchError(Exception):
    """A Pre-Match Reading could not be taken. Never raised for "nothing is due"."""


@dataclass(frozen=True)
class Readings:
    """What one fire produced: the Fixtures it read, the rows, and whether they are proven yet."""

    path: Path | None
    rows: pd.DataFrame
    fixtures: tuple[tuple[str, str], ...]
    spoke: tuple[str, ...]
    commit: str | None

    def describe(self) -> str:
        """One line for the log: which Fixtures, from whom, and whether it is committed."""
        if self.path is None:
            return "nothing written"
        proof = f"committed {self.commit[:8]}" if self.commit else "NOT COMMITTED"
        played = ", ".join(f"{home} v {away}" for home, away in self.fixtures)
        return (
            f"read {len(self.rows)} Predictions from {len(self.spoke)} Predictors on "
            f"{played} -> {self.path.name} ({proof})"
        )


def due(sealed: pd.DataFrame, *, now: pd.Timestamp) -> pd.DataFrame:
    """The Fixtures kicking off soon that have not been read yet, one row each.

    Read off the sealed store rather than off the rolling fixtures file, and that is a deliberate
    narrowing: a Fixture with no Sealed Prediction has nothing for a Reading to be compared against,
    and the message this feeds is built around the comparison. A round the loop failed to seal
    therefore gets no pre-match messages either, which is the honest outcome — there is no forecast
    to send.

    Cheap by construction. No network, no fit, no match table: one read of a small store and two
    timestamp comparisons, because most fires end here.
    """
    if sealed.empty:
        return sealed

    moment = pd.Timestamp(now)
    fixtures = (
        sealed.drop_duplicates(subset=list(schema.FIXTURE_KEY))
        .sort_values("kickoff")
        .reset_index(drop=True)
    )
    soon = fixtures.loc[
        (fixtures["kickoff"] > moment + WINDOW_SHUTS)
        & (fixtures["kickoff"] <= moment + WINDOW_OPENS)
    ]
    unread = [
        not readings.already_read(row["home_club"], row["away_club"], day=row["kickoff"])
        for _, row in soon.iterrows()
    ]
    return soon.loc[unread].reset_index(drop=True)


def select(rolling: pd.DataFrame, wanted: pd.DataFrame) -> pd.DataFrame:
    """The rolling file's rows for the Fixtures that are due, ready to be predicted.

    Two frames because they answer two different questions and only one of them is cheap.
    :func:`due` reads the sealed store, which says *which* Fixtures are due and has a `kickoff` —
    but a stored Prediction is not a Fixture: it carries no `date` and no odds, because a ledger row
    records what a Predictor said rather than what it was shown. So the frame that gets predicted
    comes from the rolling fixtures file, which carries both.

    That is not a workaround: the odds in that file are **resampled** since the round was sealed, so
    the Market Line in a Reading is current rather than Friday's. A card an hour before kickoff
    showing a fresh model against a stale market would be the one comparison this must not make.

    A due Fixture the rolling file no longer carries comes back missing rather than raising. The
    file rolls forward and upstream regenerates it irregularly (open risk 7); a Fixture that has
    dropped off it is an ordinary Saturday, not a failure.
    """
    if rolling.empty or wanted.empty:
        return rolling.iloc[0:0]
    pairs = set(zip(wanted["home_club"], wanted["away_club"], strict=True))
    chosen = [
        (home, away) in pairs
        for home, away in zip(rolling["home_club"], rolling["away_club"], strict=True)
    ]
    return rolling.loc[chosen].reset_index(drop=True)


def readings_for(
    fixtures: pd.DataFrame, corpus: Corpus, *, now: pd.Timestamp
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Every Predictor that covers these Fixtures, run at ``now``, and who spoke.

    The same walk as :func:`epl.live.seal.sealed_predictions` and deliberately the same shape: each
    Predictor is handed its own :class:`~epl.predictors.Evidence` cut at this instant, because what
    a stored row records about the inputs it saw is a fact about that Predictor rather than about
    the moment. **There is no branch per Predictor here either** — the five that cannot speak to an
    unplayed Fixture say so themselves, exactly as they do at sealing time.

    ``as_of`` is ``now`` rather than the round's midnight, and that is the entire difference between
    this and a Sealed Prediction. It is also why these rows may never go in the sealed store.
    """
    instant = pd.Timestamp(now)
    rows: list[pd.DataFrame] = []
    spoke: list[str] = []

    for predictor in registered():
        covers = schema.covered(predictor, fixtures)
        if not covers.any():
            continue
        rows.append(
            schema.predictions_for(
                predictor,
                fixtures.loc[covers],
                Evidence.before(corpus, instant),
                as_of=instant,
            )
        )
        spoke.append(predictor.name)

    if not rows:
        return schema.empty(), ()
    combined = pd.concat(rows, ignore_index=True).sort_values(
        ["kickoff", "predictor"], kind="stable"
    )
    return schema.conform(combined.reset_index(drop=True)), tuple(spoke)


def run(
    fixtures: pd.DataFrame,
    matches: pd.DataFrame | Corpus,
    *,
    now: pd.Timestamp,
    record: bool = True,
) -> Readings:
    """Read these Fixtures at ``now``, and write the Readings to their day's file.

    ``record=False`` computes and returns without writing, which is what `--dry-run` wants: seeing
    the numbers a card would carry, on any afternoon, without putting a row in a committed store.
    """
    corpus = matches if isinstance(matches, Corpus) else Corpus(matches)
    rows, spoke = readings_for(fixtures, corpus, now=now)
    played = tuple(
        (str(row["home_club"]), str(row["away_club"])) for _, row in fixtures.iterrows()
    )
    if rows.empty:
        raise PrematchError(
            f"no registered Predictor covers {', '.join(f'{h} v {a}' for h, a in played)}, "
            "so there is nothing to read"
        )
    if not record:
        return Readings(path=None, rows=rows, fixtures=played, spoke=spoke, commit=None)

    destination = readings.record(rows)
    return Readings(
        path=destination,
        rows=rows,
        fixtures=played,
        spoke=spoke,
        commit=commit([destination], message=_message(rows, played, spoke)),
    )


def publish() -> str | None:
    """Push the branch, so a Reading proves something off the machine that took it.

    The same argument as a sealed round's push (`epl.live.__main__._push`) and the same function
    behind it: a commit on a Pi is a claim only the Pi can inspect. Named separately so the caller
    reads as a sentence rather than importing the store's own verb.
    """
    return push()


def _message(
    rows: pd.DataFrame, played: tuple[tuple[str, str], ...], spoke: tuple[str, ...]
) -> str:
    """The commit message, which is this store's index in git log."""
    names = ", ".join(f"{home} v {away}" for home, away in played)
    return (
        f"Read {names} before kickoff: {len(rows)} Predictions from {', '.join(spoke)}"
    )


__all__ = [
    "NOTHING_DUE",
    "NOT_IN_THE_ROLLING_FILE",
    "WINDOW_OPENS",
    "WINDOW_SHUTS",
    "PrematchError",
    "Readings",
    "due",
    "publish",
    "readings_for",
    "run",
    "select",
]
