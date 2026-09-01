"""The Pre-Match Reading store: what the model said an hour before kickoff, and why it is separate.

A **Pre-Match Reading** (CONTEXT.md) is a Prediction computed roughly an hour before one Fixture
kicks off, *after* its Prediction Round was already sealed, from a corpus that by then holds the
results of matches played earlier in the same round. A Friday seal has not seen Saturday's results;
a Sunday afternoon Reading has. That is the whole reason the message an hour before kickoff is worth
sending rather than being a re-read of the sealed row.

**Why this cannot be a third kind of row in `outputs/live/`.** A Reading is stamped later than the
round it belongs to, and in the sealed store a later stamp for the same Predictor and Fixture means
exactly one thing: a superseding revision correcting a bug (:func:`epl.ledger.live.supersede`). The
scoreboard keeps the latest instant per Fixture, so putting Readings there would silently swap the
model's honest before-the-round forecast for one taken after two results were in — and the live
track record would get quietly, unfalsifiably better every week. That is ADR 0005's failure exactly,
arriving through a door it did not have.

**Why it is not `outputs/backtest/` either, and so needs a store of its own.** A Backtest Prediction
is regenerable: rerun the pipeline and it comes back identical, which is what makes it disposable.
A Reading is not. The corpus it was cut from has grown since it was taken, so re-running this code
tomorrow produces a different number and there is no way to get the old one back. Evidence that
cannot be regenerated is evidence, so this store is committed, like the sealed one and for the same
reason: a file nobody can date proves nothing.

**Three rules, and the third is what keeps the scoreboard honest.**

* One file per calendar day, named after the day, appended to as the day's matches come up.
* The same row schema as both other stores (:data:`epl.ledger.schema.LEDGER_COLUMNS`), so
  :func:`epl.ledger.schema.audit` applies here unchanged and a leak would be caught by the same
  check. A Reading's As-Of Instant is its own, and every input it saw is still strictly before it.
* **Nothing here reaches the scoreboard.** :func:`epl.ledger.stored` concatenates the backtest
  and sealed stores and does not know this one exists, which is checked rather than asserted:
  `tests/live/test_prematch.py` scores the Live Season with Readings on disk and without, and
  compares the two boards.

Whether a Reading actually beats the Prediction it was taken after is a real question and an open
one. It needs a season of them, and it is not answered here: this store exists so that the question
*can* be asked later, and so that the message an hour before kickoff quotes something that was
written down rather than something recomputed on demand.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from epl.ledger import schema
from epl.paths import prematch_dir

#: Files that belong in the store without being a day's Readings.
ALLOWED_EXTRAS: frozenset[str] = frozenset({"README.md"})

#: How a day's file is named. The day a Fixture kicks off on, not the round it belongs to: a round
#: spans Friday to Monday and the Readings for it are taken on four different afternoons, so naming
#: by round would mean four processes appending to one file over four days.
DAY_FORMAT = "%Y-%m-%d"


def path(day: pd.Timestamp) -> Path:
    """Where one day's Pre-Match Readings live."""
    return prematch_dir() / f"{pd.Timestamp(day).strftime(DAY_FORMAT)}.csv"


def record(rows: pd.DataFrame, *, day: pd.Timestamp | None = None) -> Path:
    """Append these Readings to their day's file, and return the path.

    Appending rather than replacing, because a day has several kickoff times and each gets its own
    fire. A Reading already in the file for the same Predictor and Fixture is **kept and the new one
    dropped**, which is the opposite of the sealed store's rule and right for the same underlying
    reason: there, a later row is a correction somebody made deliberately; here, a second fire
    inside one Fixture's window is the schedule doing its job twice, and the first Reading is the
    one that was actually sent. Silently replacing it would make the file disagree with the message.

    Written through :func:`epl.ledger.schema.write_csv`, so the row audit runs on the way in and a
    Reading that had seen its own Fixture's result could not be written at all.
    """
    if rows.empty:
        raise schema.LedgerError("there are no Pre-Match Readings to record")

    when = pd.Timestamp(day) if day is not None else pd.Timestamp(rows["kickoff"].min())
    destination = path(when)
    combined = (
        pd.concat([read_day(when), rows], ignore_index=True)
        if destination.exists()
        else rows
    )
    deduped = combined.drop_duplicates(
        subset=["predictor", *schema.FIXTURE_KEY], keep="first"
    )
    return schema.write_csv(
        schema.conform(deduped.sort_values(["kickoff", "predictor"]).reset_index(drop=True)),
        destination,
    )


def already_read(home_club: str, away_club: str, *, day: pd.Timestamp) -> bool:
    """Whether this Fixture already has a Reading on this day.

    The schedule fires every half hour against a window a Fixture sits inside for longer than that,
    so most Fixtures are seen twice. This is what makes the second fire quiet — and it is the store
    itself answering rather than a marker file somebody has to remember to keep in step, which is
    the same argument that makes `epl.ledger.live.is_sealed` a question about the store.
    """
    held = read_day(day)
    if held.empty:
        return False
    return bool(
        (held["home_club"].eq(home_club) & held["away_club"].eq(away_club)).any()
    )


def read_day(day: pd.Timestamp) -> pd.DataFrame:
    """One day's Readings, or an empty frame when that day has none."""
    destination = path(day)
    return schema.read_csv(destination) if destination.exists() else schema.empty()


def read() -> pd.DataFrame:
    """Every Pre-Match Reading, oldest day first."""
    return schema.read_all(days())


def days() -> list[Path]:
    """Every day file in the store, oldest first."""
    directory = prematch_dir()
    if not directory.exists():
        return []
    return sorted(file for file in directory.glob("*.csv") if _names_a_day(file))


def _names_a_day(file: Path) -> bool:
    try:
        pd.Timestamp(file.stem)
    except ValueError:
        return False
    return True


__all__ = [
    "ALLOWED_EXTRAS",
    "DAY_FORMAT",
    "already_read",
    "days",
    "path",
    "read",
    "read_day",
    "record",
]
