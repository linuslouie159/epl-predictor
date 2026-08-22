"""The sealed store: one file per Prediction Round, written before kickoff and never rewritten.

A Sealed Prediction is evidence, and evidence that can be silently regenerated is not evidence
(ADR 0005). So `outputs/live/` is append-only and committed, and git history — not a timestamp
inside the file, which anyone could type — is the proof of when each Prediction existed.

The concrete failure this guards against: Football-Data's current-Season file grows weekly and
backfills results and odds into rows already published. Regenerating last Friday's live Prediction
a month later can feed it odds that did not exist when the Prediction was claimed to have been
made, and nothing in the code would flag it.

Two moments matter and they are different. :func:`seal` refuses to write a round once its first
Fixture has kicked off; :func:`seal_violations` reports rounds whose bytes *changed* after that
moment. The first is a guard on the way in, the second an audit of what is already there — and only
the second survives someone bypassing the code and editing a file by hand.

Correcting a genuine bug in a sealed round means adding a superseding row under a new As-Of Instant
in a *later* round's file, never editing history.
"""

from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path, PurePosixPath

import pandas as pd

from epl.ledger import schema
from epl.paths import live_dir, project_root
from epl.rounds import anchor

#: Kickoffs are recorded in UK local time, so commit timestamps are compared in that zone rather
#: than in UTC — an hour of British Summer Time is exactly the slack a seal deadline cannot afford.
LOCAL_ZONE = "Europe/London"

#: Files that belong in the store without being sealed rounds.
ALLOWED_EXTRAS: frozenset[str] = frozenset({"README.md"})


def path(prediction_round: str) -> Path:
    """Where one Prediction Round's Sealed Predictions live. The round id is the filename."""
    return live_dir() / f"{prediction_round}.csv"


def seal(rows: pd.DataFrame, *, now: pd.Timestamp | None = None) -> Path:
    """Write one Prediction Round's Predictions, from every Predictor, and return the path.

    Refuses to write over a round that is already sealed, and refuses to seal a round whose first
    Fixture has already kicked off. Both are the same rule from two directions: what is in this
    directory has to be what was actually forecast, not what the code says today.
    """
    rounds = set(rows["prediction_round"])
    if len(rounds) != 1:
        raise schema.LedgerError(
            f"a sealed file holds one Prediction Round; got {sorted(rounds)} (ADR 0005)"
        )

    prediction_round = str(rounds.pop())
    destination = path(prediction_round)
    if destination.exists():
        raise schema.LedgerError(
            f"{prediction_round} is already sealed. Supersede it with a new As-Of Instant in a "
            "later round rather than rewriting it (ADR 0005)"
        )

    deadline = pd.Timestamp(rows["kickoff"].min())
    moment = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    if moment >= deadline:
        raise schema.LedgerError(
            f"{prediction_round}'s first kickoff was {deadline}; it is {moment}. A Prediction "
            "sealed after kickoff is not evidence of anything"
        )
    # The audit itself runs inside write_csv, the one door both stores write through.
    return schema.write_csv(rows, destination)


def read() -> pd.DataFrame:
    """Every Sealed Prediction, in round order."""
    return schema.read_all(sealed_rounds())


def sealed_rounds() -> list[Path]:
    """The sealed round files, oldest first.

    Round ids are ISO dates, so name order is time order.
    """
    if not live_dir().exists():
        return []
    return sorted(file for file in live_dir().glob("*.csv") if _names_a_round(file))


def _names_a_round(file: Path) -> bool:
    """Whether a filename is a Prediction Round id — an ISO date that is a Tuesday or a Friday.

    Checked against the anchor rule rather than merely parsed, so a file named after a Saturday is
    reported as debris instead of being read as a round that could never have existed.
    """
    try:
        day = date.fromisoformat(file.stem)
    except ValueError:
        return False
    return anchor(day) == day


def seal_violations(*, now: pd.Timestamp | None = None) -> list[str]:
    """Every way the sealed store has been rewritten after the fact.

    Reads git rather than the files: the bytes on disk cannot say when they were written, and the
    whole claim this store makes is a claim about *when*.

    A round is in violation if any commit touching its file lands at or after the round's first
    kickoff, if it has uncommitted changes once that moment has passed, or if it was never
    committed at all — an uncommitted file proves nothing, and after kickoff there is no longer any
    way to prove it.
    """
    moment = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    complaints: list[str] = []

    for extra in _unexpected_files():
        complaints.append(f"{extra.name}: not a sealed Prediction Round, and does not belong here")

    present = {file.name for file in sealed_rounds()}
    for missing in sorted(_committed_rounds() - present):
        complaints.append(
            f"{missing}: was sealed and committed, and is no longer here. Append-only means "
            "nothing leaves either (ADR 0005)"
        )

    for file in sealed_rounds():
        rows = schema.read_csv(file)
        complaints += [f"{file.name}: {complaint}" for complaint in schema.audit(rows)]
        if set(rows["prediction_round"]) != {file.stem}:
            held = sorted(set(rows["prediction_round"]))
            complaints.append(f"{file.name}: holds rows from {held}")

        deadline = _localised(pd.Timestamp(rows["kickoff"].min()))
        commits = _commit_times(file)
        late = [when for when in commits if when >= deadline]
        if late:
            complaints.append(
                f"{file.name}: changed after its round's first kickoff — committed at "
                f"{late[-1].isoformat()}, kickoff was {deadline.isoformat()}"
            )
        if _localised(moment) < deadline:
            continue
        if not commits:
            complaints.append(
                f"{file.name}: not committed, and its round has kicked off — nothing now proves "
                "when it was written"
            )
        elif _is_dirty(file):
            complaints.append(
                f"{file.name}: has uncommitted changes after its round's first kickoff"
            )

    return complaints


def _committed_rounds() -> set[str]:
    """Every sealed round file git has a record of, whether or not it is still on disk.

    The working tree cannot report its own deletions: once a file is gone there is nothing left to
    notice. Git is what remembers a round was sealed at all, which is the same reason git is what
    proves when it was.
    """
    listing = _git("log", "--pretty=format:", "--name-only", "--", str(live_dir()))
    if listing is None:
        return set()
    return {
        name
        for line in listing.splitlines()
        if (name := PurePosixPath(line.strip()).name) and _names_a_round(Path(name))
    }


def _unexpected_files() -> list[Path]:
    if not live_dir().exists():
        return []
    return sorted(
        item
        for item in live_dir().iterdir()
        if item.is_file() and item.name not in ALLOWED_EXTRAS and not _names_a_round(item)
    )


def _localised(instant: pd.Timestamp) -> pd.Timestamp:
    """A naive UK-local instant as an absolute moment, comparable with a commit timestamp."""
    return instant.tz_localize(LOCAL_ZONE)


def _commit_times(file: Path) -> list[pd.Timestamp]:
    """When every commit that touched ``file`` was made, oldest first.

    Empty when the file is untracked, when there is no repository, or when git is not installed —
    all of which mean the same thing here: nothing proves when these bytes were written.
    """
    result = _git("log", "--format=%cI", "--", str(file))
    if result is None:
        return []
    return sorted(pd.Timestamp(line) for line in result.splitlines() if line)


def _is_dirty(file: Path) -> bool:
    return bool(_git("status", "--porcelain", "--", str(file)))


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root()), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:  # pragma: no cover - git missing from the machine entirely
        return None
    return result.stdout if result.returncode == 0 else None
