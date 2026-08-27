"""The sealed store: one file per Prediction Round, written before kickoff and never rewritten.

A Sealed Prediction is evidence, and evidence that can be silently regenerated is not evidence
(ADR 0005). So `outputs/live/` is append-only and committed, and git history — not a timestamp
inside the file, which anyone could type — is the proof of when each Prediction existed.

The concrete failure this guards against: Football-Data's current-Season file grows weekly and
backfills results and odds into rows already published. Regenerating last Friday's live Prediction
a month later can feed it odds that did not exist when the Prediction was claimed to have been
made, and nothing in the code would flag it.

**Three moments matter and they are different.** :func:`seal` refuses to write a round outside its
own window — before its As-Of Instant, when the moment it would claim has not happened, or once its
first Fixture has kicked off. :func:`supersede` is the only way to correct a round already in the
store, and writes a new revision of it rather than new bytes in the old file.
:func:`seal_violations` reports rounds whose bytes *changed* after kickoff. The first two are guards
on the way in, the third an audit of what is already there — and only the third survives someone
bypassing the code and editing a file by hand.

A round's file is named after the round, and a superseding revision after the round and its
revision number: ``2026-08-28.csv``, then ``2026-08-28.1.csv``. Name order is therefore no longer
time order, which is why :func:`sealed_rounds` sorts on the round and revision it parses out rather
than on the filename.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
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

#: What separates a round id from the revision of a superseding file. A full stop, so the stem of
#: ``2026-08-28.1.csv`` is ``2026-08-28.1`` and the round it corrects is legible without parsing.
REVISION_SEPARATOR = "."

#: Which side of a round's sealing window a moment falls on — see :func:`window`.
SEALABLE = "sealable"
NOT_OPEN = "not open"
KICKED_OFF = "kicked off"


def window(as_of: pd.Timestamp, first_kickoff: pd.Timestamp, now: pd.Timestamp) -> str:
    """Whether a round whose window is ``as_of`` to ``first_kickoff`` may be sealed at ``now``.

    The rule stated once. Two callers need it and they must not be able to disagree: this module
    refuses a write outside the window, and :func:`epl.live.upcoming.rounds` chooses which round to
    offer. A second copy of the two comparisons is how the loop would come to offer a round the
    store then refused, or — far worse — the other way round.

    Both ends are refusals rather than warnings. Before the As-Of Instant, a Prediction claims a
    moment that has not happened and reads odds that do not exist yet; at or after the first
    kickoff, it is a claim about what the code would have said, which is the one thing this store
    must never hold (ADR 0005).
    """
    if pd.Timestamp(now) < pd.Timestamp(as_of):
        return NOT_OPEN
    if pd.Timestamp(now) >= pd.Timestamp(first_kickoff):
        return KICKED_OFF
    return SEALABLE


def path(prediction_round: str, *, revision: int = 0) -> Path:
    """Where one Prediction Round's Sealed Predictions live. The round id is the filename.

    ``revision`` names a superseding reading of the same round (:func:`supersede`). Revision 0 is
    the round as first sealed and is the only one with a bare name, so a store that has never been
    corrected looks exactly as it did before superseding existed.
    """
    if revision < 0:
        raise schema.LedgerError(f"a revision is 0 or more; got {revision}")
    stem = (
        prediction_round
        if revision == 0
        else f"{prediction_round}{REVISION_SEPARATOR}{revision}"
    )
    return live_dir() / f"{stem}.csv"


def seal(rows: pd.DataFrame, *, now: pd.Timestamp | None = None) -> Path:
    """Write one Prediction Round's Predictions, from every Predictor, and return the path.

    Refuses a round that is already sealed, and refuses one whose window is not open: at or after
    its As-Of Instant, and strictly before its first kickoff. All three are the same rule from
    three directions — what is in this directory has to be what was actually forecast, at the
    moment it says it was, rather than what the code says today.

    Correcting a round already here is :func:`supersede`, never this.
    """
    prediction_round = _one_round(rows)
    if is_sealed(prediction_round):
        raise schema.LedgerError(
            f"{prediction_round} is already sealed. Correct it with a superseding revision at a "
            "new As-Of Instant rather than rewriting it (ADR 0005)"
        )
    return _write(rows, path(prediction_round), prediction_round, now=now)


def is_sealed(prediction_round: str) -> bool:
    """Whether this round has been written yet.

    An accessor rather than a path check at the call site, so that what a sealed round looks like on
    disk stays inside this module — a caller asking "is it there?" must not need to know that the
    answer is a filename.
    """
    return path(prediction_round).exists()


def supersede(rows: pd.DataFrame, *, now: pd.Timestamp | None = None) -> Path:
    """Write a corrected reading of a round that is already sealed, as a new revision of it.

    ADR 0005's "adding a superseding row with a new As-Of Instant, never editing history", made
    into the only door that does it. Every row must be stamped strictly later than what the store
    already holds for that Predictor and Fixture, because a Prediction that replaces another and
    claims the same instant is indistinguishable from that other one having been rewritten — which
    is the thing this store exists to make impossible.

    A superseding Prediction may know more than the one it replaces: its Evidence is cut at its own
    later instant (:func:`epl.ledger.schema.predictions_for`). That is what actually happened, and
    it is why superseding is for correcting a bug rather than for refreshing a quote.

    **The round must still be open, and a bug found after kickoff cannot be corrected here at all.**
    That is the rule rather than a limitation. This store holds what was forecast *before* kickoff;
    a row written after it could not be evidence of that whatever it said, so admitting one would
    cost the store the only thing it has. A bug found at scoring time is fixed in the code, so the
    next round is right, and the sealed row stands and is scored as made. A track record that could
    be tidied up afterwards would not be a track record.
    """
    prediction_round = _one_round(rows)
    if not is_sealed(prediction_round):
        raise schema.LedgerError(
            f"{prediction_round} has never been sealed, so there is nothing to supersede. "
            "`seal` writes a round for the first time"
        )

    already = schema.read_all(revisions_of(prediction_round))
    _check_moves_forward(rows, already)
    return _write(
        rows, path(prediction_round, revision=_next_revision(prediction_round)),
        prediction_round, now=now,
    )


def _one_round(rows: pd.DataFrame) -> str:
    """The one Prediction Round these rows belong to, or a complaint that there is more than one."""
    rounds = set(rows["prediction_round"])
    if len(rounds) != 1:
        raise schema.LedgerError(
            f"a sealed file holds one Prediction Round; got {sorted(rounds)} (ADR 0005)"
        )
    return str(rounds.pop())


def _write(
    rows: pd.DataFrame, destination: Path, prediction_round: str, *, now: pd.Timestamp | None
) -> Path:
    """The window check both doors share, and then the one write both go through.

    The audit itself runs inside :func:`epl.ledger.schema.write_csv`, which is the single door both
    stores write through.
    """
    opening = pd.Timestamp(rows["as_of_instant"].min())
    deadline = pd.Timestamp(rows["kickoff"].min())
    moment = pd.Timestamp.now() if now is None else pd.Timestamp(now)

    verdict = window(opening, deadline, moment)
    if verdict == NOT_OPEN:
        raise schema.LedgerError(
            f"{prediction_round} is stamped {opening}; it is {moment}. A Prediction cannot be "
            "sealed at a moment that has not happened yet"
        )
    if verdict == KICKED_OFF:
        raise schema.LedgerError(
            f"{prediction_round}'s first kickoff was {deadline}; it is {moment}. A Prediction "
            "sealed after kickoff is not evidence of anything"
        )
    return schema.write_csv(rows, destination)


def _check_moves_forward(rows: pd.DataFrame, already: pd.DataFrame) -> None:
    """Every superseding row must be stamped later than the row it replaces."""
    if already.empty:
        return
    latest = already.groupby(["predictor", *schema.FIXTURE_KEY])["as_of_instant"].max()
    keyed = rows.set_index(["predictor", *schema.FIXTURE_KEY])["as_of_instant"]
    shared = keyed.index.intersection(latest.index)
    stale = keyed.loc[shared] <= latest.loc[shared]
    if stale.any():
        first = stale.loc[stale].index[0]
        raise schema.LedgerError(
            f"{int(stale.sum())} superseding Predictions are stamped at or before the ones they "
            f"replace — {first} at {keyed.loc[first]}, already sealed at {latest.loc[first]}. "
            "Superseding needs a new As-Of Instant (ADR 0005)"
        )


def _next_revision(prediction_round: str) -> int:
    """The first revision number this round has not used."""
    return max(revision for _, revision, _ in _stored(prediction_round)) + 1


def revisions_of(prediction_round: str) -> list[Path]:
    """Every file in the store holding this round, oldest revision first."""
    return [file for _, _, file in _stored(prediction_round)]


def read() -> pd.DataFrame:
    """Every Sealed Prediction, in round order."""
    return schema.read_all(sealed_rounds())


def sealed_rounds() -> list[Path]:
    """The sealed round files, oldest first, with each round's revisions in order after it."""
    return [file for _, _, file in _stored()]


def _stored(prediction_round: str | None = None) -> list[tuple[str, int, Path]]:
    """Every sealed file as (round, revision, path), oldest first — optionally for one round.

    The one place a filename is turned back into what it names, so every caller that needs the
    round or the revision has it rather than re-parsing.

    Sorted on the round and revision rather than on the name: round ids are ISO dates so they sort
    as time, but ``2026-08-28.1.csv`` sorts *before* ``2026-08-28.csv`` as a string, which would put
    a correction ahead of what it corrects.
    """
    if not live_dir().exists():
        return []
    found = [
        (*parsed, file)
        for file in live_dir().glob("*.csv")
        if (parsed := _round_of(file)) is not None
        and (prediction_round is None or parsed[0] == prediction_round)
    ]
    return sorted(found, key=lambda entry: entry[:2])


def _round_of(file: Path) -> tuple[str, int] | None:
    """The Prediction Round a sealed filename names and which revision of it, or ``None``.

    Checked against the anchor rule rather than merely parsed, so a file named after a Saturday is
    reported as debris instead of being read as a round that could never have existed.
    """
    stem, _, revision = file.stem.partition(REVISION_SEPARATOR)
    try:
        day = date.fromisoformat(stem)
    except ValueError:
        return None
    if anchor(day) != day:
        return None
    if not revision:
        return stem, 0
    return (stem, int(revision)) if revision.isdigit() and int(revision) > 0 else None


def _round_of_or_raise(file: Path) -> tuple[str, int]:
    """:func:`_round_of` for callers that have already filtered out the files it refuses."""
    parsed = _round_of(file)
    if parsed is None:  # pragma: no cover - callers pass files from `sealed_rounds`
        raise schema.LedgerError(f"{file.name} does not name a Prediction Round")
    return parsed


def _names_a_round(file: Path) -> bool:
    """Whether a filename is a Prediction Round id, optionally with a revision number."""
    return _round_of(file) is not None


def commit(paths: Iterable[Path], *, message: str) -> str | None:
    """Commit these sealed files, and return the commit's hash.

    Sealing and committing are two steps and the store is only evidence after the second: an
    uncommitted file proves nothing about when it was written, which :func:`seal_violations` says
    out loud once the round has kicked off. So the loop does both, and does the second here —
    beside :func:`_git`, because one module knowing about git is what makes "git history is the
    proof" a claim with a single place to check it.

    Only the named files are staged, so a run that seals a round never sweeps up whatever else was
    in the working tree. Returns ``None`` when there was nothing to commit or when git could not be
    reached — both leave the file unproven, and the caller has to say so rather than assume.
    """
    wanted = [str(file) for file in paths]
    if not wanted:
        return None
    if _git("add", "--", *wanted) is None:
        return None
    # `--quiet` exits 1 when there *are* staged differences, so `_git` returning a string here
    # means there are none — these bytes are already committed and a second run must add nothing.
    if _git("diff", "--cached", "--quiet", "--", *wanted) is not None:
        return None
    if _git("commit", "-m", message, "--", *wanted) is None:
        return None
    return (_git("rev-parse", "HEAD") or "").strip() or None


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
        prediction_round, _ = _round_of_or_raise(file)
        complaints += [f"{file.name}: {complaint}" for complaint in schema.audit(rows)]
        if set(rows["prediction_round"]) != {prediction_round}:
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

    return complaints + _superseding_violations()


def _superseding_violations() -> list[str]:
    """Whether the store's revisions read as corrections rather than as rewrites.

    Checked across files, where :func:`epl.ledger.schema.audit` can only see within one. Two rows
    naming the same Predictor, the same Fixture and the same As-Of Instant are the shape a rewrite
    would take if it were done by adding a file instead of editing one, and the scoreboard — which
    keeps the latest instant per Fixture — would silently pick one of them.
    """
    stored = read()
    if stored.empty:
        return []

    key = ["predictor", *schema.FIXTURE_KEY, "as_of_instant"]
    doubled = stored.duplicated(subset=key, keep=False)
    if not doubled.any():
        return []
    first = stored.loc[doubled].iloc[0]
    return [
        f"{int(doubled.sum())} Sealed Predictions across the store repeat a Fixture at one As-Of "
        f"Instant — {first['predictor']} on {first['home_club']} v {first['away_club']} at "
        f"{first['as_of_instant']}. A superseding revision needs a new instant (ADR 0005)"
    ]


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
