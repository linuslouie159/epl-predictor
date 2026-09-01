"""The schedule's own log, parsed back into the fires that wrote it.

`deploy/logs/live_loop.log` is the only place a scheduled run leaves a trace of having happened.
The sealed store records what a fire *wrote*, and most fires write nothing — two of the three
crontab lines are `seal --push` and the second is a retry designed to find the round already sealed
— so "did the loop run on Friday?" is a question only this file can answer.

**Why a log and not a status the loop reports.** Because the interesting cases are the ones where
the loop has nothing to report. A quiet exit 0 is correct twice a week (:func:`epl.live.__main__.
_seal`), and both of this project's live open risks look exactly like it: a stale upstream file
(risk 7) prints one of :mod:`epl.live.upcoming`'s two silences, and an off Pi (risk 6) prints
nothing at all. The second cannot be reported by the loop under any design — a process that is not
running cannot say so — so the reader has to be something that reads *absence*, and absence is a
gap between blocks.

Three shapes here are the shell's rather than Python's, and each is load-bearing:

* **Two spaces after `END`.** `run_live.sh` writes ``'===== END  %s  (exit %d) ====='``.
* **A trailing space in the command** when the subcommand took no arguments — ``(score )``.
* **The Pi's own zone on every stamp.** `deploy/crontab`'s times are converted into it because
  Raspberry Pi OS's cron ignores `CRON_TZ` (issue #21), so the retry slot reads `13:30 -0400` and
  means 18:30 in London. The offset is on the line, so it is kept and never assumed: a round's
  window is judged in UK time (:func:`epl.ledger.live.uk_now`) and a fire is stamped in the
  machine's, and the two are only comparable while both carry a zone.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from epl.live import upcoming
from epl.live.upcoming import ROLLING_FILE_PREFIX
from epl.paths import project_root

#: The one directory this project keeps machine-local runtime in: the schedule's log and lock, and
#: the bot's lock. Gitignored, because none of it is evidence — what a scheduled run produces that
#: *is* evidence is the committed round under `outputs/live/`.
RUNTIME_DIR = ("deploy", "logs")

#: Where `deploy/run_live.sh` writes, relative to the repository root. The bot reads the log of
#: whichever clone it is running inside, which on the Pi is the same bind-mounted checkout the loop
#: seals into — so the two agree by construction rather than by configuration.
LOG_RELATIVE_PATH = (*RUNTIME_DIR, "live_loop.log")

#: Where `prematch` fires are logged instead, and why they are not in the file above.
#:
#: `live_loop.log` is read by a person, and it holds three fires a week. `prematch` fires around
#: forty times on a matchday and the overwhelming majority of those write one line saying nothing
#: kicks off within the hour (:data:`epl.live.prematch.NOTHING_DUE`). Mixing them would bury the
#: three blocks that matter under a thousand that do not, in the one file somebody opens when the
#: schedule has gone wrong.
#:
#: They are still fires and `/health` still reports the last of them — :func:`read` takes a path
#: precisely so that a second log is a second argument rather than a second parser.
PREMATCH_LOG_RELATIVE_PATH = (*RUNTIME_DIR, "prematch.log")

#: Which log a subcommand's fires are written to. Anything not named here uses the loop's own.
LOGS: dict[str, tuple[str, ...]] = {"prematch": PREMATCH_LOG_RELATIVE_PATH}

#: The zone every round window is judged in. A fire's stamp is the *machine's*, and one of the two
#: has to be converted before they can be compared at all.
LOCAL_ZONE = "Europe/London"

#: What `run_live.sh` prints around each run. The `END` marker really does carry two spaces.
RUN_MARKER = "===== RUN "
END_MARKER = "===== END  "

#: How the shell's `date '+%Y-%m-%d %H:%M:%S %z'` reads.
STAMP_FORMAT = "%Y-%m-%d %H:%M:%S %z"

# Built from the markers above rather than repeating them. Two spellings of one shell format is how
# one of them comes to be updated and the other not — and the one that silently stopped matching
# would leave the bot reporting a schedule that never fires.
_RUN = re.compile(rf"^{re.escape(RUN_MARKER)}(?P<stamp>.+?)  \((?P<command>.*?)\) =====$")
_END = re.compile(rf"^{re.escape(END_MARKER)}(?P<stamp>.+?)  \(exit (?P<code>-?\d+)\) =====$")

#: What a fire turned out to be. The first three are the exit code restated; the last two are the
#: two silences it cannot express, because :class:`epl.live.upcoming.NothingToSeal` is deliberately
#: a success (issue #19) and both of them arrive spelt identically to a round that sealed cleanly.
WORKED = "worked"
FAILED = "failed"
UNFINISHED = "unfinished"
NOTHING_IN_FILE = "nothing in the rolling file"
OUTSIDE_EVERY_WINDOW = "outside every window"

#: The verdicts that mean the loop ran, exited 0, and sealed nothing. Grouped because that is the
#: population open risk 7 is a claim about, and a caller that had to remember which two they were
#: is a caller that will one day remember only one.
SILENCES: tuple[str, ...] = (NOTHING_IN_FILE, OUTSIDE_EVERY_WINDOW)


@dataclass(frozen=True)
class Fire:
    """One `===== RUN =====` block: when the schedule fired, what it ran, and how it went.

    ``exit_code`` is ``None`` for a block with no `END` line, which is a container killed or a Pi
    that went down mid-run. That is deliberately not folded into a failure: an exit code is the loop
    saying something, and its absence is the loop having been stopped before it could.
    """

    started: pd.Timestamp
    command: str
    finished: pd.Timestamp | None
    exit_code: int | None
    lines: tuple[str, ...]

    @property
    def subcommand(self) -> str:
        """`seal`, `score` or `upcoming` — the command without its flags."""
        return self.command.split(" ", 1)[0]

    @property
    def failed(self) -> bool:
        """Whether this fire needs somebody. An unfinished run does too."""
        return self.exit_code != 0

    @property
    def rolling_file(self) -> str | None:
        """The cached copy of `fixtures.csv` this fire read, or ``None`` if it read none.

        A durable handle rather than a passing detail: every fetch is kept under the instant it was
        taken (:func:`epl.ingest.fixtures.raw_fixtures_path`), so two fires naming two files whose
        bytes match saw a file upstream had not regenerated between them — which is the whole of
        :func:`epl.bot.watch.stale_upstream`.
        """
        for line in self.lines:
            if line.startswith(ROLLING_FILE_PREFIX):
                return line[len(ROLLING_FILE_PREFIX):].strip() or None
        return None

    @property
    def verdict(self) -> str:
        """Which of :data:`WORKED`, :data:`FAILED`, :data:`UNFINISHED` or a silence this was.

        A failure outranks a silence. The two cannot combine through `_seal` today, because
        `NothingToSeal` returns 0 before anything else can go wrong — but they can through
        ``--push``, which is stamped on the way out of a run that had nothing else to say, and a
        week in which the loop could not reach its remote is not a quiet week.
        """
        if self.exit_code is None:
            return UNFINISHED
        if self.exit_code != 0:
            return FAILED
        for line in self.lines:
            if upcoming.NO_FIXTURE_TO_PREDICT in line:
                return NOTHING_IN_FILE
            if upcoming.NONE_INSIDE_A_WINDOW in line:
                return OUTSIDE_EVERY_WINDOW
        return WORKED

    @property
    def local(self) -> pd.Timestamp:
        """When this fire started, read in UK time rather than in the machine's own.

        Every message about a fire quotes this rather than ``started``, and every comparison
        against a round's window is made through it. Here rather than at each call site so that
        `Europe/London` is named once: four callers each doing their own ``tz_convert`` is four
        places for one of them to stop doing it, and the one that stopped would report a fire
        several hours from where it happened without ever looking wrong.
        """
        return self.started.tz_convert(LOCAL_ZONE)

    @property
    def local_day(self) -> pd.Timestamp:
        """The UK day this fire happened on, which is the day its round would anchor to.

        The Pi's own date is not that day for half the evening: the 18:30 UK retry is 13:30 on a Pi
        five hours behind, and a round anchored to Friday can be sealed by a fire the Pi calls
        Friday afternoon — but the same conversion run the other way, on a machine ahead of the UK,
        puts a fire after midnight local on a day the round has never heard of.
        """
        return self.local.normalize()

    def tail(self, count: int = 12) -> str:
        """The last few lines, for a message that has to say what went wrong without a terminal."""
        return "\n".join(self.lines[-count:]).strip()


def uk_now() -> pd.Timestamp:
    """This moment as an **aware** UK instant, which is what a fire's stamp compares against.

    Deliberately not :func:`epl.ledger.live.uk_now`, and the difference is the whole reason both
    exist. That one is naive because everything it is compared against is naive: an As-Of Instant
    and a kickoff come off Football-Data as UK wall-clock readings with no zone, and pandas refuses
    to compare an aware timestamp with a naive one at all. Everything *here* carries an offset,
    because `deploy/run_live.sh` stamps every block with `date '+... %z'` on a machine whose zone is
    nobody's decision — so a naive reading would be the one thing that could not be compared.

    Two moments, two shapes, one zone. Reaching for the wrong one raises rather than misreports,
    which is the only reason it is safe to have both.
    """
    return pd.Timestamp.now(tz=LOCAL_ZONE)


def wall_clock(moment: pd.Timestamp) -> pd.Timestamp:
    """An aware instant as the naive UK wall-clock reading a kickoff is comparable with.

    The bot holds both shapes on purpose and must not mix them: :func:`uk_now` is aware because a
    fire's stamp carries the machine's offset, and every kickoff and As-Of Instant in the ledger is
    naive because that is how Football-Data writes them (:func:`epl.ledger.live.uk_now` says why).
    Comparing the two raises rather than misreporting, which is the only reason both are safe to
    have — and a bot that answers "the next match" has to cross the line, because the moment comes
    from one side and the kickoff from the other.

    So the crossing is made once, here, beside the pair it reconciles. Spelt inline at each call
    site it would be three ``tz_convert`` chains, and the one that stopped converting would answer
    with a match an hour in the past without ever looking wrong.

    A naive moment is passed through unchanged. That is not laxity: a test hands a naive instant
    because the store it is checking against is naive, and converting one that carries no zone
    would be inventing an offset.
    """
    stamped = pd.Timestamp(moment)
    if stamped.tz is None:
        return stamped
    return stamped.tz_convert(LOCAL_ZONE).tz_localize(None)


def log_path(subcommand: str | None = None) -> Path:
    """Where the schedule writes, in the clone this process is running inside.

    ``subcommand`` names which log, because `prematch` has its own (:data:`LOGS`). Defaulting to
    the loop's keeps every existing caller reading the file it always read.
    """
    return project_root().joinpath(*LOGS.get(str(subcommand), LOG_RELATIVE_PATH))


def read(path: Path | None = None, *, subcommand: str | None = None) -> tuple[Fire, ...]:
    """Every fire the log holds, oldest first.

    A log that has never been written is no fires rather than an error: on a fresh clone the bot
    starts before the first fire, and there is nothing wrong with that. Undecodable bytes are
    replaced rather than raised on for a stronger reason — this file is machine-local, unversioned
    and written by a shell redirect, and a notifier that fell over on a truncated write would be
    silent in exactly the way it exists to prevent.
    """
    log = log_path(subcommand) if path is None else Path(path)
    if not log.exists():
        return ()
    return parse(log.read_text(encoding="utf-8", errors="replace"))


def read_every() -> tuple[Fire, ...]:
    """Every fire from every log this schedule writes, oldest first.

    What `/health` reports over, because a reader asking whether the schedule is alive means all of
    it. Sorted across the files rather than concatenated: the two logs are written by the same
    crontab on the same machine, so their stamps interleave, and a listing that put forty `prematch`
    fires after Tuesday's `seal` would misreport the order things happened in.
    """
    seen = {LOG_RELATIVE_PATH, *LOGS.values()}
    found = [fire for relative in seen for fire in read(project_root().joinpath(*relative))]
    return tuple(sorted(found, key=lambda fire: fire.started))


def parse(text: str) -> tuple[Fire, ...]:
    """The fires in a log's text, oldest first.

    Lines outside any block are dropped, and two of them matter: `run_live.sh` writes a missing-
    `flock` complaint and a stand-down when a second run finds the lock held. Neither is a fire —
    the stand-down is the *absence* of one — and counting either would report a loop that did
    nothing as having run.
    """
    found: list[Fire] = []
    started: pd.Timestamp | None = None
    command = ""
    body: list[str] = []

    def close(finished: pd.Timestamp | None, exit_code: int | None) -> None:
        if started is None:
            return
        # `run_live.sh` opens each block with a leading newline, so the blank line before the next
        # `RUN` marker belongs to nobody. It falls inside the previous body only when that body was
        # never closed by an `END` line — which is exactly the case a crashed run produces.
        kept = list(body)
        while kept and not kept[-1]:
            kept.pop()
        found.append(
            Fire(
                started=started,
                command=command,
                finished=finished,
                exit_code=exit_code,
                lines=tuple(kept),
            )
        )

    for line in text.splitlines():
        stripped = line.rstrip()
        if (opened := _RUN.match(stripped)) is not None:
            # A block with no `END` line is closed by the next block rather than absorbing it.
            close(None, None)
            started = _stamp(opened["stamp"])
            command = opened["command"].strip()
            body = []
            continue
        if started is None:
            continue
        if (ended := _END.match(stripped)) is not None:
            close(_stamp(ended["stamp"]), int(ended["code"]))
            started = None
            continue
        body.append(stripped)

    close(None, None)
    return tuple(found)


def latest(found: Iterable[Fire], *, subcommand: str | None = None) -> Fire | None:
    """The most recent fire, optionally of one subcommand only.

    ``None`` when there has never been one, which the caller has to handle rather than be handed a
    default: a schedule that has never fired and a schedule that fired and failed are the two
    different alarms this module exists to tell apart.
    """
    wanted = [
        fire for fire in found if subcommand is None or fire.subcommand == subcommand
    ]
    return max(wanted, key=lambda fire: fire.started) if wanted else None



def _stamp(raw: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(raw.strip(), STAMP_FORMAT))


__all__ = [
    "END_MARKER",
    "FAILED",
    "LOCAL_ZONE",
    "LOGS",
    "LOG_RELATIVE_PATH",
    "NOTHING_IN_FILE",
    "OUTSIDE_EVERY_WINDOW",
    "PREMATCH_LOG_RELATIVE_PATH",
    "RUNTIME_DIR",
    "RUN_MARKER",
    "SILENCES",
    "STAMP_FORMAT",
    "UNFINISHED",
    "WORKED",
    "Fire",
    "latest",
    "log_path",
    "parse",
    "read",
    "read_every",
    "uk_now",
    "wall_clock",
]
