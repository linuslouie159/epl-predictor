"""The two silences the loop cannot break for itself: open risks 6 and 7.

Both look identical from `deploy/logs/live_loop.log`, and that is the whole problem. The loop's
exit code is honest and its silence is correct — :class:`epl.live.upcoming.NothingToSeal` is a
success on purpose, because most of the week no round is inside its window and a job that goes red
twice a week for a season is one whose owner stops reading it (issue #19). What that costs is the
one distinction a person actually wants:

**Open risk 7 — a stale upstream file.** `fixtures.csv` is regenerated irregularly. A fetch landing
between regenerations sees a copy from two days ago, and the loop correctly reports having nothing
to seal. The Pi is up, the container exits 0, the log has its block, and the round is gone. What
makes it checkable at all is that every fetch is cached under the instant it was taken
(:func:`epl.ingest.fixtures.raw_fixtures_path`): two fires inside one round's window whose cached
copies are **byte-identical** read a file upstream had not touched between them. That is the
ticket's "`Last-Modified` not having moved", taken off bytes this project already keeps rather than
off a header it does not record.

**Open risk 6 — an off Pi.** No block at all. This one cannot be detected while it is happening,
by anything on this machine, ever: a process that is not running cannot report that it is not
running, and the bot is on the same box as the loop. What is buildable is *retrospective* — when
the bot comes back, it can say which anchor days went by unfired. That is worth having and it is
**not a dead man's switch**, and the difference matters: an outage that is still going is not
reported by this, and closing that gap needs a second machine, which is the same argument
docs/DECISIONS.md made against a Pi in the first place.

Two things this module deliberately does not do.

It does not claim a round was **lost**. An unchanged upstream file means nobody can tell whether
there was football this week, and "cannot tell" is the finding. A message announcing a lost round
would be wrong through most of the summer, and a channel that is wrong through most of the summer
is not read in September.

It does not read `deploy/crontab` to learn which days to expect a fire on. The schedule fires on
Tuesdays and Fridays *because* a Prediction Round anchors to one (:func:`epl.rounds.anchor`), so
that is what is asked. Reading cron would make this agree with the schedule rather than with the
thing the schedule was derived from — and a hand-edited crontab is exactly the case where those
two differ and somebody should hear about it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from epl.bot import fires
from epl.ingest.fixtures import fixtures_dir
from epl.rounds import FRIDAY, TUESDAY

#: Which risk a concern is about, numbered as docs/DECISIONS.md numbers them. Named constants
#: rather than bare integers, because "6" and "7" in a message are meaningless without the document
#: and the whole point of a concern is to be readable from a phone.
NO_FIRE_AT_ALL = 6
STALE_UPSTREAM = 7

#: The days a Prediction Round anchors to, and therefore the days the schedule fires on.
SCHEDULE_DAYS: tuple[int, ...] = (TUESDAY, FRIDAY)

#: When a fire is late rather than pending. The last scheduled slot is 18:30 UK — the retry — so an
#: anchor day is missed only once an hour of grace past it has gone by. Complaining at 17:00 on a
#: Friday would be complaining about a schedule that has one more fire to come.
EXPECTED_BY = pd.Timedelta(hours=19, minutes=30)

#: How far back :func:`absent` will look. A bot restarted after a month away must not post a month
#: of missed Fridays: what the reader can still act on is the last week or two, and the rest is
#: history they already know about because they were the ones who unplugged it.
LOOKBACK_DAYS = 14


@dataclass(frozen=True)
class Concern:
    """One thing worth interrupting somebody about, and which documented risk it is.

    ``risk`` is the number in docs/DECISIONS.md's open-risks list, so a message can be traced back
    to the paragraph that predicted it rather than read as a new and unexplained alarm.
    """

    risk: int
    headline: str
    detail: str

    def message(self) -> str:
        """The whole concern as one message, naming the risk so it can be looked up."""
        return f"{self.headline}\n{self.detail}\n(docs/DECISIONS.md, open risk {self.risk})"


def concerns(
    found: Sequence[fires.Fire],
    *,
    now: pd.Timestamp,
    directory: Path | None = None,
    lookback_days: int = LOOKBACK_DAYS,
) -> tuple[Concern, ...]:
    """Everything worth saying about the schedule's health, at most one per risk.

    Ordered absence first: a Pi that was off explains a great deal, including fires that never
    happened to read a stale file, and reading that second would be reading the consequence before
    the cause.
    """
    raised = [
        absent(found, now=now, lookback_days=lookback_days),
        stale_upstream(found, directory=directory),
    ]
    return tuple(concern for concern in raised if concern is not None)


def stale_upstream(
    found: Sequence[fires.Fire], *, directory: Path | None = None
) -> Concern | None:
    """Whether both of a round's fires read a rolling file upstream had not regenerated.

    Looks only at the most recent UK day on which any `seal` fire happened, and only when every one
    of that day's fires was silent — a day that sealed something has already answered the question.
    Two fires whose cached copies match byte for byte saw the same file, so whether there was
    Premier League football that week is precisely what the loop could not learn.

    ``None`` whenever the evidence is not there: one fire alone, a day that sealed, a file that has
    changed, or a cached copy that is no longer on disk. `data/raw/` is a cache and this module
    never writes to it, so missing bytes are missing rather than suspicious.
    """
    sealing = [fire for fire in found if fire.subcommand == "seal"]
    if not sealing:
        return None

    # Grouped by the *UK* day, because that is the day a round anchors to. The Pi is five hours
    # behind it and a machine in Asia is eight ahead, which splits one round's two fires across two
    # of that machine's own days — and two groups of one compare nothing.
    latest_day = max(fire.local_day for fire in sealing)
    today = [fire for fire in sealing if fire.local_day == latest_day]
    if len(today) < 2:
        return None
    if any(fire.verdict not in fires.SILENCES for fire in today):
        return None

    seen = [(fire, _bytes_of(fire.rolling_file, directory)) for fire in today]
    read_bytes = [content for _, content in seen if content is not None]
    if len(read_bytes) < 2 or len(set(read_bytes)) != 1:
        return None

    names = ", ".join(str(fire.rolling_file) for fire, content in seen if content is not None)
    clock = ", ".join(fire.local.strftime("%H:%M") for fire in today)
    return Concern(
        risk=STALE_UPSTREAM,
        headline=(
            f"⚠️ {latest_day.date()}: both fires read the same unchanged fixtures file, "
            "so nothing could be sealed and nobody can tell whether there was a round"
        ),
        detail=(
            f"{len(today)} `seal` fires at {clock} UK read {names} — identical bytes, so upstream "
            "did not regenerate `fixtures.csv` across the window. A genuinely empty week and a "
            "week whose round was missed look exactly like this. Worth checking "
            "football-data.co.uk/fixtures.csv by hand."
        ),
    )


def absent(
    found: Sequence[fires.Fire],
    *,
    now: pd.Timestamp,
    lookback_days: int = LOOKBACK_DAYS,
) -> Concern | None:
    """Which anchor days went by with no `seal` fire on them at all.

    Measured from the first fire in the log rather than from the beginning of time, so a fresh clone
    is not reported as an outage: a bot announcing open risk 6 on its first start would be
    announcing its own installation. And bounded by ``lookback_days`` at the other end, so a bot
    restarted after a long absence reports the part its reader can still act on.

    A `score` fire does not stand in for a `seal` fire. Missing the morning scoring run costs
    nothing — it is retrospective and the next one catches up — and missing the sealing run costs a
    round that `supersede` refuses to write afterwards, on purpose (ADR 0005).
    """
    sealing = [fire for fire in found if fire.subcommand == "seal"]
    if not sealing:
        return None

    moment = pd.Timestamp(now).tz_convert(fires.LOCAL_ZONE)
    first = min(fire.local_day for fire in sealing)
    earliest = max(first, (moment - pd.Timedelta(days=lookback_days)).normalize())
    fired_on = {fire.local_day.date() for fire in sealing}

    missed = [
        day
        for day in _anchor_days(earliest.date(), moment.normalize().date())
        if day not in fired_on
        and moment >= pd.Timestamp(day, tz=fires.LOCAL_ZONE) + EXPECTED_BY
    ]
    if not missed:
        return None

    days = ", ".join(day.isoformat() for day in missed)
    return Concern(
        risk=NO_FIRE_AT_ALL,
        headline=(
            f"🔴 {len(missed)} scheduled sealing day(s) with no run at all: {days}"
        ),
        detail=(
            "The loop cannot report its own absence, so this is read backwards out of the gaps in "
            "deploy/logs/live_loop.log — which means an outage still in progress is not what you "
            "are looking at. A round whose window passed while the Pi was off cannot be sealed "
            "afterwards. Check the Pi, docker, and the crontab."
        ),
    )


def _anchor_days(first: date, last: date) -> list[date]:
    """Every Tuesday and Friday from ``first`` to ``last`` inclusive."""
    span = pd.date_range(first, last, freq="D")
    return [day.date() for day in span if day.weekday() in SCHEDULE_DAYS]


def _bytes_of(name: str | None, directory: Path | None) -> bytes | None:
    """The cached rolling file a fire read, or ``None`` if it is not on disk any more."""
    if not name:
        return None
    path = (fixtures_dir() if directory is None else directory) / name
    try:
        return path.read_bytes()
    except OSError:
        return None


__all__ = [
    "EXPECTED_BY",
    "LOOKBACK_DAYS",
    "NO_FIRE_AT_ALL",
    "SCHEDULE_DAYS",
    "STALE_UPSTREAM",
    "Concern",
    "absent",
    "concerns",
    "stale_upstream",
]
