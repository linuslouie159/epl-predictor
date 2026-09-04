"""`deploy/crontab` parsed into the fires it actually schedules.

Its own module rather than a helper in `tests/conftest.py`, and the reason is the mistake that was
made first: two test packages need this, `from conftest import ...` resolved to
`tests/bot/conftest.py` on a full run, and which conftest won depended on collection order. The
same shape as `tests/bot/log_blocks.py` — a plain module beside the tests, imported by the name
pytest puts on `sys.path` — with a name nothing else in the tree can claim.

What it is *for* is `tests/deploy/test_the_schedule_does_not_collide.py`, which explains the defect
it exists to catch. `tests/live/test_prematch.py` reads it too, so the cadence its window depends
on is the cadence the crontab actually has rather than a literal restating it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: The schedule as committed. Read rather than restated, because two tests in different packages
#: need it and a second copy of "the cadence is every half hour" is how one of them comes to be
#: updated and the other not — which is close to what went wrong: `tests/live/test_prematch.py`
#: checked the cadence against a hardcoded `range(0, 24 * 60, 30)`, so it would have gone on
#: passing against any crontab at all.
CRONTAB = Path(__file__).resolve().parents[1] / "deploy" / "crontab"


@dataclass(frozen=True)
class CronLine:
    """One scheduled entry of `deploy/crontab`, with its time fields expanded to the values they
    actually fire on.

    Days of the month and months are not modelled: every line in this file leaves both `*`, and
    :func:`crontab_lines` refuses one that does not rather than quietly assuming it.
    """

    minutes: frozenset[int]
    hours: frozenset[int]
    days_of_week: frozenset[int]
    command: str

    @property
    def subcommand(self) -> str:
        """`seal`, `score` or `prematch` — what `run_live.sh` is being asked to do."""
        after_script = self.command.split("run_live.sh", 1)[-1].split()
        return after_script[0] if after_script else ""

    def collides_with(self, other: CronLine) -> frozenset[tuple[int, int, int]]:
        """The `(day, hour, minute)` slots on which both of these fire at once.

        Two lines that share a slot are two containers cron starts in the same second, contending
        for one lock (`deploy/run_live.sh`). One of them stands down or waits; which is which used
        to be whichever the kernel handed the lock to.
        """
        return frozenset(
            (day, hour, minute)
            for day in self.days_of_week & other.days_of_week
            for hour in self.hours & other.hours
            for minute in self.minutes & other.minutes
        )


def _expand(field: str, whole: range) -> frozenset[int]:
    """A cron time field as the set of values it fires on.

    `*`, `a`, `a,b` and `a-b` are the four spellings `deploy/crontab` uses. Anything else raises
    rather than returning something: a step like `*/5` read as a literal would produce an empty
    set, and an empty set collides with nothing — so an unparsed field would report the schedule
    as safe, which is the one answer this must not be able to give by accident.
    """
    if field == "*":
        return frozenset(whole)
    found: set[int] = set()
    for part in field.split(","):
        if "-" in part:
            low, high = part.split("-", 1)
            found.update(range(int(low), int(high) + 1))
        else:
            found.add(int(part))
    return frozenset(found)


def crontab_lines(path: Path | None = None) -> tuple[CronLine, ...]:
    """Every scheduled entry of `deploy/crontab`, in file order.

    Comments and blanks are skipped. Sunday is 0 and never 7, which this file never writes.
    """
    lines: list[CronLine] = []
    # A BOM would leave the first comment line not starting with `#`, and it would then be parsed
    # as an entry and raise about a field it does not have. This file is developed on Windows.
    text = (path or CRONTAB).read_text(encoding="utf-8").lstrip("\ufeff")
    for raw in text.splitlines():
        entry = raw.strip()
        if not entry or entry.startswith("#") or "=" in entry.split()[0]:
            continue
        minute, hour, day_of_month, month, day_of_week, command = entry.split(maxsplit=5)
        assert (day_of_month, month) == ("*", "*"), (
            f"{entry!r} restricts the day of the month or the month, which crontab_lines does not "
            f"model — teach it before adding such a line, or the collision check silently narrows"
        )
        lines.append(
            CronLine(
                minutes=_expand(minute, range(60)),
                hours=_expand(hour, range(24)),
                days_of_week=_expand(day_of_week, range(7)),
                command=command,
            )
        )
    return tuple(lines)
