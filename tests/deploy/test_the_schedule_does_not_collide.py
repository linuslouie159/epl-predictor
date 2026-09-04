"""Two cron lines that fire in the same second, and the lock that has to choose between them.

All four entries of `deploy/crontab` invoke one wrapper, and that wrapper takes one lock — because
`prematch --push` commits into the same bind-mounted checkout the loop seals into, and two
containers committing at once is a corrupt index. The lock is right. What was missing is that the
four fires are not equally important.

**What happened.** Stage 18 added `prematch` at `0,30 11-22`, which is every half hour on the hour
and the half hour. The two `seal --push` lines fire at 16:00 and 18:30 — both of them on a `0,30`
minute. So every Tuesday and Friday, *both* of a round's sealing chances raced a `prematch` fire
for the same lock, and `flock -n` has no priority: whichever process cron started first took it and
the other printed one line and exited 0. Measured on the Pi at 2026-09-01 11:00:01 -0400 (16:00 UK):

    live_loop.log:  live loop already running (seal); standing down
    prematch.log:   ===== RUN 2026-09-01 11:00:01 -0400  (prematch --push) =====
                    ===== END  2026-09-01 11:00:04 -0400  (exit 0) =====

The seal lost, to a fire that had nothing to do and was finished three seconds later. The round
survived only because the 18:30 retry happened to win its own race, and a round that loses both is
lost for good — `supersede` refuses a round at or after its first kickoff (ADR 0005).

**Why no existing test saw it.** Every test of this schedule reads one line at a time: whether the
script is committed executable, whether the window catches every kickoff. A collision is a property
of a *pair*, and nothing looked at pairs. It was also invisible in the logs — `run_live.sh` exited
before the notifier and `epl.bot.fires.parse` drops lines outside a `===== RUN` block on purpose —
so the schedule reported exit 0 twice a week while skipping half its chances to seal.

**Both halves are checked here.** The lock now gives the loop priority, which is the fix; the
schedule no longer collides, which is what keeps that fix cold. Either alone would work and
together they fail independently, which is the point of having both.

This reads the committed `deploy/crontab` rather than the Pi's installed copy. That is the right
file: the installed one is this one with the three loop times translated into the Pi's own zone
(issue #21), and a whole-hour translation cannot create or remove a collision on the minute.
"""

from __future__ import annotations

from itertools import combinations

import pytest

from epl.paths import project_root
from the_schedule import CronLine  # tests/ is on sys.path, as tests/bot/log_blocks.py relies on

#: What `run_live.sh` must let stand down at once, and what it must make wait. A `prematch` fire
#: that yields costs at most one of the two chances a Fixture gets at a card; a `seal` that yields
#: can cost the round.
YIELDS = "prematch"


@pytest.fixture
def wrapper() -> str:
    return (project_root() / "deploy" / "run_live.sh").read_text(encoding="utf-8")


def test_the_crontab_parses_into_the_four_lines_it_documents(
    schedule: tuple[CronLine, ...],
) -> None:
    """A guard on the parser rather than on the schedule: a parse that silently found nothing would
    make every collision check below pass by vacuum."""
    assert [line.subcommand for line in schedule] == ["score", "seal", "seal", "prematch"]


def test_no_two_scheduled_fires_land_in_the_same_minute(schedule: tuple[CronLine, ...]) -> None:
    """The check that would have caught it, stated over pairs because that is what the defect was.

    Not "the loop wins" — that is the lock's job and is checked below. This is the stronger and
    cheaper claim: cron never starts two of these at once in the first place, so nothing has to
    win.
    """
    collisions = {
        (one.subcommand, other.subcommand): shared
        for one, other in combinations(schedule, 2)
        if (shared := one.collides_with(other))
    }

    assert not collisions, (
        f"two crontab lines fire in the same minute and will contend for one lock: "
        f"{ {pair: sorted(slots)[:3] for pair, slots in collisions.items()} }. "
        f"Move one of them off the shared minute — `prematch` fires on :05 and :35 rather than "
        f"the hour and the half hour for exactly this reason."
    )


def test_prematch_still_fires_twice_an_hour(schedule: tuple[CronLine, ...]) -> None:
    """Moving it off the hour must not have moved it off the cadence the window depends on.

    `tests/live/test_prematch.py` checks that the cadence catches every kickoff. This checks that
    the cadence is still half-hourly at all, so a shift that quietly halved it fails here rather
    than as a match with no card in November.
    """
    (prematch,) = [line for line in schedule if line.subcommand == YIELDS]
    minutes = sorted(prematch.minutes)

    assert len(minutes) == 2, f"prematch fires {len(minutes)} times an hour, not twice: {minutes}"
    assert minutes[1] - minutes[0] == 30, (
        f"prematch's two fires are not half an hour apart: {minutes}"
    )


def test_only_prematch_stands_down_when_the_lock_is_held(wrapper: str) -> None:
    """The priority the crontab's prose has always claimed, checked in the script that decides it.

    Text rather than execution, because the behaviour needs two concurrent containers and a Linux
    `flock` to observe, and the mistake this guards against is a one-character edit made on a
    machine that has neither.
    """
    assert 'if [ "$subcommand" = "prematch" ]; then\n    lock_wait=(-n)' in wrapper, (
        "run_live.sh no longer gives prematch the non-blocking lock and everything else a waiting "
        "one, so whichever fire cron starts first wins and a seal can be skipped silently"
    )
    assert 'lock_wait=(-w "$LOCK_WAIT")' in wrapper, (
        "the loop's fires no longer wait for the lock, so a prematch fire in flight makes a seal "
        "stand down — which is what lost the 16:00 fire on 2026-09-01"
    )


def test_a_loop_fire_that_cannot_take_the_lock_is_a_failed_fire_and_not_a_silence(
    wrapper: str,
) -> None:
    """The half that makes it visible if it ever happens again.

    A `seal` that could not run used to print one line outside any block and exit 0, which reaches
    nobody: cron says nothing about a zero exit, and `epl.bot.fires.parse` drops exactly that line.
    Writing the loop's own `RUN`/`END` format instead makes it a `Fire` with a non-zero exit code,
    which `Fire.failed` reports and `epl.bot.notify` already announces — no new parser, no new
    message.
    """
    assert "===== END  %s  (exit 1) =====" in wrapper, (
        "a loop fire that times out on the lock no longer writes a parseable failed block, so it "
        "would be as silent as the stand-down it replaced"
    )
    assert "notify_about_the_fire\n    exit 1" in wrapper, (
        "the lock-timeout path no longer notifies before exiting, so the one fire nobody can "
        "afford to miss is the one fire the bot is never told about"
    )
