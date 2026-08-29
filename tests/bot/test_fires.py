"""What one scheduled fire looks like from the outside — issue #20.

`deploy/logs/live_loop.log` is the only record that the schedule ran at all. The sealed store
records what a fire *wrote*, and most fires write nothing: two of the three crontab lines are
`seal --push`, and the second of them is a retry that is designed to find the round already sealed.
So a reader asking "did the loop run on Friday?" has the log and nothing else, which is why open
risks 6 and 7 both look like a quiet week from here.

The format is `deploy/run_live.sh`'s and is deliberately the other project's on this Pi. It is
written by `printf` in a shell script, so the awkward parts are shell-shaped rather than
Python-shaped: two spaces after `END`, a trailing space when a subcommand takes no arguments, and a
timestamp in **the Pi's own zone** rather than in the UK's — the zone `deploy/crontab`'s times were
converted into, and the one every round window has to be judged against.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from log_blocks import THE_FIRE_THAT_PROVED_THE_SCHEDULE, block

from epl.bot import fires
from epl.live import upcoming


class TestOneBlock:
    def test_it_reads_the_fire_that_proved_the_schedule(self) -> None:
        (fire,) = fires.parse(THE_FIRE_THAT_PROVED_THE_SCHEDULE)

        assert fire.command == "seal --push"
        assert fire.subcommand == "seal"
        assert fire.exit_code == 0
        assert fire.started == pd.Timestamp("2026-08-28 13:30:01-04:00")
        assert fire.finished == pd.Timestamp("2026-08-28 13:30:08-04:00")

    def test_the_stamp_keeps_the_offset_the_pi_wrote(self) -> None:
        """The Pi is not in the UK and the rounds it seals are. Both zones are real here.

        `deploy/crontab`'s times are converted into the Pi's zone because its cron ignores
        `CRON_TZ`, so a fire that reads `13:30` locally is the 18:30 UK retry. Dropping the offset
        would put every fire five hours from where it happened, which is most of the distance
        between one round's window and the next.
        """
        (fire,) = fires.parse(THE_FIRE_THAT_PROVED_THE_SCHEDULE)

        assert fire.started.tz_convert(fires.LOCAL_ZONE).hour == 18

    def test_a_subcommand_with_no_arguments_does_not_keep_the_shell_s_trailing_space(self) -> None:
        """`printf '(%s %s)' "$subcommand" "$*"` writes `(score )` when there is no `$*`."""
        (fire,) = fires.parse(block("2026-08-28 06:00:00 -0400", "score "))

        assert fire.command == "score"
        assert fire.subcommand == "score"

    def test_the_body_is_kept_and_the_markers_are_not(self) -> None:
        text = block("2026-08-28 12:00:00 -0400", "seal --push", "rolling file: a.csv", "worked")
        (fire,) = fires.parse(text)

        assert fire.lines == ("rolling file: a.csv", "worked")

    def test_it_names_the_rolling_file_the_fire_read(self) -> None:
        """The handle on what this fire actually saw — open risk 7 is a claim about those bytes."""
        text = block(
            "2026-08-28 12:00:00 -0400", "seal --push", "rolling file: fixtures_20260828T1200Z.csv"
        )
        (fire,) = fires.parse(text)

        assert fire.rolling_file == "fixtures_20260828T1200Z.csv"

    def test_a_fire_that_read_no_rolling_file_says_so_rather_than_guessing(self) -> None:
        (fire,) = fires.parse(block("2026-08-28 06:00:00 -0400", "score "))

        assert fire.rolling_file is None


class TestAFireThatNeverFinished:
    """A block with no `END` line. The container was killed, or the Pi went down mid-run.

    Worth telling apart from a failure: an exit code of 1 is the loop saying something, and a
    missing exit code is the loop having been stopped before it could.
    """

    def test_it_is_read_rather_than_dropped(self) -> None:
        text = block("2026-08-28 16:00:00 -0400", "seal --push", "rolling file: a.csv",
                     exit_code=None)
        (fire,) = fires.parse(text)

        assert fire.exit_code is None
        assert fire.finished is None
        assert fire.verdict == fires.UNFINISHED

    def test_it_does_not_swallow_the_block_after_it(self) -> None:
        text = (
            block("2026-08-28 16:00:00 -0400", "seal --push", "half a run", exit_code=None)
            + block("2026-08-28 18:30:00 -0400", "seal --push", "all of one")
        )
        first, second = fires.parse(text)

        assert first.lines == ("half a run",)
        assert second.lines == ("all of one",)


class TestTheVerdict:
    """Which of the loop's outcomes this was — and, above all, which silence.

    The exit code answers three of these and cannot answer the last two. `NothingToSeal` is exit 0
    on purpose (issue #19), so "the rolling file held no Premier League row" and "no round is inside
    its window" arrive here spelt identically to a round that sealed cleanly.
    """

    def test_a_clean_run_worked(self) -> None:
        (fire,) = fires.parse(THE_FIRE_THAT_PROVED_THE_SCHEDULE)

        assert fire.verdict == fires.WORKED

    def test_a_non_zero_exit_failed(self) -> None:
        text = block("2026-08-28 16:00:00 -0400", "seal --push", "NOT PUSHED", exit_code=1)
        (fire,) = fires.parse(text)

        assert fire.verdict == fires.FAILED
        assert fire.failed

    def test_an_empty_rolling_file_is_told_apart_from_a_clock_outside_every_window(self) -> None:
        """The two silences, quoted from `epl.live.upcoming` rather than retyped here.

        Retyping them is how this check would go on passing while the loop's message moved.
        """
        empty = block(
            "2026-08-25 16:00:00 -0400",
            "seal --push",
            f"{upcoming.NO_FIXTURE_TO_PREDICT} at 2026-08-25T21:00:00. The rolling file held no...",
        )
        shut = block(
            "2026-08-26 16:00:00 -0400",
            "seal --push",
            f"{upcoming.NONE_INSIDE_A_WINDOW} at 2026-08-26T21:00:00; the rolling file holds...",
        )

        assert fires.parse(empty)[0].verdict == fires.NOTHING_IN_FILE
        assert fires.parse(shut)[0].verdict == fires.OUTSIDE_EVERY_WINDOW

    def test_a_failure_outranks_a_silence(self) -> None:
        """A run that printed one of the two silences and then exited 1 did not have a quiet week.

        It cannot happen through `_seal` today — `NothingToSeal` returns 0 — but it can through
        `--push`, whose failure is stamped on the way out of a run that had nothing else to say.
        """
        text = block(
            "2026-08-28 16:00:00 -0400",
            "seal --push",
            f"{upcoming.NO_FIXTURE_TO_PREDICT} at 2026-08-28T21:00:00.",
            "WARNING: the round is committed here and NOT PUSHED",
            exit_code=1,
        )

        assert fires.parse(text)[0].verdict == fires.FAILED


class TestAWholeLog:
    def test_blocks_come_back_in_the_order_they_were_written(self) -> None:
        text = "".join(
            block(f"2026-08-2{day} 16:00:00 -0400", "seal --push") for day in (1, 4, 8)
        )
        parsed = fires.parse(text)

        assert [fire.started.day for fire in parsed] == [21, 24, 28]

    def test_lines_outside_any_block_are_ignored(self) -> None:
        """`run_live.sh` writes two of these outside a block: a missing `flock`, and a stand-down.

        Neither is a fire. The stand-down in particular is the *absence* of one — a second run
        finding the lock held — and counting it as a run would report a loop that did nothing as
        having run twice.
        """
        text = (
            "2026-08-28 16:00:02 -0400  live loop already running (seal); standing down\n"
            + THE_FIRE_THAT_PROVED_THE_SCHEDULE
        )

        assert len(fires.parse(text)) == 1

    def test_an_empty_log_is_no_fires_rather_than_an_error(self) -> None:
        assert fires.parse("") == ()

    def test_the_latest_fire_of_a_subcommand_can_be_singled_out(self) -> None:
        """`score` and `seal` fire on the same days, and their health is asked about separately."""
        text = (
            block("2026-08-28 06:00:00 -0400", "score ")
            + block("2026-08-28 12:00:00 -0400", "seal --push")
            + block("2026-08-28 13:30:00 -0400", "seal --push")
        )
        parsed = fires.parse(text)

        assert fires.latest(parsed).started.hour == 13
        assert fires.latest(parsed, subcommand="score").started.hour == 6
        assert fires.latest(parsed, subcommand="upcoming") is None
        assert fires.latest(()) is None


class TestReadingItOffDisk:
    def test_a_log_that_has_never_been_written_is_no_fires(self) -> None:
        """The bot starts before the first fire on a fresh clone, and that is not a failure."""
        assert fires.read(Path("nowhere") / "live_loop.log") == ()

    def test_it_reads_the_log_the_schedule_writes(self, project_root: Path) -> None:
        log = project_root / "deploy" / "logs" / "live_loop.log"
        log.parent.mkdir(parents=True)
        log.write_text(THE_FIRE_THAT_PROVED_THE_SCHEDULE, encoding="utf-8")

        assert fires.log_path() == log
        assert len(fires.read()) == 1

    def test_undecodable_bytes_do_not_stop_it_reading(self, project_root: Path) -> None:
        """A log is machine-local and unversioned, and half a line of it is not worth a crash.

        The bot's whole job here is to speak up when the loop is quiet; a notifier that fell over
        on a truncated write would be silent in exactly the way it exists to prevent.
        """
        log = project_root / "deploy" / "logs" / "live_loop.log"
        log.parent.mkdir(parents=True)
        log.write_bytes(THE_FIRE_THAT_PROVED_THE_SCHEDULE.encode("utf-8") + b"\xff\xfe")

        assert len(fires.read()) == 1


class TestTheParserWouldNoticeItStoppedWorking:
    """Everything above is written against a sample. This is written against the real thing."""

    def test_the_loop_still_prints_the_line_the_rolling_file_is_read_from(self) -> None:
        from epl.live import __main__ as cli

        assert cli.ROLLING_FILE_PREFIX == "rolling file: "
        (fire,) = fires.parse(THE_FIRE_THAT_PROVED_THE_SCHEDULE)
        assert fire.rolling_file is not None

    @pytest.mark.parametrize("silence", ["NO_FIXTURE_TO_PREDICT", "NONE_INSIDE_A_WINDOW"])
    def test_each_silence_the_verdict_depends_on_is_still_a_named_constant(
        self, silence: str
    ) -> None:
        assert getattr(upcoming, silence) in upcoming.__all__ or hasattr(upcoming, silence)
        assert isinstance(getattr(upcoming, silence), str)
