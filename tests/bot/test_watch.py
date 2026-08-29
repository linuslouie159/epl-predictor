"""The two silences the loop cannot break for itself — issue #20's third and fifth criteria.

Open risk 7 is the dangerous one because it recurs, and because everything about it looks fine: the
Pi is up, the container exits 0, the log has a `===== RUN` block, and the round is gone. It happens
when upstream has not regenerated `fixtures.csv` inside a round's window, so both fires read the
same stale copy and both correctly report having nothing to seal. Nothing in the loop can tell that
apart from a week with no Premier League football, and nothing should try — `NothingToSeal` is a
success on purpose (issue #19).

Open risk 6 is the other half of the same blind spot from the other side. A Pi that is off writes no
block at all, and a process that is not running cannot report that it is not running. What is
buildable on one machine is *retrospective* detection: when the bot comes back, it can say which
anchor days went by without a fire. That is worth having and it is not a dead man's switch, and the
difference is stated in the module rather than glossed.

Both tests are written against the measured shape of the thing. Three fetches across 21-27 Aug 2026
found a file upstream had not rewritten in two and a half days across a matchday; the fourth, on 28
Aug, found one written three hours earlier carrying the whole round. The first case is what this
must catch and the second is what it must not cry wolf over.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from log_blocks import block

from epl.bot import fires, watch
from epl.ingest.fixtures import fixtures_dir
from epl.live import upcoming

#: A rolling file with no row in a tier this project predicts — the measured 21 and 27 Aug shape.
NOTHING_FOR_US = (
    b"Div,Date,Time,HomeTeam,AwayTeam\r\n"
    b"EC,27/08/2026,19:45,Boreham Wood,Boston Utd\r\n"
)

#: The same file after upstream regenerated it: the round, with odds. Measured 28 Aug 2026.
THE_WHOLE_ROUND = (
    b"Div,Date,Time,HomeTeam,AwayTeam,AvgH,AvgD,AvgA\r\n"
    b"E0,28/08/2026,20:00,Crystal Palace,Man City,4.10,3.80,1.85\r\n"
)


@pytest.fixture
def cache(project_root: Path) -> Path:
    """The raw cache of fetched rolling files, where two fires' bytes can be compared."""
    directory = fixtures_dir()
    directory.mkdir(parents=True)
    return directory


def silent_fire(clock: str, cached: str, silence: str = upcoming.NO_FIXTURE_TO_PREDICT) -> str:
    """A `seal --push` fire that read ``cached`` and had nothing to seal."""
    return block(
        clock,
        "seal --push",
        f"rolling file: {cached}",
        f"{silence} at 2026-08-28T21:00:00. ...",
    )


class TestAStaleUpstreamFileThroughBothFires:
    """Open risk 7. Two fires, one window, one unchanged file, and no round."""

    def test_two_silent_fires_that_read_identical_bytes_are_reported(self, cache: Path) -> None:
        (cache / "fixtures_a.csv").write_bytes(NOTHING_FOR_US)
        (cache / "fixtures_b.csv").write_bytes(NOTHING_FOR_US)
        found = fires.parse(
            silent_fire("2026-08-28 11:00:00 -0400", "fixtures_a.csv")
            + silent_fire("2026-08-28 13:30:00 -0400", "fixtures_b.csv")
        )

        concern = watch.stale_upstream(found)

        assert concern is not None
        assert concern.risk == watch.STALE_UPSTREAM
        assert "fixtures_a.csv" in concern.detail and "fixtures_b.csv" in concern.detail

    def test_it_says_the_round_may_be_lost_rather_than_that_it_was(self, cache: Path) -> None:
        """The honest claim is that nobody can tell, which is the finding rather than a hedge.

        An unchanged file means upstream did not regenerate, so whether there was football this
        week is exactly what the loop could not learn. A message that announced a lost round would
        be wrong most weeks of the summer.
        """
        (cache / "fixtures_a.csv").write_bytes(NOTHING_FOR_US)
        (cache / "fixtures_b.csv").write_bytes(NOTHING_FOR_US)
        found = fires.parse(
            silent_fire("2026-08-28 11:00:00 -0400", "fixtures_a.csv")
            + silent_fire("2026-08-28 13:30:00 -0400", "fixtures_b.csv")
        )

        text = watch.stale_upstream(found).message()

        assert "lost" not in text.lower()
        assert "open risk 7" in text

    def test_a_file_upstream_did_regenerate_is_not_a_concern(self, cache: Path) -> None:
        """The quiet weeks this must stay quiet through: a genuinely empty fixture list."""
        (cache / "fixtures_a.csv").write_bytes(NOTHING_FOR_US)
        (cache / "fixtures_b.csv").write_bytes(THE_WHOLE_ROUND)
        found = fires.parse(
            silent_fire("2026-08-25 11:00:00 -0400", "fixtures_a.csv")
            + silent_fire("2026-08-25 13:30:00 -0400", "fixtures_b.csv")
        )

        assert watch.stale_upstream(found) is None

    def test_a_day_that_sealed_something_is_not_a_concern(self, cache: Path) -> None:
        """The 28 Aug shape: the first fire sealed, the retry found it sealed and pushed."""
        (cache / "fixtures_a.csv").write_bytes(THE_WHOLE_ROUND)
        (cache / "fixtures_b.csv").write_bytes(THE_WHOLE_ROUND)
        found = fires.parse(
            block("2026-08-28 11:00:00 -0400", "seal --push", "rolling file: fixtures_a.csv",
                  "sealed 40 Predictions from 4 Predictors -> 2026-08-28.csv (committed abcd1234)")
            + block("2026-08-28 13:30:00 -0400", "seal --push", "rolling file: fixtures_b.csv",
                    "2026-08-28 is already sealed — nothing to do")
        )

        assert watch.stale_upstream(found) is None

    def test_one_fire_alone_is_not_enough_to_say(self, cache: Path) -> None:
        """There is nothing to compare against yet; the retry has not happened."""
        (cache / "fixtures_a.csv").write_bytes(NOTHING_FOR_US)
        found = fires.parse(silent_fire("2026-08-28 11:00:00 -0400", "fixtures_a.csv"))

        assert watch.stale_upstream(found) is None

    def test_fires_from_different_days_are_not_compared(self, cache: Path) -> None:
        """Two Fridays a week apart reading the same bytes is a different claim, and a weaker one.

        The signal is that upstream did not regenerate *inside one round's window*. Across a week it
        would just be an off-season, and reporting it every Tuesday until August is how a channel
        stops being read.
        """
        (cache / "fixtures_a.csv").write_bytes(NOTHING_FOR_US)
        (cache / "fixtures_b.csv").write_bytes(NOTHING_FOR_US)
        found = fires.parse(
            silent_fire("2026-07-21 11:00:00 -0400", "fixtures_a.csv")
            + silent_fire("2026-07-28 13:30:00 -0400", "fixtures_b.csv")
        )

        assert watch.stale_upstream(found) is None

    def test_the_uk_day_is_what_groups_them_and_not_the_pi_s(self, cache: Path) -> None:
        """A round anchors to a UK day, and the Pi is five hours behind it.

        Both of these are 28 Aug in London — 16:00 and 18:30, the two scheduled slots. On the Pi
        they are 11:00 and 13:30, which happens to be the same date; a machine *ahead* of the UK
        splits the same pair across two of its own days, and grouping by the wrong one would
        compare nothing.
        """
        (cache / "fixtures_a.csv").write_bytes(NOTHING_FOR_US)
        (cache / "fixtures_b.csv").write_bytes(NOTHING_FOR_US)
        found = fires.parse(
            silent_fire("2026-08-29 04:00:00 +0800", "fixtures_a.csv")
            + silent_fire("2026-08-29 06:30:00 +0800", "fixtures_b.csv")
        )

        assert watch.stale_upstream(found) is not None

    def test_a_cached_file_that_is_gone_is_not_evidence_either_way(self, cache: Path) -> None:
        """`data/raw/` is a cache and the bot never writes to it. Missing bytes are missing."""
        found = fires.parse(
            silent_fire("2026-08-28 11:00:00 -0400", "fixtures_gone.csv")
            + silent_fire("2026-08-28 13:30:00 -0400", "fixtures_also_gone.csv")
        )

        assert watch.stale_upstream(found) is None


class TestAnAnchorDayWithNoFireAtAll:
    """Open risk 6. The loop cannot report its own absence, so the gap is read afterwards."""

    def test_a_missed_friday_is_noticed(self, project_root: Path) -> None:
        found = fires.parse(block("2026-08-25 11:00:00 -0400", "seal --push", "worked"))

        concern = watch.absent(found, now=pd.Timestamp("2026-09-01 12:00", tz=fires.LOCAL_ZONE))

        assert concern is not None
        assert concern.risk == watch.NO_FIRE_AT_ALL
        assert "2026-08-28" in concern.message()

    def test_the_days_it_expects_are_the_anchor_days_the_project_already_defines(
        self, project_root: Path
    ) -> None:
        """Tuesday and Friday, from `epl.rounds` — the crontab's days, but not read from cron.

        A Prediction Round anchors to a Tuesday or a Friday (`epl.rounds.anchor`), and the schedule
        fires on those days *because* of that. Reading the days back out of `deploy/crontab` would
        make this agree with the schedule rather than with the thing the schedule was derived from,
        and a hand-edited crontab is the case where those two differ.
        """
        found = fires.parse(block("2026-08-25 11:00:00 -0400", "seal --push", "worked"))

        concern = watch.absent(found, now=pd.Timestamp("2026-09-02 12:00", tz=fires.LOCAL_ZONE))

        assert concern is not None
        missed = concern.message()
        assert "2026-08-28" in missed and "2026-09-01" in missed  # a Friday and a Tuesday
        assert "2026-08-29" not in missed  # a Saturday anchors to nothing

    def test_a_day_that_fired_is_not_missed(self, project_root: Path) -> None:
        found = fires.parse(
            block("2026-08-25 11:00:00 -0400", "seal --push", "worked")
            + block("2026-08-28 11:00:00 -0400", "seal --push", "worked")
        )

        assert watch.absent(
            found, now=pd.Timestamp("2026-08-28 23:00", tz=fires.LOCAL_ZONE)
        ) is None

    def test_today_is_not_missed_until_the_retry_slot_has_passed(self, project_root: Path) -> None:
        """16:00 has fired and 18:30 has not. Complaining at 17:00 would be complaining early."""
        found = fires.parse(block("2026-08-25 11:00:00 -0400", "seal --push", "worked"))
        friday_afternoon = pd.Timestamp("2026-08-28 17:00", tz=fires.LOCAL_ZONE)

        assert watch.absent(found, now=friday_afternoon) is None

    def test_a_log_with_no_fire_in_it_at_all_says_nothing(self, project_root: Path) -> None:
        """A fresh clone, or a bot started before the schedule was installed.

        There is no gap to measure without a first fire to measure it from, and a bot that
        announced open risk 6 on its first start would be announcing its own installation.
        """
        assert watch.absent((), now=pd.Timestamp("2026-09-01 20:00", tz=fires.LOCAL_ZONE)) is None

    def test_it_does_not_reach_back_past_its_own_window(self, project_root: Path) -> None:
        """A bot restarted after a month off must not post a month of missed Fridays."""
        found = fires.parse(block("2026-06-02 11:00:00 -0400", "seal --push", "worked"))

        concern = watch.absent(
            found,
            now=pd.Timestamp("2026-09-01 20:00", tz=fires.LOCAL_ZONE),
            lookback_days=14,
        )

        assert concern is not None
        assert "2026-06" not in concern.detail

    def test_a_score_fire_does_not_stand_in_for_a_seal_fire(self, project_root: Path) -> None:
        """A missed morning `score` run costs nothing; a missed sealing run costs a round."""
        found = fires.parse(
            block("2026-08-25 11:00:00 -0400", "seal --push", "worked")
            + block("2026-08-28 01:00:00 -0400", "score ", "nothing has been sealed yet")
        )

        concern = watch.absent(found, now=pd.Timestamp("2026-08-29 12:00", tz=fires.LOCAL_ZONE))

        assert concern is not None
        assert "2026-08-28" in concern.message()


class TestBothTogether:
    def test_concerns_reports_each_risk_at_most_once(self, cache: Path) -> None:
        (cache / "fixtures_a.csv").write_bytes(NOTHING_FOR_US)
        (cache / "fixtures_b.csv").write_bytes(NOTHING_FOR_US)
        found = fires.parse(
            silent_fire("2026-08-25 11:00:00 -0400", "fixtures_a.csv")
            + silent_fire("2026-08-25 13:30:00 -0400", "fixtures_b.csv")
        )

        raised = watch.concerns(found, now=pd.Timestamp("2026-08-29 12:00", tz=fires.LOCAL_ZONE))

        assert [concern.risk for concern in raised] == [watch.NO_FIRE_AT_ALL, watch.STALE_UPSTREAM]

    def test_a_healthy_schedule_raises_nothing(self, cache: Path) -> None:
        (cache / "fixtures_a.csv").write_bytes(THE_WHOLE_ROUND)
        found = fires.parse(
            block("2026-08-28 11:00:00 -0400", "seal --push", "rolling file: fixtures_a.csv",
                  "sealed 40 Predictions from 4 Predictors -> 2026-08-28.csv (committed abcd1234)")
        )

        assert watch.concerns(
            found, now=pd.Timestamp("2026-08-28 22:00", tz=fires.LOCAL_ZONE)
        ) == ()

    def test_every_concern_names_the_risk_it_is_about(self, cache: Path) -> None:
        """A message that said only "something looks wrong" would send its reader to the docs."""
        (cache / "fixtures_a.csv").write_bytes(NOTHING_FOR_US)
        (cache / "fixtures_b.csv").write_bytes(NOTHING_FOR_US)
        found = fires.parse(
            silent_fire("2026-08-25 11:00:00 -0400", "fixtures_a.csv")
            + silent_fire("2026-08-25 13:30:00 -0400", "fixtures_b.csv")
        )

        for concern in watch.concerns(
            found, now=pd.Timestamp("2026-08-29 12:00", tz=fires.LOCAL_ZONE)
        ):
            assert f"open risk {concern.risk}" in concern.message()
            assert concern.headline
