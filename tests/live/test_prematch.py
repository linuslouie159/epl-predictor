"""The fresh look an hour before kickoff, and the one thing it must never touch.

A Pre-Match Reading is taken *after* its round was sealed, from a corpus that has grown since. That
is the whole point of it — a Sunday afternoon Reading has seen Friday's and Saturday's results — and
it is also exactly what makes it dangerous. In the sealed store a later As-Of Instant for the same
Predictor and Fixture means one thing: a superseding revision correcting a bug. If Readings ever
reached that store, or reached the scoreboard, the live track record would get quietly and
unfalsifiably better every week, because a forecast made with two more results in hand is a better
forecast. Nobody would see it happen.

So the tests below are in two halves. The first is ordinary behaviour — which Fixtures are due, what
the window is, that the second fire of a pair says nothing. The second half is the separation, and
it is checked from both ends: the sealed store is byte-identical after a Reading is recorded, and
the Live Season's board is identical with Readings on disk and without them.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

import epl.ledger as ledger
from epl.ledger import live as store
from epl.ledger import readings, schema, scoreboard
from epl.live import prematch, seal, upcoming
from epl.paths import live_dir, prematch_dir
from epl.predictors import Corpus
from the_schedule import CronLine  # tests/ is on sys.path, as tests/bot/log_blocks.py relies on

#: The round every test here works from: two Fixtures on the Saturday, anchored to Friday.
ROUND_FIXTURES: tuple[dict[str, object], ...] = (
    {"date": "2026-08-29", "time": "15:00", "home_club": "arsenal", "away_club": "wolves"},
    {"date": "2026-08-29", "time": "17:30", "home_club": "everton", "away_club": "brighton"},
)

#: Inside the round's sealing window, so the round can be sealed before anything is read.
INSIDE_THE_WINDOW = pd.Timestamp("2026-08-28 14:00:30")

#: An hour before the first Fixture, which is where a Reading is taken.
AN_HOUR_BEFORE = pd.Timestamp("2026-08-29 14:00:00")

#: The hours a Premier League match can kick off in. `deploy/crontab` names 12:30 as the earliest
#: and 20:15 as the latest, and bounds its own fires to 11:00-22:00 on that basis — so this is the
#: span the cadence is required to cover, and a kickoff outside it is out of scope rather than a
#: gap. Whole hours, because the test sweeps every quarter-hour inside each of them.
EARLIEST_KICKOFF = 12
LATEST_KICKOFF = 20


@pytest.fixture
def corpus(make_matches: Callable[..., pd.DataFrame]) -> Corpus:
    return Corpus(
        make_matches({"date": "2026-08-21"}, {"date": "2026-08-22"}, {"date": "2026-08-24"})
    )


@pytest.fixture
def fixtures(make_matches: Callable[..., pd.DataFrame]) -> pd.DataFrame:
    """The round's Fixtures as the rolling file supplies them — no result, because none exists."""
    return make_matches(*ROUND_FIXTURES).drop(columns=["home_goals", "away_goals", "outcome"])


@pytest.fixture
def sealed(
    registry: dict[str, object],
    project_root: Path,
    fixtures: pd.DataFrame,
    corpus: Corpus,
    make_predictor: Callable[..., object],
) -> pd.DataFrame:
    """A round already sealed, which is what a Reading is taken after and compared against."""
    from epl.predictors import register

    register(make_predictor("talker"))
    chosen = upcoming.next_round(fixtures, now=INSIDE_THE_WINDOW)
    seal.run(chosen, corpus, now=INSIDE_THE_WINDOW, commit=False)
    return store.read()


class TestWhichFixturesAreDue:
    def test_a_fixture_an_hour_away_is_due(self, sealed: pd.DataFrame) -> None:
        due = prematch.due(sealed, now=AN_HOUR_BEFORE)

        assert list(due["home_club"]) == ["arsenal"]

    def test_a_fixture_three_hours_away_is_not(self, sealed: pd.DataFrame) -> None:
        """The window means "soon". Widening it would start reading Fixtures whose
        earlier-in-the-round results are not played yet, which is what this exists to use."""
        assert prematch.due(sealed, now=pd.Timestamp("2026-08-29 12:00")).empty

    def test_a_fixture_that_has_kicked_off_is_not(self, sealed: pd.DataFrame) -> None:
        assert prematch.due(sealed, now=pd.Timestamp("2026-08-29 15:30")).empty

    @pytest.mark.parametrize(
        "kickoff_time",
        [
            f"{hour:02d}:{minute:02d}"
            for hour in range(EARLIEST_KICKOFF, LATEST_KICKOFF + 1)
            for minute in (0, 15, 30, 45)
        ],
    )
    def test_every_kickoff_on_a_quarter_hour_is_caught_by_the_crontab_s_own_cadence(
        self, schedule: tuple[CronLine, ...], kickoff_time: str
    ) -> None:
        """The window and the crontab cadence are one decision and have to be checked together.

        The window is 45 to 75 minutes and every Premier League kickoff falls on a quarter-hour, so
        each must be inside at least one fire's window — otherwise a match silently gets no message
        and the schedule looks fine.

        **The fires come from `deploy/crontab` rather than from a literal here**, and that is the
        repair rather than a tidy-up. This test used to build them from `range(0, 24 * 60, 30)`, a
        restatement of a cadence the crontab was free to change underneath it — and the crontab did
        change, moving to :05 and :35 so that `prematch` stops colliding with the two `seal` fires
        (see `tests/deploy/test_the_schedule_does_not_collide.py`). Read from the file, the
        docstring's claim is true of the code; restated, it was true only of the sentence.
        """
        (line,) = [entry for entry in schedule if entry.subcommand == "prematch"]
        kickoff = pd.Timestamp(f"2026-08-29 {kickoff_time}")
        fires = [
            kickoff.normalize() + pd.Timedelta(hours=hour, minutes=fired)
            for hour in sorted(line.hours)
            for fired in sorted(line.minutes)
        ]
        caught = [
            moment
            for moment in fires
            if moment + prematch.WINDOW_SHUTS < kickoff <= moment + prematch.WINDOW_OPENS
        ]

        assert len(caught) >= 1, (
            f"a kickoff at {kickoff:%H:%M} falls in no fire's window, so it would silently get no "
            f"card: the crontab fires at minutes {sorted(line.minutes)} of hours "
            f"{sorted(line.hours)} and the window is "
            f"{prematch.WINDOW_SHUTS}-{prematch.WINDOW_OPENS} before kickoff"
        )

    def test_nothing_is_due_when_nothing_was_sealed(self, project_root: Path) -> None:
        """A round the loop failed to seal gets no pre-match messages either, and that is honest:
        there is no forecast to send and nothing for a Reading to be compared against."""
        assert prematch.due(schema.empty(), now=AN_HOUR_BEFORE).empty


class TestTakingAReading:
    def test_it_is_stamped_now_rather_than_at_the_round_s_instant(
        self, sealed: pd.DataFrame, corpus: Corpus, fixtures: pd.DataFrame
    ) -> None:
        """The entire difference between a Reading and a Sealed Prediction, in one assertion."""
        due = prematch.due(sealed, now=AN_HOUR_BEFORE)
        rows, _ = prematch.readings_for(
            prematch.select(fixtures, due), corpus, now=AN_HOUR_BEFORE
        )

        assert set(rows["as_of_instant"]) == {AN_HOUR_BEFORE}
        assert set(sealed["as_of_instant"]) == {pd.Timestamp("2026-08-28")}

    def test_the_rolling_file_supplies_the_fixture_and_the_sealed_store_says_which(
        self, sealed: pd.DataFrame, fixtures: pd.DataFrame
    ) -> None:
        """Two frames because they answer two different questions. A ledger row has a kickoff and no
        `date` and no odds — it records what a Predictor said, not what it was shown."""
        due = prematch.due(sealed, now=AN_HOUR_BEFORE)
        chosen = prematch.select(fixtures, due)

        assert list(chosen["home_club"]) == ["arsenal"]
        assert "date" in chosen.columns

    def test_a_fixture_that_has_dropped_off_the_rolling_file_comes_back_missing(
        self, sealed: pd.DataFrame, fixtures: pd.DataFrame
    ) -> None:
        """Upstream regenerates that file irregularly (open risk 7) and it rolls forward. A due
        Fixture no longer in it is an ordinary Saturday, not a failure to raise on."""
        due = prematch.due(sealed, now=AN_HOUR_BEFORE)

        assert prematch.select(fixtures.iloc[0:0], due).empty


class TestTheSecondFireSaysNothing:
    def test_a_fixture_already_read_is_no_longer_due(
        self, sealed: pd.DataFrame, corpus: Corpus, fixtures: pd.DataFrame
    ) -> None:
        """The window is wider than the gap between fires, so most Fixtures are seen twice. The
        store itself is what makes the second sighting quiet — not a marker file somebody has to
        remember to keep in step."""
        due = prematch.due(sealed, now=AN_HOUR_BEFORE)
        prematch.run(
            prematch.select(fixtures, due), corpus, now=AN_HOUR_BEFORE, record=True
        )

        assert prematch.due(sealed, now=AN_HOUR_BEFORE + pd.Timedelta(minutes=15)).empty

    def test_recording_the_same_fixture_twice_keeps_the_first_reading(
        self, sealed: pd.DataFrame, corpus: Corpus, fixtures: pd.DataFrame
    ) -> None:
        """The opposite of the sealed store's rule, and right for the opposite reason: there a later
        row is a correction somebody made deliberately, here it is the schedule doing its job twice.
        The first Reading is the one that was actually sent."""
        due = prematch.due(sealed, now=AN_HOUR_BEFORE)
        chosen = prematch.select(fixtures, due)
        prematch.run(chosen, corpus, now=AN_HOUR_BEFORE, record=True)
        prematch.run(chosen, corpus, now=AN_HOUR_BEFORE + pd.Timedelta(minutes=15), record=True)

        held = readings.read_day(pd.Timestamp("2026-08-29"))
        assert set(held["as_of_instant"]) == {AN_HOUR_BEFORE}


class TestAReadingNeverReachesTheSealedStoreOrTheBoard:
    def test_the_sealed_store_is_byte_identical_after_a_reading_is_recorded(
        self, sealed: pd.DataFrame, corpus: Corpus, fixtures: pd.DataFrame
    ) -> None:
        """The bytes are the assertion, the same way `tests/bot` checks the bot cannot write."""
        before = {path: path.read_bytes() for path in live_dir().glob("*.csv")}

        due = prematch.due(sealed, now=AN_HOUR_BEFORE)
        prematch.run(prematch.select(fixtures, due), corpus, now=AN_HOUR_BEFORE, record=True)

        assert {path: path.read_bytes() for path in live_dir().glob("*.csv")} == before

    def test_a_reading_is_written_somewhere_else_entirely(
        self, sealed: pd.DataFrame, corpus: Corpus, fixtures: pd.DataFrame
    ) -> None:
        due = prematch.due(sealed, now=AN_HOUR_BEFORE)
        taken = prematch.run(
            prematch.select(fixtures, due), corpus, now=AN_HOUR_BEFORE, record=True
        )

        assert taken.path is not None
        assert taken.path.parent == prematch_dir()
        assert prematch_dir() != live_dir()

    def test_the_live_board_is_the_same_with_readings_on_disk_and_without(
        self,
        sealed: pd.DataFrame,
        corpus: Corpus,
        fixtures: pd.DataFrame,
        make_matches: Callable[..., pd.DataFrame],
    ) -> None:
        """The load-bearing one. `epl.ledger.stored` concatenates the backtest and sealed stores and
        must not learn about this third one: a Reading on the board would mean the live track record
        improving every Sunday because the model was asked later, which nobody would see happen.
        """
        played = make_matches(
            {"date": "2026-08-29", "home_club": "arsenal", "away_club": "wolves",
             "home_goals": 2, "away_goals": 0, "outcome": "H", "season": 2026, "division": "E0"},
        )
        before = scoreboard.build(ledger.stored(), played, seasons=[2026])

        due = prematch.due(sealed, now=AN_HOUR_BEFORE)
        prematch.run(prematch.select(fixtures, due), corpus, now=AN_HOUR_BEFORE, record=True)
        after = scoreboard.build(ledger.stored(), played, seasons=[2026])

        assert not readings.read().empty  # there really are Readings on disk
        pd.testing.assert_frame_equal(before, after)


class TestADryRunWritesNothing:
    def test_it_returns_the_rows_without_a_path(
        self, sealed: pd.DataFrame, corpus: Corpus, fixtures: pd.DataFrame
    ) -> None:
        due = prematch.due(sealed, now=AN_HOUR_BEFORE)

        taken = prematch.run(
            prematch.select(fixtures, due), corpus, now=AN_HOUR_BEFORE, record=False
        )

        assert taken.path is None
        assert not taken.rows.empty
        assert not prematch_dir().exists() or list(prematch_dir().glob("*.csv")) == []
