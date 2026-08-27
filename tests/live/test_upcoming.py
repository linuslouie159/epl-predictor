"""The rolling fixtures file turned into the one Prediction Round that can be sealed now.

Every test here is about a refusal. The live loop's job is to seal a round before kickoff, and the
ways it can go wrong all end with a Sealed Prediction that is not evidence of anything: a round
sealed under an instant that has not happened, a Fixture stamped with the wrong Season, or half a
round sealed because the rolling file was thin. So the interesting cases are the ones where nothing
is sealed and the reason is said out loud.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from epl.live import upcoming
from epl.windows import LIVE_SEASON

#: A row of the rolling file as `epl.ingest.fixtures.parse_fixtures` hands it over: no Season, no
#: result, and the Market Line already on it.
ROLLING_DEFAULTS: dict[str, object] = {
    "division": "E0",
    "date": "2026-08-29",
    "time": "15:00",
    "home_club": "arsenal",
    "away_club": "chelsea",
    "prematch_odds_home": 2.0,
    "prematch_odds_draw": 3.4,
    "prematch_odds_away": 4.0,
}


@pytest.fixture
def make_rolling() -> Callable[..., pd.DataFrame]:
    """Build a rolling fixtures frame, naming only the fields the test is about."""

    def _make(*rows: dict[str, object]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=list(ROLLING_DEFAULTS))
        frame = pd.DataFrame([{**ROLLING_DEFAULTS, **row} for row in rows])
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        return frame

    return _make


@pytest.fixture
def in_progress(make_matches: Callable[..., pd.DataFrame]) -> pd.DataFrame:
    """A corpus in which the live Season has started and is nowhere near finished."""
    return make_matches(
        {"season": LIVE_SEASON, "date": "2026-08-22", "home_club": "arsenal"},
        {"season": LIVE_SEASON, "date": "2026-08-22", "home_club": "everton"},
    )


class TestWhichFixturesArePredicted:
    def test_the_season_in_progress_is_stamped_on_every_fixture(
        self, make_rolling: Callable[..., pd.DataFrame], in_progress: pd.DataFrame
    ) -> None:
        """The rolling file carries no Season and nothing infers one from the month — 2019/20 ran
        into July and would have been read as 2020/21."""
        stamped = upcoming.to_predict(make_rolling({}), in_progress)

        assert list(stamped["season"]) == [LIVE_SEASON]

    def test_other_tiers_are_dropped(
        self, make_rolling: Callable[..., pd.DataFrame], in_progress: pd.DataFrame
    ) -> None:
        """All four tiers are rated (ADR 0004); only the Premier League is predicted."""
        rolling = make_rolling({"division": "E0"}, {"division": "E2"}, {"division": "E3"})

        assert list(upcoming.to_predict(rolling, in_progress)["division"]) == ["E0"]

    def test_a_season_the_corpus_has_never_seen_is_refused(
        self, make_rolling: Callable[..., pd.DataFrame], make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """Stamping Fixtures with a Season nobody has ingested would seal Predictions that can
        never be joined to a result."""
        corpus = make_matches({"season": LIVE_SEASON - 1})

        with pytest.raises(upcoming.LiveError, match="not under way"):
            upcoming.to_predict(make_rolling({}), corpus)

    def test_a_season_the_corpus_holds_complete_is_refused(
        self, make_rolling: Callable[..., pd.DataFrame], make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """The guard that catches a LIVE_SEASON nobody moved: a full campaign is a Season that is
        over, and a Season that is over has nothing upcoming."""
        corpus = make_matches(
            *[{"season": LIVE_SEASON, "home_club": f"club_{n}"} for n in range(380)]
        )

        with pytest.raises(upcoming.LiveError, match="a Season behind"):
            upcoming.to_predict(make_rolling({}), corpus)

    def test_the_fixtures_carry_what_the_ledger_needs(
        self, make_rolling: Callable[..., pd.DataFrame], in_progress: pd.DataFrame
    ) -> None:
        from epl.ledger.schema import FIXTURE_COLUMNS

        stamped = upcoming.to_predict(make_rolling({}), in_progress)

        assert set(FIXTURE_COLUMNS) <= set(stamped.columns)


class TestTheSealingWindow:
    """A round is sealable from its As-Of Instant until its first kickoff, and at no other time."""

    def test_a_round_inside_its_window_is_sealable(
        self, make_rolling: Callable[..., pd.DataFrame], in_progress: pd.DataFrame
    ) -> None:
        fixtures = upcoming.to_predict(make_rolling({"date": "2026-08-29"}), in_progress)

        table = upcoming.rounds(fixtures, now=pd.Timestamp("2026-08-28 14:00"))

        assert list(table["status"]) == [upcoming.SEALABLE]
        assert list(table["prediction_round"]) == ["2026-08-28"]

    def test_a_round_whose_instant_has_not_arrived_is_not_open(
        self, make_rolling: Callable[..., pd.DataFrame], in_progress: pd.DataFrame
    ) -> None:
        """Sealing on Thursday under Friday's midnight instant claims a moment that has not
        happened, and reads odds that do not exist yet."""
        fixtures = upcoming.to_predict(make_rolling({"date": "2026-08-29"}), in_progress)

        table = upcoming.rounds(fixtures, now=pd.Timestamp("2026-08-27 14:00"))

        assert list(table["status"]) == [upcoming.NOT_OPEN]

    def test_a_round_that_has_started_is_closed_even_for_its_later_fixtures(
        self, make_rolling: Callable[..., pd.DataFrame], in_progress: pd.DataFrame
    ) -> None:
        """ADR 0005 makes the round's *first* kickoff the deadline, not each Fixture's own. A
        Sunday Fixture whose round began on Friday night is no longer sealable."""
        rolling = make_rolling(
            {"date": "2026-08-28", "time": "20:00", "home_club": "everton"},
            {"date": "2026-08-30", "time": "16:30", "home_club": "arsenal"},
        )
        fixtures = upcoming.to_predict(rolling, in_progress)

        table = upcoming.rounds(fixtures, now=pd.Timestamp("2026-08-29 09:00"))

        assert list(table["status"]) == [upcoming.KICKED_OFF]


class TestChoosingTheRound:
    def test_only_the_chosen_round_s_fixtures_come_back(
        self, make_rolling: Callable[..., pd.DataFrame], in_progress: pd.DataFrame
    ) -> None:
        """The rolling file is a week wide and can straddle two rounds; a sealed file holds one."""
        rolling = make_rolling(
            {"date": "2026-08-29", "home_club": "arsenal"},
            {"date": "2026-08-29", "home_club": "everton"},
            {"date": "2026-09-02", "home_club": "chelsea"},
        )
        fixtures = upcoming.to_predict(rolling, in_progress)

        chosen = upcoming.next_round(fixtures, now=pd.Timestamp("2026-08-28 14:00"))

        assert chosen.prediction_round == "2026-08-28"
        assert list(chosen.fixtures["home_club"]) == ["arsenal", "everton"]
        assert chosen.as_of == pd.Timestamp("2026-08-28")
        assert chosen.first_kickoff == pd.Timestamp("2026-08-29 15:00")

    def test_the_earliest_open_round_is_taken(
        self, make_rolling: Callable[..., pd.DataFrame], in_progress: pd.DataFrame
    ) -> None:
        rolling = make_rolling(
            {"date": "2026-09-02", "home_club": "chelsea"},
            {"date": "2026-08-29", "home_club": "arsenal"},
        )
        fixtures = upcoming.to_predict(rolling, in_progress)

        chosen = upcoming.next_round(fixtures, now=pd.Timestamp("2026-08-28 14:00"))

        assert chosen.prediction_round == "2026-08-28"

    def test_nothing_sealable_names_every_round_the_file_held(
        self, make_rolling: Callable[..., pd.DataFrame], in_progress: pd.DataFrame
    ) -> None:
        fixtures = upcoming.to_predict(make_rolling({"date": "2026-08-29"}), in_progress)

        with pytest.raises(upcoming.LiveError, match=r"2026-08-28 \(not open, 1 Fixtures\)"):
            upcoming.next_round(fixtures, now=pd.Timestamp("2026-08-27 14:00"))

    def test_an_empty_file_is_a_different_complaint_from_a_closed_window(
        self, make_rolling: Callable[..., pd.DataFrame], in_progress: pd.DataFrame
    ) -> None:
        """The measured case: a rolling file with no Premier League row in it at all. Reading that
        as "no round is open yet" would hide the thing worth knowing."""
        fixtures = upcoming.to_predict(make_rolling({"division": "E2"}), in_progress)

        with pytest.raises(upcoming.LiveError, match="no row in a tier this project predicts"):
            upcoming.next_round(fixtures, now=pd.Timestamp("2026-08-28 14:00"))

    def test_both_silences_are_the_same_refusal_because_neither_needs_anybody(
        self, make_rolling: Callable[..., pd.DataFrame], in_progress: pd.DataFrame
    ) -> None:
        """The two complaints above say different things and mean the same thing to a schedule:
        there is nothing to seal, and that is not a problem. A stale
        :data:`epl.windows.LIVE_SEASON` or a rolling file that changed shape stays a plain
        :class:`~epl.live.upcoming.LiveError`, because those do need somebody (issue #19)."""
        shut = upcoming.to_predict(make_rolling({"date": "2026-08-29"}), in_progress)
        empty = upcoming.to_predict(make_rolling({"division": "E2"}), in_progress)

        for fixtures in (shut, empty):
            with pytest.raises(upcoming.NothingToSeal):
                upcoming.next_round(fixtures, now=pd.Timestamp("2026-08-27 14:00"))

    def test_a_round_is_described_in_one_line(
        self, make_rolling: Callable[..., pd.DataFrame], in_progress: pd.DataFrame
    ) -> None:
        fixtures = upcoming.to_predict(make_rolling({}), in_progress)

        described = upcoming.next_round(
            fixtures, now=pd.Timestamp("2026-08-28 14:00")
        ).describe()

        assert "2026-08-28" in described
        assert "1 Fixtures" in described
