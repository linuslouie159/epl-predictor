"""Prediction Rounds derived from kickoff dates.

The source data has no matchweek column - 2024/25's 380 Fixtures spanned 109 distinct dates - so a
Fixture's As-Of Instant is derived from its kickoff date alone (ADR 0002). These tests pin the
anchor rule to the seven weekdays and to the awkward Fixtures that motivated it.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from epl import rounds


class TestAnchor:
    """The rule from docs/DECISIONS.md, weekday by weekday.

    Anchor dates below are checked by hand: 2024-08-16 is a Friday, 2024-08-20 a Tuesday.
    """

    def test_a_friday_fixture_anchors_to_that_friday(self) -> None:
        assert rounds.anchor(date(2024, 8, 16)) == date(2024, 8, 16)

    def test_a_saturday_fixture_anchors_back_to_that_friday(self) -> None:
        assert rounds.anchor(date(2024, 8, 17)) == date(2024, 8, 16)

    def test_a_sunday_fixture_anchors_back_to_that_friday(self) -> None:
        assert rounds.anchor(date(2024, 8, 18)) == date(2024, 8, 16)

    def test_a_monday_fixture_anchors_back_to_the_previous_friday(self) -> None:
        """Monday night football belongs to the weekend round, not to the Tuesday after it."""
        assert rounds.anchor(date(2024, 8, 19)) == date(2024, 8, 16)

    def test_a_tuesday_fixture_anchors_to_that_tuesday(self) -> None:
        assert rounds.anchor(date(2024, 8, 20)) == date(2024, 8, 20)

    def test_a_wednesday_fixture_anchors_back_to_that_tuesday(self) -> None:
        assert rounds.anchor(date(2024, 8, 21)) == date(2024, 8, 20)

    def test_a_thursday_fixture_anchors_back_to_that_tuesday(self) -> None:
        assert rounds.anchor(date(2024, 8, 22)) == date(2024, 8, 20)

    @pytest.mark.parametrize(
        "kickoff",
        [date(2024, 8, d) for d in range(16, 23)],
        ids=["fri", "sat", "sun", "mon", "tue", "wed", "thu"],
    )
    def test_every_anchor_is_a_tuesday_or_a_friday(self, kickoff: date) -> None:
        assert rounds.anchor(kickoff).weekday() in (rounds.TUESDAY, rounds.FRIDAY)

    @pytest.mark.parametrize(
        "kickoff",
        [date(2024, 8, d) for d in range(16, 23)],
        ids=["fri", "sat", "sun", "mon", "tue", "wed", "thu"],
    )
    def test_no_anchor_is_later_than_its_kickoff(self, kickoff: date) -> None:
        """An anchor after kickoff would be a Prediction made after the Fixture started."""
        assert rounds.anchor(kickoff) <= kickoff

    @pytest.mark.parametrize(
        "kickoff",
        [date(2024, 8, d) for d in range(16, 23)],
        ids=["fri", "sat", "sun", "mon", "tue", "wed", "thu"],
    )
    def test_no_anchor_is_more_than_three_days_before_kickoff(self, kickoff: date) -> None:
        """Tue/Fri anchoring means the widest gap is Monday back to the previous Friday."""
        assert (kickoff - rounds.anchor(kickoff)).days <= 3


class TestAnchorAcrossBoundaries:
    def test_anchors_across_a_month_boundary(self) -> None:
        """1 September 2024 was a Sunday; the Friday before it was 30 August."""
        assert rounds.anchor(date(2024, 9, 1)) == date(2024, 8, 30)

    def test_anchors_across_a_year_boundary(self) -> None:
        """1 January 2024 was a Monday; the Friday before it was 29 December 2023."""
        assert rounds.anchor(date(2024, 1, 1)) == date(2023, 12, 29)

    def test_anchors_across_a_leap_day(self) -> None:
        """1 March 2024 was a Friday, so the leap day sits inside the preceding round."""
        assert rounds.anchor(date(2024, 2, 29)) == date(2024, 2, 27)


class TestAsOfInstant:
    """The anchor is a date; the As-Of Instant is the moment on it a Prediction may be made."""

    def test_is_midnight_at_the_start_of_the_anchor_day(self) -> None:
        assert rounds.as_of_instant(date(2024, 8, 17)) == datetime(2024, 8, 16, 0, 0)

    def test_falls_strictly_before_the_earliest_kickoff_ever_recorded_on_an_anchor_day(
        self,
    ) -> None:
        """12:30 is the earliest kickoff in the corpus on any Tuesday or Friday.

        Kickoff times are absent before 2019/20, so an instant that depended on time-of-day
        precision would be unverifiable across two thirds of the corpus.
        """
        assert rounds.as_of_instant(date(2024, 8, 20)) < datetime(2024, 8, 20, 12, 30)

    def test_is_never_later_than_the_kickoff_day_it_was_derived_from(self) -> None:
        assert rounds.as_of_instant(date(2024, 8, 19)).date() <= date(2024, 8, 19)

    def test_accepts_a_datetime_and_ignores_its_time_of_day(self) -> None:
        """Callers hold kickoffs as timestamps; the anchor depends on the date alone."""
        assert rounds.as_of_instant(datetime(2024, 8, 17, 15, 0)) == datetime(2024, 8, 16, 0, 0)


class TestRoundId:
    def test_is_the_anchor_date_in_iso_form(self) -> None:
        """Sortable, self-describing, and safe as the ledger's per-round filename (ADR 0005)."""
        assert rounds.round_id(date(2024, 8, 17)) == "2024-08-16"

    def test_every_fixture_in_one_round_shares_it(self) -> None:
        weekend = [date(2024, 8, 16), date(2024, 8, 17), date(2024, 8, 18), date(2024, 8, 19)]
        assert {rounds.round_id(kickoff) for kickoff in weekend} == {"2024-08-16"}

    def test_a_midweek_fixture_gets_a_different_id_from_the_weekend_around_it(self) -> None:
        assert rounds.round_id(date(2024, 8, 21)) != rounds.round_id(date(2024, 8, 17))

    def test_an_as_of_instant_maps_back_to_its_own_round(self) -> None:
        """The id of a round derived from a round's own instant is that round."""
        instant = rounds.as_of_instant(date(2024, 8, 17))
        assert rounds.round_id(instant) == "2024-08-16"


def _matches(*rows: tuple[str, str | None]) -> pd.DataFrame:
    """A minimal matches frame: kickoff date, and the time when the era recorded one."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime([kickoff for kickoff, _ in rows]),
            "time": pd.Series([time for _, time in rows], dtype="string"),
            "home_club": [f"club_{i}" for i in range(len(rows))],
        }
    )


class TestKickoffInstants:
    def test_combines_the_date_with_the_recorded_time(self) -> None:
        instants = rounds.kickoff_instants(_matches(("2024-08-17", "15:00")))
        assert instants.iloc[0] == pd.Timestamp("2024-08-17 15:00")

    def test_falls_back_to_the_start_of_the_day_when_no_time_was_recorded(self) -> None:
        """Football-Data carries no kickoff time before 2019/20."""
        instants = rounds.kickoff_instants(_matches(("2004-08-17", None)))
        assert instants.iloc[0] == pd.Timestamp("2004-08-17 00:00")

    def test_works_on_a_frame_with_no_time_column_at_all(self) -> None:
        frame = pd.DataFrame({"date": pd.to_datetime(["2004-08-17"])})
        assert rounds.kickoff_instants(frame).iloc[0] == pd.Timestamp("2004-08-17 00:00")


class TestAssignRounds:
    def test_adds_an_as_of_instant_and_a_prediction_round_to_every_fixture(self) -> None:
        assigned = rounds.assign_rounds(_matches(("2024-08-17", "15:00")))
        assert assigned.loc[0, "as_of_instant"] == pd.Timestamp("2024-08-16")
        assert assigned.loc[0, "prediction_round"] == "2024-08-16"

    def test_groups_a_whole_weekend_and_its_monday_into_one_round(self) -> None:
        assigned = rounds.assign_rounds(
            _matches(
                ("2024-08-16", "20:00"),
                ("2024-08-17", "15:00"),
                ("2024-08-18", "14:00"),
                ("2024-08-19", "20:00"),
            )
        )
        assert assigned["prediction_round"].nunique() == 1

    def test_splits_a_midweek_slate_off_from_the_weekend_around_it(self) -> None:
        assigned = rounds.assign_rounds(
            _matches(("2024-08-17", "15:00"), ("2024-08-21", "19:45"))
        )
        assert list(assigned["prediction_round"]) == ["2024-08-16", "2024-08-20"]

    def test_leaves_the_input_frame_untouched(self) -> None:
        frame = _matches(("2024-08-17", "15:00"))
        rounds.assign_rounds(frame)
        assert "as_of_instant" not in frame.columns

    def test_preserves_row_order_and_every_original_column(self) -> None:
        frame = _matches(("2024-08-21", "19:45"), ("2024-08-17", "15:00"))
        assigned = rounds.assign_rounds(frame)
        assert list(assigned["home_club"]) == list(frame["home_club"])
        assert set(frame.columns) <= set(assigned.columns)

    def test_rejects_a_frame_with_no_kickoff_date(self) -> None:
        with pytest.raises(rounds.RoundsError, match="date"):
            rounds.assign_rounds(pd.DataFrame({"home_club": ["arsenal"]}))

    def test_handles_an_empty_frame(self) -> None:
        assigned = rounds.assign_rounds(_matches())
        assert assigned.empty
        assert "prediction_round" in assigned.columns


class TestEveryFixtureKicksOffAfterItsAsOfInstant:
    """The project's central claim, enforced where rounds are built rather than checked later."""

    def test_a_recorded_kickoff_on_the_anchor_day_is_still_after_the_instant(self) -> None:
        assigned = rounds.assign_rounds(_matches(("2024-08-16", "12:30")))
        assert assigned.loc[0, "as_of_instant"] < pd.Timestamp("2024-08-16 12:30")

    def test_a_fixture_kicking_off_at_its_own_as_of_instant_is_rejected(self) -> None:
        with pytest.raises(rounds.RoundsError, match="kick off at or before"):
            rounds.assign_rounds(_matches(("2024-08-16", "00:00")))

    def test_every_kickoff_in_a_mixed_slate_is_strictly_after_its_instant(self) -> None:
        assigned = rounds.assign_rounds(
            _matches(
                ("2024-08-16", "20:00"),
                ("2024-08-18", "14:00"),
                ("2024-08-20", "19:45"),
                ("2024-08-22", None),
            )
        )
        assert (rounds.kickoff_instants(assigned) > assigned["as_of_instant"]).all()


class TestPredictionRounds:
    def test_returns_one_row_per_round(self) -> None:
        summary = rounds.prediction_rounds(
            _matches(("2024-08-17", "15:00"), ("2024-08-18", "14:00"), ("2024-08-21", "19:45"))
        )
        assert list(summary["prediction_round"]) == ["2024-08-16", "2024-08-20"]

    def test_counts_the_fixtures_in_each_round(self) -> None:
        summary = rounds.prediction_rounds(
            _matches(("2024-08-17", "15:00"), ("2024-08-18", "14:00"), ("2024-08-21", "19:45"))
        )
        assert list(summary["fixtures"]) == [2, 1]

    def test_carries_the_as_of_instant_the_round_is_predicted_from(self) -> None:
        summary = rounds.prediction_rounds(_matches(("2024-08-17", "15:00")))
        assert summary.loc[0, "as_of_instant"] == pd.Timestamp("2024-08-16")

    def test_records_the_first_kickoff_so_a_sealed_round_has_a_deadline(self) -> None:
        """ADR 0005: a Sealed Prediction must be written before its round's first kickoff."""
        summary = rounds.prediction_rounds(
            _matches(("2024-08-17", "15:00"), ("2024-08-16", "20:00"))
        )
        assert summary.loc[0, "first_kickoff"] == pd.Timestamp("2024-08-16 20:00")

    def test_orders_rounds_by_when_they_were_predicted(self) -> None:
        summary = rounds.prediction_rounds(
            _matches(("2024-08-21", "19:45"), ("2024-08-17", "15:00"))
        )
        assert list(summary["as_of_instant"]) == [
            pd.Timestamp("2024-08-16"),
            pd.Timestamp("2024-08-20"),
        ]

    def test_carries_the_season_when_the_frame_names_one(self) -> None:
        frame = _matches(("2024-08-17", "15:00"))
        frame["season"] = 2024
        assert list(rounds.prediction_rounds(frame)["season"]) == [2024]

    def test_works_without_a_season_column(self) -> None:
        summary = rounds.prediction_rounds(_matches(("2024-08-17", "15:00")))
        assert "season" not in summary.columns

    def test_every_round_holds_at_least_one_fixture(self) -> None:
        summary = rounds.prediction_rounds(
            _matches(("2024-08-17", "15:00"), ("2024-08-21", "19:45"))
        )
        assert (summary["fixtures"] >= 1).all()

    def test_every_round_starts_after_the_instant_it_was_predicted_from(self) -> None:
        summary = rounds.prediction_rounds(
            _matches(("2024-08-16", "20:00"), ("2024-08-20", "19:45"))
        )
        assert (summary["first_kickoff"] > summary["as_of_instant"]).all()

    def test_an_empty_frame_returns_the_same_columns_a_populated_one_would(self) -> None:
        """Downstream code should not have to branch on whether any Fixture was found."""
        empty = rounds.prediction_rounds(_matches())
        populated = rounds.prediction_rounds(_matches(("2024-08-17", "15:00")))
        assert empty.empty
        assert list(empty.columns) == list(populated.columns)

    def test_an_empty_frame_that_names_a_season_keeps_the_season_column(self) -> None:
        frame = _matches()
        frame["season"] = pd.Series(dtype="int64")
        assert list(rounds.prediction_rounds(frame).columns) == list(rounds.ROUND_COLUMNS)


class TestRoundsIgnoreAnyUpstreamGameweek:
    """ADR 0002: rounds come from kickoff dates and from nothing else."""

    def test_a_contradictory_gameweek_column_changes_nothing(self) -> None:
        plain = _matches(("2024-08-17", "15:00"), ("2024-08-18", "14:00"))
        labelled = plain.assign(gameweek=[7, 41])
        assert list(rounds.assign_rounds(labelled)["prediction_round"]) == list(
            rounds.assign_rounds(plain)["prediction_round"]
        )

    def test_a_frame_carrying_only_kickoff_dates_is_enough(self) -> None:
        frame = pd.DataFrame({"date": pd.to_datetime(["2024-08-17", "2024-08-21"])})
        assert list(rounds.assign_rounds(frame)["prediction_round"]) == [
            "2024-08-16",
            "2024-08-20",
        ]


@pytest.mark.cache
class TestTheWholePremierLeagueCorpus:
    """Ticket 5 over every E0 Fixture of the two Windows — 2000/01 to 2025/26.

    Deliberately not the whole corpus, which from stage 13 also holds the Season being played
    (ADR 0010). That Season grows every Saturday, so a count including it would be a test that
    failed weekly and told nobody anything; it is checked for shape in
    :class:`TestTheSeasonInProgress` instead.

    Needs a populated ``data/raw/``, which is gitignored, so these skip when it is absent:

        python -c "from epl.ingest import fetch_all; fetch_all()"

    The counts below are re-derived here rather than trusted, in the same spirit as
    ``tests/ingest/test_raw_cache_integrity.py``. Issue #5 and an earlier draft of
    docs/DECISIONS.md recorded 1,332 rounds; the anchor rule as specified yields 1,189 over this
    corpus, and the rule is the artifact the spec states as executable code, so the count was
    corrected to match it rather than the other way round.
    """

    def test_every_premier_league_fixture_carries_a_round(self, e0: pd.DataFrame) -> None:
        assigned = rounds.assign_rounds(e0)
        assert assigned["as_of_instant"].notna().all()
        assert assigned["prediction_round"].notna().all()

    def test_the_corpus_is_the_9880_fixtures_the_windows_span(self, e0: pd.DataFrame) -> None:
        assert len(e0) == 9880

    def test_there_are_1189_prediction_rounds(self, summary: pd.DataFrame) -> None:
        assert len(summary) == 1189

    def test_a_season_averages_45_7_rounds(self, summary: pd.DataFrame) -> None:
        assert round(summary.groupby("season").size().mean(), 1) == 45.7

    def test_a_round_averages_8_31_fixtures(self, summary: pd.DataFrame) -> None:
        assert round(summary["fixtures"].mean(), 2) == 8.31

    def test_no_round_is_empty(self, summary: pd.DataFrame) -> None:
        assert summary["fixtures"].min() >= 1

    def test_no_round_spans_two_seasons(self, summary: pd.DataFrame) -> None:
        assert summary["prediction_round"].nunique() == len(summary)

    def test_every_as_of_instant_is_a_tuesday_or_a_friday(self, summary: pd.DataFrame) -> None:
        weekdays = set(summary["as_of_instant"].dt.weekday.unique())
        assert weekdays <= {rounds.TUESDAY, rounds.FRIDAY}

    def test_no_round_starts_before_the_instant_it_was_predicted_from(
        self, summary: pd.DataFrame
    ) -> None:
        """``>=`` rather than ``>`` on purpose, and the two tests below are why.

        A round whose first Fixture has no recorded kickoff time has that Fixture placed at the
        start of its day, which for a Fixture on its own anchor day is the As-Of Instant itself.
        Strictness for those is established by the pair of tests below, not by this one.
        """
        assert (summary["first_kickoff"] >= summary["as_of_instant"]).all()

    def test_only_437_fixtures_are_known_no_more_precisely_than_to_the_day(
        self, e0: pd.DataFrame
    ) -> None:
        """Pinned exactly so the carve-out below cannot quietly grow.

        These are the Tuesday and Friday kickoffs from 2000/01 to 2018/19, the Seasons before
        Football-Data recorded a kickoff time. Every other Fixture either carries a time or kicks
        off on a later day than the one it was predicted on.
        """
        assigned = rounds.assign_rounds(e0)
        to_the_day = rounds.kickoff_instants(assigned) == assigned["as_of_instant"]
        assert to_the_day.sum() == 437
        assert assigned.loc[to_the_day, "time"].isna().all()
        assert assigned.loc[to_the_day, "season"].max() == 2018

    def test_every_fixture_outside_that_437_kicks_off_strictly_after_its_as_of_instant(
        self, e0: pd.DataFrame
    ) -> None:
        """The strict form of the claim, over the other 9,443 Fixtures."""
        assigned = rounds.assign_rounds(e0)
        kickoffs = rounds.kickoff_instants(assigned)
        strict = kickoffs > assigned["as_of_instant"]
        assert strict.sum() == 9443
        assert (kickoffs[strict] > assigned.loc[strict, "as_of_instant"]).all()

    def test_every_timed_fixture_kicks_off_strictly_after_its_as_of_instant(
        self, e0: pd.DataFrame
    ) -> None:
        """The exact form of the check, over the 2,660 Fixtures whose kickoff time is recorded."""
        assigned = rounds.assign_rounds(e0)
        timed = assigned["time"].notna()
        assert timed.sum() == 2660
        kickoffs = rounds.kickoff_instants(assigned)
        assert (kickoffs[timed] > assigned.loc[timed, "as_of_instant"]).all()

    def test_no_untimed_fixture_kicks_off_before_the_day_it_was_predicted_on(
        self, e0: pd.DataFrame
    ) -> None:
        """Football-Data records no kickoff time before 2019/20.

        For those Fixtures the strongest verifiable claim is that the As-Of Instant lands at or
        before the start of the kickoff day. Since no Fixture in the corpus kicks off at midnight,
        that still gives strictness.
        """
        assigned = rounds.assign_rounds(e0)
        assert (pd.to_datetime(assigned["date"]) >= assigned["as_of_instant"]).all()
        assert not (assigned["time"] == "00:00").any()


@pytest.mark.cache
class TestTheSeasonInProgress:
    """The anchor rule over the Season being played, which is what the live loop seals from.

    Checked for shape rather than for size — the Season grows every Saturday, and a Fixture count
    here would be a weekly failure. What must hold is that a Season in progress is not a special
    case: the same rule anchors it, and the same strictness holds.
    """

    def test_every_fixture_played_so_far_carries_a_round(self, live_e0: pd.DataFrame) -> None:
        assigned = rounds.assign_rounds(live_e0)

        assert len(assigned) > 0
        assert assigned["prediction_round"].notna().all()

    def test_it_is_a_partial_season(self, live_e0: pd.DataFrame) -> None:
        assert 0 < len(live_e0) < 380

    def test_every_fixture_kicks_off_strictly_after_its_own_as_of_instant(
        self, live_e0: pd.DataFrame
    ) -> None:
        """Football-Data has recorded a kickoff time since 2019/20, so the exact form of the check
        applies to every Fixture of a live Season — no 437-style carve-out."""
        assigned = rounds.assign_rounds(live_e0)

        assert assigned["time"].notna().all()
        assert (rounds.kickoff_instants(assigned) > assigned["as_of_instant"]).all()


@pytest.fixture(scope="module")
def e0() -> pd.DataFrame:
    """Every Premier League Fixture across the two Windows, from the raw cache.

    The two Windows, not the whole corpus: the Season in progress is in neither (ADR 0010) and grows
    every Saturday, so the counts in :class:`TestTheWholePremierLeagueCorpus` are over the closed
    Seasons and stay put. :func:`live_e0` is the Season in progress, checked for shape rather than
    for size.
    """
    from epl.ingest import DIVISIONS, FIRST_SEASON, LAST_SEASON, load_matches, raw_season_path
    from epl.windows import LIVE_SEASON

    missing = [
        season
        for season in range(FIRST_SEASON, LAST_SEASON + 1)
        if not raw_season_path(season, DIVISIONS[0]).exists()
    ]
    if missing:
        pytest.skip(f"raw cache incomplete ({len(missing)} E0 Seasons missing)")

    matches = load_matches(divisions=(DIVISIONS[0],))
    return matches[matches["season"] < LIVE_SEASON].reset_index(drop=True)


@pytest.fixture(scope="module")
def live_e0() -> pd.DataFrame:
    """The Season in progress, which the anchor rule has to handle as readily as a closed one."""
    from epl.ingest import DIVISIONS, load_matches, raw_season_path
    from epl.windows import LIVE_SEASON

    if not raw_season_path(LIVE_SEASON, DIVISIONS[0]).exists():
        pytest.skip("raw cache holds no Season in progress")

    matches = load_matches(seasons=(LIVE_SEASON,), divisions=(DIVISIONS[0],))
    return matches.reset_index(drop=True)


@pytest.fixture(scope="module")
def summary(e0: pd.DataFrame) -> pd.DataFrame:
    return rounds.prediction_rounds(e0)
