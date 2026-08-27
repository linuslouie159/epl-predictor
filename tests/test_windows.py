"""Season identity and the two Windows.

These constants decide which Fixtures are ever scored, so they are defined once and imported
everywhere. A second definition drifting out of step with this one is how a hyperparameter tuned on
the Burn-In Window quietly ends up tuned on scored data.
"""

from __future__ import annotations

import pytest

from epl import windows


class TestSeasonIdentity:
    def test_a_season_is_its_start_year(self) -> None:
        assert windows.season_label(2019) == "2019/20"

    def test_labels_a_century_rollover(self) -> None:
        assert windows.season_label(1999) == "1999/00"

    @pytest.mark.parametrize(("season", "label"), [(2000, "2000/01"), (2025, "2025/26")])
    def test_labels_the_boundaries_of_the_ingested_range(self, season: int, label: str) -> None:
        assert windows.season_label(season) == label


class TestWindows:
    def test_the_burn_in_window_is_2000_01_to_2004_05(self) -> None:
        assert list(windows.BURN_IN_WINDOW) == [2000, 2001, 2002, 2003, 2004]

    def test_the_evaluation_window_is_2005_06_to_2025_26(self) -> None:
        assert list(windows.EVALUATION_WINDOW) == list(range(2005, 2026))

    def test_the_evaluation_window_is_21_seasons(self) -> None:
        assert len(windows.EVALUATION_WINDOW) == 21

    def test_the_windows_do_not_overlap(self) -> None:
        """The whole leakage protocol rests on this."""
        assert not set(windows.BURN_IN_WINDOW) & set(windows.EVALUATION_WINDOW)

    def test_the_windows_are_contiguous(self) -> None:
        assert max(windows.BURN_IN_WINDOW) + 1 == min(windows.EVALUATION_WINDOW)

    def test_they_span_every_ingested_season_except_the_one_being_played(self) -> None:
        """The two Windows are the *closed* Seasons. The Season in progress is a third span, and
        is in neither (ADR 0010): never fitted on, never backfilled, scored on its own board."""
        spanned = list(windows.BURN_IN_WINDOW) + list(windows.EVALUATION_WINDOW)

        assert spanned == list(range(windows.FIRST_SEASON, windows.LIVE_SEASON))
        assert windows.LIVE_SEASON not in spanned

    def test_27_seasons_are_ingested_and_21_are_scored(self) -> None:
        """26 closed plus the one being played, 21 scored — deliberate, and the thing readers
        assume is a bug (ADR 0008, ADR 0010)."""
        assert len(range(windows.FIRST_SEASON, windows.LAST_SEASON + 1)) == 27
        assert len(windows.EVALUATION_WINDOW) == 21

    def test_the_live_season_is_the_last_one_ingested(self) -> None:
        """Defined as `LAST_SEASON` rather than as a second literal, so there is one place to move
        at the start of a campaign and no way for the two to disagree."""
        assert windows.LIVE_SEASON == windows.LAST_SEASON


class TestClassification:
    @pytest.mark.parametrize("season", [2000, 2004])
    def test_burn_in_seasons_are_never_scored(self, season: int) -> None:
        assert windows.is_burn_in(season)
        assert not windows.is_evaluation(season)

    @pytest.mark.parametrize("season", [2005, 2025])
    def test_evaluation_seasons_are_scored(self, season: int) -> None:
        assert windows.is_evaluation(season)
        assert not windows.is_burn_in(season)

    @pytest.mark.parametrize("season", [1999, 2026])
    def test_a_season_outside_the_ingested_range_is_in_neither_window(self, season: int) -> None:
        assert not windows.is_burn_in(season)
        assert not windows.is_evaluation(season)


class TestNoModuleHardCodesASeason:
    """Ticket 2: 'defined once and imported everywhere — no module hard-codes a season year'."""

    def test_the_ingest_takes_its_range_from_here(self) -> None:
        from epl.ingest import football_data as fd

        assert fd.FIRST_SEASON is windows.FIRST_SEASON
        assert fd.LAST_SEASON is windows.LAST_SEASON
