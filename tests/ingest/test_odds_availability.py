"""Which odds columns each Season file actually carries.

Ticket 3: availability is *recorded* per Season — none before 2002/03, Bet365 from 2002/03,
``BbAv*`` from 2005/06, ``Avg*`` and ``AvgC*`` from 2019/20 — "and a missing Market Line is never
treated as a zero".

The record matters because absence and value look the same downstream once a column is a float
column full of nulls. A Season with no Market Line is not a Season where the market thought a home
win paid nothing; it is a Season with no market to compare against, and the benchmark code has to be
able to tell the difference without re-deriving it from the raw files.
"""

from __future__ import annotations

import pandas as pd
import pytest

from epl.ingest import football_data as fd


def _availability(path, season: int, division: str = "E0") -> pd.Series:
    return fd.odds_availability_for(path, season, division)


class TestPerFileAvailability:
    def test_2000_01_has_no_odds_at_all(self, data_dir) -> None:
        row = _availability(data_dir / "E0_0001_sample.csv", 2000)
        assert not row["bet365"]
        assert row["prematch_average"] is pd.NA or pd.isna(row["prematch_average"])
        assert not row["closing_average"]
        assert not row["has_market_line"]

    def test_2002_03_has_bet365_but_no_market_average(self, data_dir) -> None:
        """Bet365 arrives three Seasons before the Market Line does."""
        row = _availability(data_dir / "E0_0203_sample.csv", 2002)
        assert row["bet365"]
        assert pd.isna(row["prematch_average"])
        assert not row["has_market_line"]

    def test_2005_06_has_the_market_average_under_its_old_spelling(self, data_dir) -> None:
        row = _availability(data_dir / "E0_0506_sample.csv", 2005)
        assert row["prematch_average"] == "BbAv"
        assert row["has_market_line"]
        assert not row["closing_average"]

    def test_2019_20_has_the_new_spelling_and_the_ceiling_line(self, data_dir) -> None:
        row = _availability(data_dir / "E0_1920_sample.csv", 2019)
        assert row["prematch_average"] == "Avg"
        assert row["has_market_line"]
        assert row["closing_average"]

    def test_records_which_season_and_tier_it_describes(self, data_dir) -> None:
        row = _availability(data_dir / "E0_1920_sample.csv", 2019, "E0")
        assert row["season"] == 2019
        assert row["division"] == "E0"
        assert row["season_label"] == "2019/20"


class TestAbsenceIsNotZero:
    def test_a_missing_market_line_is_null_not_zero(self, data_dir, resolver) -> None:
        """The failure this record exists to prevent."""
        frame = fd.parse_season_csv(data_dir / "E0_0001_sample.csv", 2000, "E0", resolver=resolver)
        assert frame["prematch_odds_home"].isna().all()
        assert (frame["prematch_odds_home"] == 0).sum() == 0

    def test_a_present_market_line_is_never_zero_either(self, data_dir, resolver) -> None:
        """Odds are payout multipliers; anything at or below 1 would be nonsense."""
        frame = fd.parse_season_csv(data_dir / "E0_1920_sample.csv", 2019, "E0", resolver=resolver)
        assert (frame["prematch_odds_home"] > 1).all()


class TestTableOverTheCache:
    def test_returns_one_row_per_season_and_tier(self, project_root, data_dir) -> None:
        for season, sample in [(2000, "E0_0001_sample.csv"), (2019, "E0_1920_sample.csv")]:
            target = fd.raw_season_path(season, "E0")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((data_dir / sample).read_bytes())

        table = fd.odds_availability([2000, 2019], ("E0",))
        assert list(table["season"]) == [2000, 2019]
        assert list(table["has_market_line"]) == [False, True]

    def test_skips_seasons_absent_from_the_cache(self, project_root) -> None:
        assert fd.odds_availability([2000], ("E0",)).empty

    def test_columns_are_stable(self, project_root, data_dir) -> None:
        target = fd.raw_season_path(2019, "E0")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((data_dir / "E0_1920_sample.csv").read_bytes())
        assert tuple(fd.odds_availability([2019], ("E0",)).columns) == fd.ODDS_AVAILABILITY_COLUMNS


@pytest.mark.cache
class TestAgainstTheRealCorpus:
    """The era ladder in docs/DECISIONS.md, re-derived from the cache rather than trusted."""

    def test_the_ladder_holds_for_the_premier_league(self) -> None:
        from epl.windows import FIRST_SEASON, LAST_SEASON

        if not fd.raw_season_path(LAST_SEASON, "E0").exists():
            pytest.skip("raw cache not populated")

        table = fd.odds_availability(range(FIRST_SEASON, LAST_SEASON + 1), ("E0",))
        by_season = table.set_index("season")

        assert not by_season.loc[2000:2001, "bet365"].any()
        assert by_season.loc[2002:2025, "bet365"].all()
        assert not by_season.loc[2000:2004, "has_market_line"].any()
        assert by_season.loc[2005:2025, "has_market_line"].all()
        assert (by_season.loc[2005:2018, "prematch_average"] == "BbAv").all()
        assert (by_season.loc[2019:2025, "prematch_average"] == "Avg").all()
        assert not by_season.loc[2000:2018, "closing_average"].any()
        assert by_season.loc[2019:2025, "closing_average"].all()
