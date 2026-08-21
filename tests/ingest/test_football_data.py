"""The Football-Data Season ingester.

The sample files under ``tests/data/`` are real upstream rows, one per era boundary recorded in
docs/DECISIONS.md, so these tests fail if the era handling is wrong rather than if a hand-written
fixture was written wrong.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from epl.clubs import UnknownAliasError
from epl.ingest import football_data as fd
from epl.ingest.fetcher import mapping_fetcher
from epl.ingest.football_data import IngestError


class TestSeasonCode:
    @pytest.mark.parametrize(
        ("season", "code"),
        [(2000, "0001"), (2005, "0506"), (2009, "0910"), (2019, "1920"), (2025, "2526")],
    )
    def test_encodes_a_season_by_its_start_year(self, season: int, code: str) -> None:
        assert fd.season_code(season) == code

    def test_handles_the_century_rollover(self) -> None:
        assert fd.season_code(1999) == "9900"

    def test_labels_a_season_for_humans(self) -> None:
        assert fd.season_label(2005) == "2005/06"
        assert fd.season_label(1999) == "1999/00"


class TestUrlsAndPaths:
    def test_builds_the_upstream_url(self) -> None:
        assert (
            fd.season_csv_url(2025, "E0")
            == "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
        )

    def test_the_cache_layout_mirrors_the_url(self, project_root) -> None:
        path = fd.raw_season_path(2025, "E0")
        assert path == project_root / "data" / "raw" / "football-data" / "2526" / "E0.csv"

    @pytest.mark.parametrize("division", ["E4", "e0", "SP1", ""])
    def test_rejects_a_division_outside_the_english_pyramid(self, division: str) -> None:
        with pytest.raises(ValueError, match="unknown division"):
            fd.season_csv_url(2025, division)

    def test_covers_all_four_tiers(self) -> None:
        assert fd.DIVISIONS == ("E0", "E1", "E2", "E3")


class TestReadRawCsv:
    def test_strips_the_byte_order_mark_on_newer_files(self, data_dir) -> None:
        frame = fd.read_raw_csv(data_dir / "E0_2526_sample.csv")
        assert frame.columns[0] == "Div"

    def test_reads_older_files_without_a_mark(self, data_dir) -> None:
        frame = fd.read_raw_csv(data_dir / "E0_0001_sample.csv")
        assert frame.columns[0] == "Div"
        assert len(frame) == 3

    def test_drops_unnamed_trailing_columns(self, data_dir) -> None:
        frame = fd.read_raw_csv(data_dir / "E0_0203_sample.csv")
        assert all(not c.startswith("Unnamed") and c for c in frame.columns)

    def test_pads_a_row_that_runs_short_of_its_header(self, write_csv) -> None:
        path = write_csv(
            "short.csv", "Div,Date,HomeTeam,AwayTeam,X", "E0,19/08/00,Charlton,Man City"
        )
        frame = fd.read_raw_csv(path)
        assert frame.loc[0, "AwayTeam"] == "Man City"
        assert pd.isna(frame.loc[0, "X"])

    def test_drops_blank_surplus_fields(self, write_csv) -> None:
        path = write_csv(
            "long.csv", "Div,Date,HomeTeam,AwayTeam", "E0,19/08/00,Charlton,Man City,,,"
        )
        frame = fd.read_raw_csv(path)
        assert list(frame.columns) == ["Div", "Date", "HomeTeam", "AwayTeam"]
        assert frame.loc[0, "AwayTeam"] == "Man City"

    def test_refuses_surplus_that_is_not_blank(self, write_csv) -> None:
        """Non-blank surplus means the columns shifted, which would silently misread odds."""
        path = write_csv(
            "shifted.csv", "Div,Date,HomeTeam,AwayTeam", "E0,19/08/00,Charlton,Man City,4"
        )
        with pytest.raises(IngestError, match="surplus is not blank"):
            fd.read_raw_csv(path)

    def test_drops_blank_trailing_rows(self, write_csv) -> None:
        path = write_csv(
            "blanks.csv", "Div,Date,HomeTeam,AwayTeam", "E0,19/08/00,Charlton,Man City", ",,,", ""
        )
        assert len(fd.read_raw_csv(path)) == 1


class TestParseSeasonCsv:
    def test_reads_a_2000_01_row(self, data_dir, resolver) -> None:
        frame = fd.parse_season_csv(data_dir / "E0_0001_sample.csv", 2000, "E0", resolver=resolver)
        row = frame.iloc[0]
        assert row["season"] == 2000
        assert row["division"] == "E0"
        assert row["date"] == dt.date(2000, 8, 19)
        assert row["home_club"] == "charlton"
        assert row["away_club"] == "man_city"
        assert row["home_goals"] == 4
        assert row["away_goals"] == 0
        assert row["outcome"] == "H"
        assert row["home_shots"] == 17
        assert row["referee"] == "Rob Harris"

    def test_2000_01_has_no_market_line(self, data_dir, resolver) -> None:
        """No market-average odds exist before 2005/06, so those Seasons have no Market Line."""
        frame = fd.parse_season_csv(data_dir / "E0_0001_sample.csv", 2000, "E0", resolver=resolver)
        assert frame["prematch_odds_home"].isna().all()
        assert frame["closing_odds_home"].isna().all()

    def test_has_no_kickoff_time_before_2019_20(self, data_dir, resolver) -> None:
        frame = fd.parse_season_csv(data_dir / "E0_0001_sample.csv", 2000, "E0", resolver=resolver)
        assert frame["time"].isna().all()

    def test_parses_four_digit_years(self, data_dir, resolver) -> None:
        frame = fd.parse_season_csv(data_dir / "E0_0203_sample.csv", 2002, "E0", resolver=resolver)
        assert frame.iloc[0]["date"] == dt.date(2002, 8, 17)

    def test_reads_the_pre_2019_market_average_from_bbav(self, data_dir, resolver) -> None:
        frame = fd.parse_season_csv(data_dir / "E0_0506_sample.csv", 2005, "E0", resolver=resolver)
        row = frame.iloc[0]
        assert row["prematch_odds_home"] == pytest.approx(2.2)
        assert row["prematch_odds_draw"] == pytest.approx(3.16)
        assert row["prematch_odds_away"] == pytest.approx(3.05)

    def test_has_no_ceiling_line_before_2019_20(self, data_dir, resolver) -> None:
        frame = fd.parse_season_csv(data_dir / "E0_0506_sample.csv", 2005, "E0", resolver=resolver)
        assert frame["closing_odds_home"].isna().all()

    def test_reads_the_post_2019_market_average_from_avg(self, data_dir, resolver) -> None:
        """BbAv* was renamed Avg* in 2019/20; the spliced column must not notice."""
        frame = fd.parse_season_csv(data_dir / "E0_1920_sample.csv", 2019, "E0", resolver=resolver)
        row = frame.iloc[0]
        assert row["prematch_odds_home"] == pytest.approx(1.14)
        assert row["prematch_odds_draw"] == pytest.approx(8.75)
        assert row["prematch_odds_away"] == pytest.approx(19.83)

    def test_reads_the_ceiling_line_from_2019_20(self, data_dir, resolver) -> None:
        frame = fd.parse_season_csv(data_dir / "E0_1920_sample.csv", 2019, "E0", resolver=resolver)
        row = frame.iloc[0]
        assert row["closing_odds_home"] == pytest.approx(1.14)
        assert row["closing_odds_draw"] == pytest.approx(9.52)
        assert row["closing_odds_away"] == pytest.approx(19.18)

    def test_keeps_the_kickoff_time_from_2019_20(self, data_dir, resolver) -> None:
        frame = fd.parse_season_csv(data_dir / "E0_1920_sample.csv", 2019, "E0", resolver=resolver)
        assert frame.iloc[0]["time"] == "20:00"

    def test_reads_the_current_season(self, data_dir, resolver) -> None:
        frame = fd.parse_season_csv(data_dir / "E0_2526_sample.csv", 2025, "E0", resolver=resolver)
        row = frame.iloc[0]
        assert row["home_club"] == "liverpool"
        assert row["away_club"] == "bournemouth"
        assert row["date"] == dt.date(2025, 8, 15)
        assert row["prematch_odds_home"] == pytest.approx(1.31)

    def test_emits_the_canonical_columns_in_order(self, data_dir, resolver) -> None:
        frame = fd.parse_season_csv(data_dir / "E0_1920_sample.csv", 2019, "E0", resolver=resolver)
        assert tuple(frame.columns) == fd.MATCH_COLUMNS

    def test_a_22_club_season_of_462_fixtures_parses_intact(self, write_csv, resolver) -> None:
        """Fixture counts come from the data, never from an assumed 380.

        Seasons before 1995/96 had 22 Clubs and 462 Fixtures. Those Seasons sit outside the
        ingested range today, but the loader must not be the reason they could not be added — an
        assumption of 380 anywhere would silently truncate or misalign them.
        """
        # The actual 1994/95 Premier League, in Football-Data's spellings.
        clubs = [
            "Arsenal", "Aston Villa", "Blackburn", "Chelsea", "Coventry", "Crystal Palace",
            "Everton", "Ipswich", "Leeds", "Leicester", "Liverpool", "Man City", "Man United",
            "Newcastle", "Norwich", "Nott'm Forest", "QPR", "Sheffield Weds", "Southampton",
            "Tottenham", "West Ham", "Wimbledon",
        ]
        assert len(clubs) == 22

        rows = [
            f"E0,20/08/94,{home},{away},2,1,H"
            for home in clubs
            for away in clubs
            if home != away
        ]
        path = write_csv("season_1994.csv", "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR", *rows)

        frame = fd.parse_season_csv(path, 1994, "E0", resolver=resolver)
        assert len(frame) == 462
        assert frame["home_club"].nunique() == 22

    def test_drops_rows_with_no_outcome(self, write_csv, resolver) -> None:
        """A Season file occasionally carries an unplayed or abandoned Fixture."""
        path = write_csv(
            "partial.csv",
            "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR",
            "E0,19/08/00,Charlton,Man City,4,0,H",
            "E0,20/08/00,Arsenal,Chelsea,,,",
        )
        frame = fd.parse_season_csv(path, 2000, "E0", resolver=resolver)
        assert len(frame) == 1
        assert frame.iloc[0]["home_club"] == "charlton"


class TestParseGuards:
    def test_rejects_a_file_missing_required_columns(self, write_csv, resolver) -> None:
        path = write_csv("thin.csv", "Div,Date,HomeTeam,AwayTeam", "E0,19/08/00,Charlton,Man City")
        with pytest.raises(IngestError, match="missing required columns"):
            fd.parse_season_csv(path, 2000, "E0", resolver=resolver)

    def test_rejects_a_date_outside_the_season(self, write_csv, resolver) -> None:
        """A misparsed year is the corruption most likely to look like a well-behaved model."""
        path = write_csv(
            "wrong_year.csv",
            "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR",
            "E0,19/08/98,Charlton,Man City,4,0,H",
        )
        with pytest.raises(IngestError, match="outside Season 2000/01"):
            fd.parse_season_csv(path, 2000, "E0", resolver=resolver)

    def test_rejects_an_unparseable_date(self, write_csv, resolver) -> None:
        path = write_csv(
            "bad_date.csv",
            "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR",
            "E0,19-Aug-2000,Charlton,Man City,4,0,H",
        )
        with pytest.raises(IngestError, match="unparseable dates"):
            fd.parse_season_csv(path, 2000, "E0", resolver=resolver)

    def test_rejects_both_spellings_of_the_market_average(self, write_csv, resolver) -> None:
        """BbAvH and AvgH are one quantity; if both ever appeared, splicing would double-count."""
        path = write_csv(
            "both.csv",
            "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,BbAvH,AvgH",
            "E0,19/08/00,Charlton,Man City,4,0,H,2.2,2.3",
        )
        with pytest.raises(IngestError, match="same quantity"):
            fd.parse_season_csv(path, 2000, "E0", resolver=resolver)

    def test_an_unknown_club_stops_the_ingest(self, write_csv, resolver) -> None:
        path = write_csv(
            "unknown.csv",
            "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR",
            "E0,19/08/00,Charlton,Real Madrid,4,0,H",
        )
        with pytest.raises(UnknownAliasError, match="Real Madrid"):
            fd.parse_season_csv(path, 2000, "E0", resolver=resolver)


class TestFetch:
    def test_reuses_a_cached_file_without_a_request(self, project_root) -> None:
        path = fd.raw_season_path(2025, "E0")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"cached")

        fetch = mapping_fetcher({})  # raises if asked for anything at all
        assert fd.fetch_season(2025, "E0", fetcher=fetch).read_bytes() == b"cached"
        assert fetch.requested == []

    def test_refresh_updates_the_cache(self, project_root) -> None:
        """The current Season's upstream file grows weekly, so it must be re-fetchable."""
        path = fd.raw_season_path(2025, "E0")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stale")
        fetch = mapping_fetcher({fd.season_csv_url(2025, "E0"): b"fresh"})
        assert fd.fetch_season(2025, "E0", refresh=True, fetcher=fetch).read_bytes() == b"fresh"

    def test_refresh_keeps_the_bytes_it_replaces(self, project_root) -> None:
        """Upstream backfills odds into rows already published (ADR 0005).

        Overwriting in place would destroy the only record of what the cache held when a Sealed
        Prediction was made, so the superseded copy is kept.
        """
        path = fd.raw_season_path(2025, "E0")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"as it stood at seal time")

        fetch = mapping_fetcher({fd.season_csv_url(2025, "E0"): b"backfilled a month later"})
        fd.fetch_season(2025, "E0", refresh=True, fetcher=fetch)

        archived = sorted(fd.superseded_dir(2025, "E0").glob("E0_*.csv"))
        assert len(archived) == 1
        assert archived[0].read_bytes() == b"as it stood at seal time"
        assert path.read_bytes() == b"backfilled a month later"

    def test_refresh_archives_nothing_when_upstream_is_unchanged(self, project_root) -> None:
        """Most refreshes change nothing; an archive per run would be noise, not evidence."""
        path = fd.raw_season_path(2025, "E0")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"identical")

        fetch = mapping_fetcher({fd.season_csv_url(2025, "E0"): b"identical"})
        fd.fetch_season(2025, "E0", refresh=True, fetcher=fetch)

        assert not fd.superseded_dir(2025, "E0").exists()

    def test_superseded_copies_are_not_read_back_as_matches(
        self, project_root, data_dir, resolver
    ) -> None:
        """The archive is evidence, not corpus; loading it would double-count a Season."""
        path = fd.raw_season_path(2019, "E0")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((data_dir / "E0_1920_sample.csv").read_bytes())

        archive = fd.superseded_dir(2019, "E0")
        archive.mkdir(parents=True, exist_ok=True)
        (archive / "E0_20190101T000000Z.csv").write_bytes(
            (data_dir / "E0_1920_sample.csv").read_bytes()
        )

        assert len(fd.load_matches([2019], ("E0",), resolver=resolver)) == 3

    def test_writes_the_bytes_upstream_sent(self, project_root) -> None:
        payload = "Div,Date\r\nE0,19/08/00\r\n".encode("cp1252")
        fetch = mapping_fetcher({fd.season_csv_url(2000, "E0"): payload})
        assert fd.fetch_season(2000, "E0", fetcher=fetch).read_bytes() == payload


class TestLoadMatches:
    def test_returns_the_canonical_columns_when_the_cache_is_empty(self, project_root) -> None:
        frame = fd.load_matches([2025], ("E0",))
        assert tuple(frame.columns) == fd.MATCH_COLUMNS
        assert frame.empty

    def test_concatenates_seasons_and_tiers_in_date_order(
        self, project_root, data_dir, resolver
    ) -> None:
        for season, sample in [(2000, "E0_0001_sample.csv"), (2019, "E0_1920_sample.csv")]:
            target = fd.raw_season_path(season, "E0")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((data_dir / sample).read_bytes())

        frame = fd.load_matches([2019, 2000], ("E0",), resolver=resolver)
        assert len(frame) == 6
        assert frame["date"].is_monotonic_increasing
        assert set(frame["season"]) == {2000, 2019}


