"""Forward Fixtures from the rolling ``fixtures.csv``."""

from __future__ import annotations

import datetime as dt

import pytest

from epl.ingest import fixtures as fx
from epl.ingest import football_data as fd
from epl.ingest.football_data import IngestError


class TestParseFixtures:
    def test_keeps_only_the_english_tiers(self, data_dir, resolver) -> None:
        """The file carries every league Football-Data covers, Spain's included."""
        frame = fx.parse_fixtures(data_dir / "fixtures_sample.csv", resolver=resolver)
        assert set(frame["division"]) == {"E2"}
        assert len(frame) == 1

    def test_resolves_the_fuller_spellings_this_file_uses(self, data_dir, resolver) -> None:
        """The rolling file writes 'Sheffield Wed' where the Season files write 'Sheffield Weds'."""
        row = fx.parse_fixtures(data_dir / "fixtures_sample.csv", resolver=resolver).iloc[0]
        assert row["home_club"] == "sheffield_wednesday"
        assert row["away_club"] == "bradford"

    def test_carries_the_market_line_for_an_unplayed_fixture(self, data_dir, resolver) -> None:
        row = fx.parse_fixtures(data_dir / "fixtures_sample.csv", resolver=resolver).iloc[0]
        assert row["prematch_odds_home"] == pytest.approx(2.44)
        assert row["prematch_odds_draw"] == pytest.approx(3.36)
        assert row["prematch_odds_away"] == pytest.approx(2.68)

    def test_a_fixture_carries_no_result(self, data_dir, resolver) -> None:
        """A Fixture exists before it is played and carries no result until it is (CONTEXT.md)."""
        frame = fx.parse_fixtures(data_dir / "fixtures_sample.csv", resolver=resolver)
        assert tuple(frame.columns) == fx.FIXTURE_COLUMNS
        assert not {"home_goals", "away_goals", "outcome"} & set(frame.columns)

    def test_reads_date_and_kickoff_time(self, data_dir, resolver) -> None:
        row = fx.parse_fixtures(data_dir / "fixtures_sample.csv", resolver=resolver).iloc[0]
        assert row["date"] == dt.date(2026, 8, 20)
        assert row["time"] == "20:00"

    def test_can_be_narrowed_to_one_tier(self, data_dir, resolver) -> None:
        frame = fx.parse_fixtures(
            data_dir / "fixtures_sample.csv", divisions=("E0",), resolver=resolver
        )
        assert frame.empty
        assert tuple(frame.columns) == fx.FIXTURE_COLUMNS

    def test_rejects_a_file_missing_required_columns(self, write_csv, resolver) -> None:
        path = write_csv("thin.csv", "Div,Date,HomeTeam", "E0,20/08/2026,Arsenal")
        with pytest.raises(IngestError, match="missing required columns"):
            fx.parse_fixtures(path, resolver=resolver)


class TestCachePaths:
    def test_stamps_each_fetch_with_its_instant(self, project_root) -> None:
        """The upstream file is replaced in place; two fetches are different evidence."""
        first = fx.raw_fixtures_path(dt.datetime(2026, 8, 21, 9, 0, tzinfo=dt.UTC))
        second = fx.raw_fixtures_path(dt.datetime(2026, 8, 28, 9, 0, tzinfo=dt.UTC))
        assert first != second
        assert first.name == "fixtures_20260821T090000Z.csv"

    def test_normalises_the_instant_to_utc(self, project_root) -> None:
        aware = dt.datetime(2026, 8, 21, 11, 0, tzinfo=dt.timezone(dt.timedelta(hours=2)))
        assert fx.raw_fixtures_path(aware).name == "fixtures_20260821T090000Z.csv"

    def test_latest_is_none_when_nothing_is_cached(self, project_root) -> None:
        assert fx.latest_fixtures_path() is None

    def test_latest_returns_the_most_recent_fetch(self, project_root) -> None:
        fx.fixtures_dir().mkdir(parents=True, exist_ok=True)
        for stamp in ["20260821T090000Z", "20260828T090000Z", "20260814T090000Z"]:
            (fx.fixtures_dir() / f"fixtures_{stamp}.csv").write_bytes(b"x")
        assert fx.latest_fixtures_path().name == "fixtures_20260828T090000Z.csv"

    def test_fetch_never_overwrites_an_earlier_fetch(self, project_root, monkeypatch) -> None:
        monkeypatch.setattr(fd.requests, "get", _fake_get(b"week one"))
        first = fx.fetch_fixtures(fetched_at=dt.datetime(2026, 8, 21, 9, 0, tzinfo=dt.UTC))
        monkeypatch.setattr(fd.requests, "get", _fake_get(b"week two"))
        second = fx.fetch_fixtures(fetched_at=dt.datetime(2026, 8, 28, 9, 0, tzinfo=dt.UTC))
        assert first.read_bytes() == b"week one"
        assert second.read_bytes() == b"week two"


def _fake_get(payload: bytes):
    class _Response:
        content = payload

        def raise_for_status(self) -> None:
            return None

    def _get(url, timeout=None):
        return _Response()

    return _get
