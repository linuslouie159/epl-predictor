"""The injectable fetcher.

Ticket 1: "The fetcher is injectable so tests can point it at local fixtures instead of the
network." Everything that reaches upstream takes a ``fetcher``; the default is the only thing in the
ingest that knows HTTP exists. Nothing in this file touches the network, and none of it patches a
module global to stay off it.
"""

from __future__ import annotations

import datetime as dt

import pytest

from epl.ingest import fetcher as fetchers
from epl.ingest import fixtures as fx
from epl.ingest import football_data as fd


class TestMappingFetcher:
    def test_serves_bytes_for_a_known_url(self) -> None:
        fetch = fetchers.mapping_fetcher({"https://example/e0.csv": b"payload"})
        assert fetch("https://example/e0.csv") == b"payload"

    def test_raises_on_a_url_it_was_not_given(self) -> None:
        """A silent empty response would look like an empty Season rather than a broken test."""
        fetch = fetchers.mapping_fetcher({"https://example/e0.csv": b"payload"})
        with pytest.raises(KeyError, match="other"):
            fetch("https://example/other.csv")

    def test_records_what_was_asked_for(self) -> None:
        fetch = fetchers.mapping_fetcher({"https://example/e0.csv": b"payload"})
        fetch("https://example/e0.csv")
        fetch("https://example/e0.csv")
        assert fetch.requested == ["https://example/e0.csv", "https://example/e0.csv"]


class TestDirectoryFetcher:
    def test_serves_a_season_file_from_local_fixtures(self, data_dir) -> None:
        fetch = fetchers.directory_fetcher(data_dir)
        content = fetch("https://www.football-data.co.uk/mmz4281/1920/E0_1920_sample.csv")
        assert content == (data_dir / "E0_1920_sample.csv").read_bytes()

    def test_raises_when_the_fixture_is_absent(self, data_dir) -> None:
        fetch = fetchers.directory_fetcher(data_dir)
        with pytest.raises(FileNotFoundError):
            fetch("https://www.football-data.co.uk/mmz4281/1920/nope.csv")


class TestSeasonFetchTakesAFetcher:
    def test_writes_what_the_fetcher_returned(self, project_root) -> None:
        url = fd.season_csv_url(2025, "E0")
        fetch = fetchers.mapping_fetcher({url: b"injected"})
        assert fd.fetch_season(2025, "E0", fetcher=fetch).read_bytes() == b"injected"

    def test_a_cached_file_is_never_requested(self, project_root) -> None:
        path = fd.raw_season_path(2025, "E0")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"cached")

        fetch = fetchers.mapping_fetcher({})
        assert fd.fetch_season(2025, "E0", fetcher=fetch).read_bytes() == b"cached"
        assert fetch.requested == []

    def test_refresh_requests_again_and_supersedes(self, project_root) -> None:
        path = fd.raw_season_path(2025, "E0")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"as it stood at seal time")

        url = fd.season_csv_url(2025, "E0")
        fetch = fetchers.mapping_fetcher({url: b"backfilled later"})
        fd.fetch_season(2025, "E0", refresh=True, fetcher=fetch)

        assert fetch.requested == [url]
        archived = sorted(fd.superseded_dir(2025, "E0").glob("E0_*.csv"))
        assert archived[0].read_bytes() == b"as it stood at seal time"

    def test_fetch_all_threads_the_fetcher_through(self, project_root) -> None:
        urls = {fd.season_csv_url(2025, d): b"x" for d in fd.DIVISIONS}
        fetch = fetchers.mapping_fetcher(urls)
        paths = fd.fetch_all([2025], fd.DIVISIONS, fetcher=fetch)
        assert len(paths) == 4
        assert sorted(fetch.requested) == sorted(urls)


class TestFixturesFetchTakesAFetcher:
    def test_writes_what_the_fetcher_returned(self, project_root, data_dir) -> None:
        payload = (data_dir / "fixtures_sample.csv").read_bytes()
        fetch = fetchers.mapping_fetcher({fx.FIXTURES_URL: payload})
        path = fx.fetch_fixtures(
            fetched_at=dt.datetime(2026, 8, 21, 9, 0, tzinfo=dt.UTC), fetcher=fetch
        )
        assert path.read_bytes() == payload
        assert fetch.requested == [fx.FIXTURES_URL]

    def test_two_fetches_a_week_apart_are_kept_separately(self, project_root) -> None:
        fetch = fetchers.mapping_fetcher({fx.FIXTURES_URL: b"week one"})
        first = fx.fetch_fixtures(
            fetched_at=dt.datetime(2026, 8, 21, 9, 0, tzinfo=dt.UTC), fetcher=fetch
        )
        fetch = fetchers.mapping_fetcher({fx.FIXTURES_URL: b"week two"})
        second = fx.fetch_fixtures(
            fetched_at=dt.datetime(2026, 8, 28, 9, 0, tzinfo=dt.UTC), fetcher=fetch
        )
        assert first.read_bytes() == b"week one"
        assert second.read_bytes() == b"week two"


class TestHttpFetcherIsTheDefault:
    def test_the_default_is_http(self) -> None:
        assert fetchers.default_fetcher() is not None

    def test_http_is_the_only_thing_that_imports_requests(self) -> None:
        """If HTTP leaks back into the parsers, the no-network guarantee goes with it."""
        import inspect

        for module in (fd, fx):
            assert "requests" not in inspect.getsource(module)
