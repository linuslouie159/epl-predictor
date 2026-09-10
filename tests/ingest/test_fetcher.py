"""The injectable fetcher.

Ticket 1: "The fetcher is injectable so tests can point it at local fixtures instead of the
network." Everything that reaches upstream takes a ``fetcher``; the default is the only thing in the
ingest that knows HTTP exists. Nothing in this file touches the network, and none of it patches a
module global to stay off it.

That injectability is what makes the retry checkable too. ``epl.ingest.fetcher.retrying`` wraps a
Fetcher rather than living inside the HTTP one, so the failures below are hand-built objects with a
status on them and the waiting is a list that gets appended to — no network, no stand-in for
``requests``, and a suite that is not three seconds slower for owning a backoff.
"""

from __future__ import annotations

import datetime as dt

import pytest

from epl.ingest import fetcher as fetchers
from epl.ingest import fixtures as fx
from epl.ingest import football_data as fd

URL = "https://example/e0.csv"


class Response:
    """Just enough of a ``requests`` response for the retry policy to read a status off it."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class Refused(Exception):
    """Shaped like ``requests.exceptions.HTTPError``: it carries the response that refused."""

    def __init__(self, status: int) -> None:
        super().__init__(f"{status} Error for url: {URL}")
        self.response = Response(status)


class Unanswered(Exception):
    """Shaped like ``requests.exceptions.ConnectionError``: no response, so no status."""


class Flaky:
    """A Fetcher that raises the given failures in order, then serves bytes."""

    def __init__(self, *failures: Exception, payload: bytes = b"served") -> None:
        self.failures = list(failures)
        self.payload = payload
        self.calls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        if self.failures:
            raise self.failures.pop(0)
        return self.payload


class Waits:
    """A ``sleep`` that records what it was asked to wait for and waits for none of it."""

    def __init__(self) -> None:
        self.slept: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.slept.append(seconds)


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

    def test_the_default_is_the_retrying_one(self) -> None:
        """What every call site gets is the composition, not the bare HTTP call.

        Checked by type against a fetcher built the documented way, rather than by naming the
        private class: the claim is that `default_fetcher` returns what `retrying` returns.
        """
        assert type(fetchers.default_fetcher()) is type(fetchers.retrying(lambda url: b""))


class TestATransientFailureIsAskedAgain:
    """Issue: a `seal` fire died on a 503 from `fixtures.csv` on 10 Sep 2026.

    Three fires a day already gave the round three chances, but each one died at the fetch before it
    could learn whether there was a round at all — so a bad afternoon at the host took every one.
    """

    def test_a_503_is_asked_again_and_then_serves(self) -> None:
        flaky = Flaky(Refused(503), Refused(503))
        waits = Waits()

        assert fetchers.retrying(flaky, sleep=waits)(URL) == b"served"
        assert flaky.calls == [URL, URL, URL]
        assert waits.slept == list(fetchers.BACKOFF)

    def test_a_404_is_not_asked_again(self) -> None:
        """Upstream saying the file is not there, and asking three times does not publish it."""
        flaky = Flaky(Refused(404))
        waits = Waits()

        with pytest.raises(Refused):
            fetchers.retrying(flaky, sleep=waits)(URL)

        assert flaky.calls == [URL]
        assert waits.slept == []

    def test_a_rate_limit_is_asked_again(self) -> None:
        """The one 4xx that means "not now" rather than "not here"."""
        flaky = Flaky(Refused(429))
        assert fetchers.retrying(flaky, sleep=Waits())(URL) == b"served"
        assert flaky.calls == [URL, URL]

    def test_a_failure_with_no_response_at_all_is_asked_again(self) -> None:
        """A reset connection or a read timeout is not an answer, so there is nothing to believe."""
        flaky = Flaky(Unanswered("connection reset by peer"))
        assert fetchers.retrying(flaky, sleep=Waits())(URL) == b"served"
        assert flaky.calls == [URL, URL]

    def test_the_caller_sees_the_last_failure_itself(self) -> None:
        """Not a wrapper around it: a traceback and the bot both quote the real cause."""
        last = Refused(503)
        flaky = Flaky(Refused(503), Refused(503), last)
        waits = Waits()

        with pytest.raises(Refused) as raised:
            fetchers.retrying(flaky, sleep=waits)(URL)

        assert raised.value is last
        assert flaky.calls == [URL, URL, URL]
        assert waits.slept == list(fetchers.BACKOFF)

    def test_a_retry_says_so(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A fire that needed two attempts must not be identical on disk to one that needed one."""
        fetchers.retrying(Flaky(Refused(503)), sleep=Waits())(URL)

        printed = capsys.readouterr().out
        assert fetchers.RETRYING in printed
        assert URL in printed
        assert "HTTP 503" in printed
        assert "attempt 1 of 3" in printed

    def test_an_ordinary_fetch_is_silent_and_asks_once(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        flaky = Flaky()
        waits = Waits()

        assert fetchers.retrying(flaky, sleep=waits)(URL) == b"served"
        assert flaky.calls == [URL]
        assert waits.slept == []
        assert capsys.readouterr().out == ""


class TestTheRetryPolicyStandsAlone:
    """`worth_retrying` is public because the policy is the interesting half, not the loop."""

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_not_now(self, status: int) -> None:
        assert fetchers.worth_retrying(Refused(status))

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 410])
    def test_not_here(self, status: int) -> None:
        assert not fetchers.worth_retrying(Refused(status))

    def test_no_answer_at_all(self) -> None:
        assert fetchers.worth_retrying(Unanswered("name or service not known"))
