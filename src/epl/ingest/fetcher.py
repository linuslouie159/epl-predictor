"""How the ingest reaches upstream — and the seam that lets tests not reach it at all.

A Fetcher is anything that turns a URL into bytes. Every function that pulls from Football-Data
takes one, so this module is the only place in the ingest that knows HTTP exists. Tests inject
:func:`mapping_fetcher` or :func:`directory_fetcher` and run with no network access, without
patching a module global — patching one would leave the production path untested and would break
silently the moment the call site moved.

Both test fetchers **raise** on a URL they were not given, rather than returning empty bytes. An
empty response parses as a Season with no matches, which looks like a data gap rather than a broken
test.

**One fetch is not one attempt, and that is new.** A 503 from ``fixtures.csv`` used to end a `seal`
fire outright: nothing between :func:`epl.ingest.fixtures.fetch_fixtures` and ``main`` catches a
``requests`` exception, so the run exited 1 on a traceback before it could learn whether there was
even a round to seal. So :func:`default_fetcher` wraps the real one in :func:`retrying`, and what
that does and deliberately does not cover is :func:`worth_retrying`'s docstring.

**It covers a moment and not a day, and the difference was measured.** On 8 Sep 2026 all four of the
Pi's scheduled fires failed on 503s, from 06:00 to 18:30 UK and across two different URLs —
Football-Data was down for the day, and no backoff that fits inside a sealing window would have
helped. Three attempts are for the commoner shape, one fire meeting one bad moment at a host that is
otherwise up. See docs/DECISIONS.md, "The fetch upstream refused", and open risk 7.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

#: Waits before each retry, in seconds; two entries mean three attempts. Sized to outlast a restart
#: at the host rather than a maintenance window — ten seconds of waiting, against a sealing window
#: that opens at 16:00 UK and shuts no earlier than a 19:45 first kickoff, fired three times a day.
#: A schedule that retried for minutes would be spending the one thing it cannot get back, which is
#: the round.
BACKOFF: tuple[float, ...] = (2.0, 8.0)

#: HTTP's two ways of saying "not now": anything the server blames on itself, and a rate limit.
SERVER_ERROR = 500
TOO_MANY_REQUESTS = 429

#: What a retried fetch prints, so that a fire which needed two attempts is not identical on disk to
#: one that needed a single attempt. The line lands inside `deploy/run_live.sh`'s own ``===== RUN``
#: block, which is what :func:`epl.bot.answers.failure` quotes and what a person opens
#: `deploy/logs/live_loop.log` to read — a retry that healed silently would be a fact about
#: upstream's reliability that nothing recorded.
RETRYING = "[epl.ingest] upstream would not serve"


class Fetcher(Protocol):
    """Turns a URL into bytes."""

    def __call__(self, url: str) -> bytes: ...


class RecordingFetcher:
    """Base for the test fetchers: remembers every URL it was asked for.

    ``requested`` is what lets a test assert that a cached file was *not* re-downloaded — an
    assertion about absence that no amount of checking the returned bytes can make.
    """

    def __init__(self) -> None:
        self.requested: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.requested.append(url)
        return self._fetch(url)

    def _fetch(self, url: str) -> bytes:  # pragma: no cover - overridden
        raise NotImplementedError


class _MappingFetcher(RecordingFetcher):
    def __init__(self, responses: Mapping[str, bytes]) -> None:
        super().__init__()
        self._responses = dict(responses)

    def _fetch(self, url: str) -> bytes:
        try:
            return self._responses[url]
        except KeyError:
            raise KeyError(
                f"no canned response for {url!r}; known URLs: {sorted(self._responses)}"
            ) from None


class _DirectoryFetcher(RecordingFetcher):
    def __init__(self, root: Path | str) -> None:
        super().__init__()
        self._root = Path(root)

    def _fetch(self, url: str) -> bytes:
        path = self._root / url.rsplit("/", 1)[-1]
        if not path.exists():
            raise FileNotFoundError(f"{url} maps to {path}, which does not exist")
        return path.read_bytes()


class _RetryingFetcher:
    """One Fetcher wrapping another. See :func:`retrying` for why it is shaped this way."""

    def __init__(
        self,
        inner: Fetcher,
        backoff: tuple[float, ...],
        sleep: Callable[[float], None],
    ) -> None:
        self._inner = inner
        self._backoff = tuple(backoff)
        self._sleep = sleep

    @property
    def attempts(self) -> int:
        """How many times one URL is asked for before the failure becomes the caller's."""
        return len(self._backoff) + 1

    def __call__(self, url: str) -> bytes:
        # The last attempt is deliberately outside the loop rather than a final pass through it.
        # Its exception is the one the caller must see — a traceback names the real cause, and
        # `epl.bot.answers.failure` quotes the loop's own words — so letting it propagate from here
        # means there is no re-raise to get subtly wrong and no unreachable branch underneath.
        for attempt, pause in enumerate(self._backoff, start=1):
            try:
                return self._inner(url)
            except Exception as failure:
                if not worth_retrying(failure):
                    raise
                print(
                    f"{RETRYING} {url}: {_why(failure)}; attempt {attempt} of "
                    f"{self.attempts}, retrying in {pause:g}s"
                )
                self._sleep(pause)
        return self._inner(url)


def worth_retrying(failure: BaseException) -> bool:
    """Whether a failed fetch is upstream saying "not now" rather than "not here".

    The status is read by attribute rather than by catching a ``requests`` class, so the lazy import
    in :func:`http_fetcher` stays the only place in this project that names the library.

    **A 4xx other than 429 is not retried, and the asymmetry is the point.** A 404 is upstream
    saying the file is not there, and asking three times does not publish it: ``python -m epl.live
    score`` refreshes the Live Season twice a week all year round, including the weeks before a new
    Season's four files appear, and retrying that would dress "not published yet" as a network fault
    and pay the backoff four times over to do it.

    A failure carrying no status at all is the opposite case and is retried. A reset connection, a
    DNS failure or a read timeout is not an answer from upstream, so there is nothing in it to
    believe.
    """
    response = getattr(failure, "response", None)
    status = getattr(response, "status_code", None)
    if status is None:
        return True
    return status >= SERVER_ERROR or status == TOO_MANY_REQUESTS


def _why(failure: BaseException) -> str:
    """The failure in a few characters: its status where it has one, otherwise its type."""
    status = getattr(getattr(failure, "response", None), "status_code", None)
    return f"HTTP {status}" if status is not None else type(failure).__name__


def mapping_fetcher(responses: Mapping[str, bytes]) -> _MappingFetcher:
    """A Fetcher serving canned bytes per URL. Raises on anything it was not given."""
    return _MappingFetcher(responses)


def directory_fetcher(root: Path | str) -> _DirectoryFetcher:
    """A Fetcher serving local files, matching a URL to the file with the same basename."""
    return _DirectoryFetcher(root)


def retrying(
    inner: Fetcher,
    *,
    backoff: tuple[float, ...] = BACKOFF,
    sleep: Callable[[float], None] = time.sleep,
) -> Fetcher:
    """``inner``, asked again after each wait in ``backoff`` when the failure is worth retrying.

    A wrapper rather than a loop inside :func:`http_fetcher`, for three reasons. It matches the
    shape this module already has — small Fetchers behind small factories — so nothing new has to be
    learnt to read it. It keeps :func:`http_fetcher` meaning exactly one HTTP call, which is what a
    caller asking for one should get. And it is drivable by a test with a fetcher that raises on cue
    and a ``sleep`` that records rather than waits, so the policy is checked with no network, no
    stand-in for ``requests`` and no waiting.
    """
    return _RetryingFetcher(inner, backoff, sleep)


def http_fetcher(timeout: float = 60.0) -> Fetcher:
    """The real one, and exactly one HTTP call.

    ``requests`` is imported here so nothing else in the ingest depends on it. Retrying is
    :func:`default_fetcher`'s job rather than this one's.
    """
    import requests

    def fetch(url: str) -> bytes:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.content

    return fetch


def default_fetcher(timeout: float = 60.0) -> Fetcher:
    """What the ingest uses when a caller injects nothing: the real one, retried.

    Every function that reaches upstream lands here — :func:`epl.ingest.football_data.fetch_season`,
    :func:`epl.ingest.football_data.fetch_all`, :func:`epl.ingest.fixtures.fetch_fixtures` and the
    Pundit fetch in :mod:`epl.pundits.myfootballfacts` — so the retry is composed in one place and
    no call site knows about it.
    """
    return retrying(http_fetcher(timeout))
