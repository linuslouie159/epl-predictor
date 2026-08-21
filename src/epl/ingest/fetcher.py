"""How the ingest reaches upstream — and the seam that lets tests not reach it at all.

A Fetcher is anything that turns a URL into bytes. Every function that pulls from Football-Data
takes one, so this module is the only place in the ingest that knows HTTP exists. Tests inject
:func:`mapping_fetcher` or :func:`directory_fetcher` and run with no network access, without
patching a module global — patching one would leave the production path untested and would break
silently the moment the call site moved.

Both test fetchers **raise** on a URL they were not given, rather than returning empty bytes. An
empty response parses as a Season with no matches, which looks like a data gap rather than a broken
test.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol


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


def mapping_fetcher(responses: Mapping[str, bytes]) -> _MappingFetcher:
    """A Fetcher serving canned bytes per URL. Raises on anything it was not given."""
    return _MappingFetcher(responses)


def directory_fetcher(root: Path | str) -> _DirectoryFetcher:
    """A Fetcher serving local files, matching a URL to the file with the same basename."""
    return _DirectoryFetcher(root)


def http_fetcher(timeout: float = 60.0) -> Fetcher:
    """The real one. ``requests`` is imported here so nothing else in the ingest depends on it."""
    import requests

    def fetch(url: str) -> bytes:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.content

    return fetch


def default_fetcher(timeout: float = 60.0) -> Fetcher:
    """What the ingest uses when a caller injects nothing."""
    return http_fetcher(timeout)
