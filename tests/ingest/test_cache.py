"""The raw cache's write rule, tested where it lives rather than through one ingester.

Two sources cache through :mod:`epl.ingest.cache` now — Football-Data's season files and the
MyFootballFacts pundit pages — so the rule that a refresh archives rather than overwrites is
checked once, here, instead of once per caller.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from epl.ingest import cache


class TestStore:
    def test_writes_content_and_creates_the_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "down" / "E0.csv"

        assert cache.store(path, b"first") == path
        assert path.read_bytes() == b"first"

    def test_identical_bytes_archive_nothing(self, tmp_path: Path) -> None:
        """Re-fetching an unchanged file is the ordinary case, and must leave no trace."""
        path = tmp_path / "E0.csv"
        cache.store(path, b"same")

        cache.store(path, b"same")

        assert not cache.superseded_dir(path).exists()

    def test_different_bytes_archive_the_old_copy(self, tmp_path: Path) -> None:
        """Upstream backfills into rows it has already published. The bytes a Sealed Prediction
        was made from are the only record of what it was made from, so they are kept."""
        path = tmp_path / "E0.csv"
        cache.store(path, b"as published in March")

        cache.store(path, b"as backfilled in May")

        archived = sorted(cache.superseded_dir(path).glob("E0_*.csv"))
        assert len(archived) == 1
        assert archived[0].read_bytes() == b"as published in March"
        assert path.read_bytes() == b"as backfilled in May"

    def test_the_archive_is_stamped_with_when_the_old_copy_was_fetched(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "E0.csv"
        cache.store(path, b"old")
        os.utime(path, (time.time(), 1_700_000_000))  # 2023-11-14T22:13:20Z

        archived = cache.supersede(path, b"new")

        assert archived is not None
        assert archived.name == "E0_20231114T221320Z.csv"

    def test_supersede_reports_nothing_archived_when_there_was_no_file(
        self, tmp_path: Path
    ) -> None:
        assert cache.supersede(tmp_path / "absent.csv", b"anything") is None
