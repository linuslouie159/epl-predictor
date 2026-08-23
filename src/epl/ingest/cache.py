"""The raw cache's one write rule: refreshing archives, it never overwrites.

`data/raw/` is a byte-identical copy of what upstream published, and CLAUDE.md forbids editing it.
That is only half a guarantee. Upstream files *change* — Football-Data backfills results and odds
into rows it has already published, and MyFootballFacts adds matchdays to a season page as the
season runs — so a refresh that simply wrote the new bytes over the old ones would destroy the only
record of what the source said at the moment a Sealed Prediction was made.

So a refresh that brings different bytes moves the cached copy into ``superseded/`` first, stamped
with the time it was fetched. Named for what ADR 0005 says to do with a Prediction that turns out
wrong: supersede it, never rewrite it. The same rule applies to the bytes a Prediction was made
from, which is why it lives here rather than inside one ingester — two sources now cache through
it, and a second copy of this rule is a second chance to get it wrong.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path


def superseded_dir(path: Path) -> Path:
    """Where earlier copies of ``path`` go when a refresh brings different bytes."""
    return path.parent / "superseded"


def supersede(path: Path, content: bytes) -> Path | None:
    """Move ``path`` aside if ``content`` would change it. Returns the archived path, if any.

    Identical bytes archive nothing: re-fetching an unchanged file is the ordinary case, and a
    ``superseded/`` directory that filled up with copies of one file would bury the refreshes that
    actually changed something.
    """
    if not path.exists() or path.read_bytes() == content:
        return None

    fetched_at = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.UTC)
    archive = superseded_dir(path) / f"{path.stem}_{fetched_at:%Y%m%dT%H%M%SZ}{path.suffix}"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(path.read_bytes())
    return archive


def store(path: Path, content: bytes) -> Path:
    """Write ``content`` to ``path`` in the raw cache, archiving whatever was there first."""
    supersede(path, content)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path
