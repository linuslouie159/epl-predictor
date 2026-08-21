"""Where things live on disk.

One module owns the layout so that no other module hard-codes a relative path and quietly breaks
when called from a different working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_VAR = "EPL_PROJECT_ROOT"


def project_root() -> Path:
    """The repository root.

    Resolved from this file's location, which holds for the editable install described in
    ADR 0009. ``EPL_PROJECT_ROOT`` overrides it, which is how tests point the whole layout at a
    temporary directory.
    """
    override = os.environ.get(_ENV_VAR)
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2]


def raw_dir() -> Path:
    """Byte-identical cached downloads. Never hand-edited (see CLAUDE.md)."""
    return project_root() / "data" / "raw"


def processed_dir() -> Path:
    """Cleaned tables, derived from the raw cache and regenerable from it."""
    return project_root() / "data" / "processed"


def backtest_dir() -> Path:
    """Regenerable Backtest Predictions. Deletable at will (ADR 0005)."""
    return project_root() / "outputs" / "backtest"


def live_dir() -> Path:
    """Sealed Predictions. Append-only, committed, never rewritten (ADR 0005)."""
    return project_root() / "outputs" / "live"
