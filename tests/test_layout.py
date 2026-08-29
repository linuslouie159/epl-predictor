"""The package layout the README documents.

Ticket 2: the modules must be "present and independently importable". Independently matters — a
module that only imports as a side effect of another one is not a seam, and the per-module seam
decision is what lets each stage be built and tested on its own.
"""

from __future__ import annotations

import importlib

import pytest

#: The layout from README.md, in build order.
MODULES = (
    "epl.ingest",
    "epl.clubs",
    "epl.metrics",
    "epl.models",
    "epl.benchmarks",
    "epl.pundits",
    "epl.simulate",
    "epl.ledger",
    "epl.live",
    # The Telegram bot (issue #20). Here rather than under `epl.live` because it is not part of the
    # loop and must never become part of it: it reports on both stores and can write to neither
    # (ADR 0005), which `tests/bot/test_the_bot_is_read_only.py` checks structurally.
    "epl.bot",
    # In the README's layout and in no build order: the deferred features, written down rather than
    # built (decision 12, issue #18). It is here because the README documents it and this list is
    # what keeps the two in step — and because "importing it costs nothing" is a claim worth
    # checking in a fresh interpreter like any other. What it must *not* be is imported by the
    # pipeline, and tests/v2/test_stubs_are_unreachable.py is what checks that.
    "epl.v2",
)

#: Single-file modules that every stage imports rather than a stage of their own: three scope which
#: data a Prediction may see, one owns the on-disk layout. Ticket 5 puts Prediction Rounds here,
#: beside the Windows; ticket 7 puts the Predictor contract here for the same reason — every stage
#: from Elo onward implements it, and it belongs inside none of them. Ticket 10 puts the shared
#: calibration layer here too: it wraps every Predictor, so it belongs to none of them either.
FOUNDATIONS = (
    "epl.windows",
    "epl.rounds",
    "epl.paths",
    "epl.predictors",
    "epl.calibration",
)

#: The modules that are still documented shells. Each is removed from here as its stage is built:
#: ingest and clubs by tickets 2-4, metrics by ticket 6, the ledger and the Naive Baseline half of
#: benchmarks by ticket 7, the Market Line and Ceiling Line half of it by ticket 8, Elo through an
#: ordered logit by ticket 9, the Pundit backfill by ticket 11, and `epl.simulate` by ticket 14 —
#: which built the posterior half of it. The Monte Carlo projection over those draws is ticket 15,
#: but the module is no longer a shell, so the list is empty and these tests no longer collect.
UNBUILT: tuple[str, ...] = ()


@pytest.mark.parametrize("name", MODULES)
def test_each_module_imports(name: str) -> None:
    assert importlib.import_module(name) is not None


@pytest.mark.parametrize("name", MODULES)
def test_each_module_imports_in_a_fresh_interpreter(name: str) -> None:
    """Independently importable: no module may rely on another having been imported first."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", f"import {name}"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", MODULES)
def test_each_module_says_what_it_is_for(name: str) -> None:
    """A module with no docstring is a checkbox; one that explains itself is a handover note."""
    module = importlib.import_module(name)
    assert module.__doc__ is not None
    summary = module.__doc__.strip().splitlines()[0]
    assert summary.endswith("."), f"{name}: docstring should open with a full sentence"


@pytest.mark.parametrize("name", UNBUILT)
def test_each_unbuilt_module_records_what_it_will_hold(name: str) -> None:
    """The shells exist to satisfy the layout, so they must be worth more than an empty file.

    Each names the issues that build it and the decisions that constrain it, so whoever picks the
    stage up starts from the reasoning rather than rediscovering it.
    """
    module = importlib.import_module(name)
    assert "#" in module.__doc__, f"{name}: should name the issue that builds it"
    assert len(module.__doc__) > 500, f"{name}: too thin to hand over"


def test_the_top_level_package_imports() -> None:
    import epl

    assert epl.__version__


@pytest.mark.parametrize("name", FOUNDATIONS)
def test_each_foundation_module_imports_and_says_what_it_is_for(name: str) -> None:
    """Ticket 5 puts Prediction Rounds beside the Windows: both scope which data a Prediction may
    see, both are imported by every later stage, and neither belongs inside one of them."""
    module = importlib.import_module(name)
    assert module.__doc__ is not None
    assert module.__doc__.strip().splitlines()[0].endswith(".")
