"""Features deferred out of v1, written down instead of built (decision 12, issue #18).

Nothing in this package runs. Nothing in this package is imported by anything else in ``epl``, and
``tests/v2/test_stubs_are_unreachable.py`` is what keeps that true — deleting the whole directory
would break no import and change no number on any scoreboard.

**Why a package of prose rather than a section of the README.** A deferred feature is not a wish
list entry; it is a decision with a reason and a price, and both are perishable. The README says
what v1 is. These modules say what each deferred feature *would do*, what it *would need*, and —
the part that actually saves a future contributor a week — what would have to be true before
picking it up is a good idea. Written beside the code they would join, they are found by someone
reading ``src/epl`` rather than by someone who thought to search the docs.

Each stub carries its entry price as a ``WHAT_IT_NEEDS`` tuple. That is a named constant rather
than a closing paragraph because a paragraph can lose its last sentence in an edit and still read
fine, and the sentence it loses is the one the ticket asked for (criterion 4).

**These are not a roadmap.** Nothing here is scheduled, and two of the three may never be worth
building. What they are is a record of reasoning, so that the next person to consider an ML layer
reaches the same conclusion faster, or reaches a different one knowing what this one rested on.

================================  =========================================================
:mod:`epl.v2.xgboost_layer`       a learned Outcome model beside Elo and Dixon-Coles
:mod:`epl.v2.golden_boot`         per-player Season goal distributions, and a top scorer
:mod:`epl.v2.api_football`        a paid Fixtures API and the free file meant to replace it
================================  =========================================================

The third is the live one. Its stated reason for deferral — that ``fixtures.csv`` already carries
upcoming Fixtures with the Market Line — is the premise stage 13 measured and could not confirm,
so that module records the measurement rather than the assumption.
"""

from __future__ import annotations

#: The stub modules, named rather than imported: this package must stay free to import.
#:
#: Naming them as strings is not pedantry. If this file imported them, ``import epl.v2`` would
#: execute all three, and "nothing in the pipeline executes a stub" would depend on nothing in the
#: pipeline importing the *package* either — a weaker claim, and a harder one to check.
STUBS: tuple[str, ...] = ("xgboost_layer", "golden_boot", "api_football")

__all__ = ["STUBS"]
