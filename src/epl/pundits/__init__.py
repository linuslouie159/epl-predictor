"""Pundits: the backfill, the two gradings, and the Pundits registered as Predictors.

Built by issue #11. Issue #12 adds the Calibrated Pundit and issue #16 the BBC live spike.

A Pundit publishes a Scoreline, not a distribution. Scoring that as `[1, 0, 0]` charges 1.00 RPS for
calling Home when Away happens — punishing a claim of certainty the Pundit never made, and putting
them at ~0.36 against a market at ~0.19 in a gap that mostly measures the format of the question.
That would fail this project's own honesty bar.

So two distinct Predictors are registered per Pundit (ADR 0003). **Pundit** is the raw Scoreline
scored as-stated, and is built here. **Calibrated Pundit** maps the Scoreline — bucketed by
predicted goal margin, since a 3-0 call is a stronger claim than 2-1 — onto the Outcome frequencies
that call has historically produced, fitted walk-forward on past calls only, and is issue #12. The
headline three-way comparison uses the calibrated form; the as-stated number is published beside it
as the cost of stating certainty.

**Calibrated Pundit is a one-feature model, not a person.** It may beat our own models, which is a
real finding about the information in pundit calls — but it must never be presented as "Sutton beat
the model". The naming keeps that distinction visible in code and in output. For the same reason
there are two named Pundits here rather than one anonymous slot: Mark Lawrenson worked 2017/18 to
2021/22 and Chris Sutton has worked 2022/23 onward, and a single averaged line would be a Predictor
that is nobody.

Store only the facts — Fixture, predicted Scoreline, Predictor, date — never the prose, and
attribute BBC as the origin. `predictions.csv` beside this module is that dataset, frozen and
committed so the accountability feature is backtestable on day one rather than in a year, and so a
scoreboard run does not depend on nine pages of someone else's HTML still being up.

Four modules:

* :mod:`epl.pundits.myfootballfacts` — the nine archive pages, fetched and parsed
* :mod:`epl.pundits.dataset` — resolved to Clubs, reconciled with the corpus, and frozen
* :mod:`epl.pundits.grading` — the exact-score and correct-Outcome readings
* :mod:`epl.pundits.predictor` — the two Pundits, registered and scored as-stated
"""

from epl.pundits.dataset import (
    CALL_COLUMNS,
    FIXTURE_KEY,
    Backfill,
    PunditDatasetError,
    build,
    load,
    outcomes_of,
)
from epl.pundits.grading import GRADE_COLUMNS, SUMMARY_COLUMNS, grade, summary
from epl.pundits.myfootballfacts import ORIGIN, PAGES, SOURCE, Page, PunditSourceError
from epl.pundits.predictor import LAWRENSON, SUTTON, Pundit, PunditError

__all__ = [
    "CALL_COLUMNS",
    "FIXTURE_KEY",
    "GRADE_COLUMNS",
    "LAWRENSON",
    "ORIGIN",
    "PAGES",
    "SOURCE",
    "SUMMARY_COLUMNS",
    "SUTTON",
    "Backfill",
    "Page",
    "Pundit",
    "PunditDatasetError",
    "PunditError",
    "PunditSourceError",
    "build",
    "grade",
    "load",
    "outcomes_of",
    "summary",
]
