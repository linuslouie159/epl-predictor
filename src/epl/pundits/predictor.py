"""The Pundits registered as Predictors, scored exactly as they published.

A Pundit publishes a Scoreline. Read as a Prediction that is ``[1, 0, 0]`` — total certainty — and
RPS charges 1.00 for calling Home when Away happens. That is not a fair reading of what anybody
said, and ADR 0003 is explicit that it "would fail our own honesty bar" as a headline. It is
published anyway, beside the fair reading the Calibrated Pundit gives
(:mod:`epl.pundits.calibrated`), because the *difference* between the two is the number worth
having: the cost of stating certainty. Measured, that difference is 0.12 RPS.

So the as-stated score is a real measurement of a deliberately unfair question, and its
:attr:`Pundit.note` says so on the scoreboard, every time, wherever the number is reported.

**Two named Pundits, not one pundit slot.** A Pundit is "a *named* public forecaster" (CONTEXT.md)
and ADR 0003 spends its Consequences section on keeping a person distinguishable from a model
built out of their calls. Mark Lawrenson worked 2017/18-2021/22 and Chris Sutton 2022/23-2025/26;
one line averaging the two would be a Predictor that is nobody, and neither of them could be held
to it. Each therefore covers the Seasons they worked and no others — which is what ``covers`` on
the Predictor contract is for (issue #8 added it for the Ceiling Line; this is its second user).

**A Pundit reads no history.** Its call is a fact about the Fixture, published before it, so it
never touches its Evidence and every stored row records ``inputs_seen = 0`` with no
``latest_input``. Correct, and the same thing the Market Line records: a Predictor that consumes
no history has no history to leak.

**An uncovered Fixture is an error, not a fallback.** Twelve of the 3,420 Fixtures in these nine
Seasons have no call — the archive simply never listed them. Quoting a base rate for those would
put a Prediction the Pundit never made onto their track record, indistinguishable from one they
did, and ``covers`` exists precisely so it never comes to that.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from epl.metrics import OUTCOMES
from epl.predictors import Evidence, register
from epl.pundits import dataset
from epl.pundits.dataset import fixture_keys, outcomes_of
from epl.pundits.myfootballfacts import ORIGIN, PAGES
from epl.windows import season_label

#: What every Pundit's note has to say, whatever else it says. Spelled once so two Pundits cannot
#: drift into carrying different caveats about the same reading.
#:
#: Three things have to travel with an as-stated RPS wherever it is reported, and all three are
#: here rather than in a docstring nobody reading the scoreboard will open: who published the calls
#: (issue #11 asks that the BBC be attributed as the origin), that this Pundit's Fixtures are a
#: fraction of the board's so the number is not comparable to a full-window one — the Ceiling
#: Line's lesson, ADR 0001 — and what reading the number is (ADR 0003).
AS_STATED_CAVEAT = (
    "as-stated: a published Scoreline read as [1, 0, 0], a claim of certainty no Pundit made, so "
    "this RPS measures the format of the question as much as the answer (ADR 0003). The fair "
    "reading is `{fair}` beside it, and the gap between the two is the cost of stating certainty"
)

#: What a Calibrated Pundit's name is built out of. That Predictor is named for the **map** rather
#: than for the forecaster, which is what stops any row of any output from reading as a person
#: (ADR 0003).
#:
#: Here rather than in :mod:`epl.pundits.calibrated`, which is where the Calibrated Pundit itself
#: lives, because both halves of the pair need it and this half exists first — a Calibrated Pundit
#: is built *from* a Pundit, so it may import this module and this module may not import it. Two
#: copies of the prefix is how a note comes to point at a Predictor nobody registered.
NAME_PREFIX = "margin_map_"


def map_name(pundit: str) -> str:
    """The Calibrated Pundit built from this Pundit's calls, by name."""
    return f"{NAME_PREFIX}{pundit}"


class PunditError(Exception):
    """A Pundit was asked about a Fixture they never published a call on."""


class Pundit:
    """One named forecaster's published Scorelines, scored as-stated.

    Stateless once loaded: the calls are the frozen dataset and nothing here accumulates, so one
    instance serves the whole scoreboard.
    """

    def __init__(
        self,
        name: str,
        display_name: str,
        *,
        note: str = "",
        calls: pd.DataFrame | None = None,
    ) -> None:
        self.name = name
        self.display_name = display_name
        self.note = note
        self._calls = calls
        self._one_hot: dict[tuple[int, str, str], npt.NDArray[np.float64]] | None = None

    @property
    def calls(self) -> pd.DataFrame:
        """This Pundit's rows of the frozen dataset.

        Read on first use rather than at import. ``python -m epl.pundits build`` is what *writes*
        that file, and it imports this module to do it — a Pundit that loaded eagerly could never
        be the thing that builds its own dataset.
        """
        if self._calls is None:
            everyone = dataset.load()
            self._calls = everyone.loc[everyone["pundit"] == self.name].reset_index(drop=True)
        return self._calls

    def covers(self, fixtures: pd.DataFrame) -> npt.NDArray[np.bool_]:
        """Which Fixtures this Pundit published a call on."""
        return np.array([key in self._lookup() for key in fixture_keys(fixtures)], dtype=bool)

    def predict(
        self, fixtures: pd.DataFrame, evidence: Evidence
    ) -> npt.NDArray[np.float64]:
        """Each call as the distribution it literally claims — and nothing from ``evidence``.

        Reshaped rather than left to numpy, so an empty round comes back as ``(0, 3)`` rather than
        as ``(0,)``. The backfill filters by :meth:`covers` first and never asks for one, but the
        live loop at issue #17 has no such guarantee, and a Prediction of the wrong shape would
        fail somewhere further away than here.
        """
        lookup = self._lookup()
        keys = fixture_keys(fixtures)
        uncovered = [key for key in keys if key not in lookup]
        if uncovered:
            season, home, away = uncovered[0]
            raise PunditError(
                f"{self.display_name} published no call on {len(uncovered)} of these Fixtures — "
                f"{home} v {away} in {season_label(season)} — and quoting one would put a "
                "Prediction they never made onto their record. `covers` is what keeps such a "
                "Fixture off the walk"
            )
        return np.asarray([lookup[key] for key in keys], dtype=np.float64).reshape(
            -1, len(OUTCOMES)
        )

    def _lookup(self) -> dict[tuple[int, str, str], npt.NDArray[np.float64]]:
        """Every call as a one-hot Prediction, keyed by Fixture. Built once per Pundit.

        A dictionary rather than a merge because the ledger walks a Pundit over 952 Prediction
        Rounds and asks twice at each — once for ``covers`` and once for ``predict``.
        """
        if self._one_hot is None:
            calls = self.calls
            index = {outcome: position for position, outcome in enumerate(OUTCOMES)}
            certainty = np.eye(len(OUTCOMES), dtype=np.float64)
            self._one_hot = {
                key: certainty[index[outcome]]
                for key, outcome in zip(
                    fixture_keys(calls),
                    outcomes_of(calls["pred_home_goals"], calls["pred_away_goals"]),
                    strict=True,
                )
            }
        return self._one_hot


def a_pundit(name: str, display_name: str) -> Pundit:
    """One named forecaster, with the Seasons they worked read off the pages rather than retyped.

    :data:`epl.pundits.myfootballfacts.PAGES` already says who published when, so the span in the
    note is derived from it. Written out here it would be a third copy of that fact — after the
    pages and the frozen dataset — and the one nothing would catch drifting.
    """
    seasons = [page.season for page in PAGES if page.pundit == name]
    span = f"{season_label(min(seasons))}-{season_label(max(seasons))}"
    return Pundit(
        name,
        display_name,
        note=(
            f"{display_name} for the {ORIGIN}, {span}, archived by MyFootballFacts; "
            f"measured over the Fixtures they called, so not comparable to a full-window RPS. "
            f"{AS_STATED_CAVEAT.format(fair=map_name(name))}"
        ),
    )


#: Mark Lawrenson's BBC column, as MyFootballFacts archived it. 1,896 calls over five Seasons.
LAWRENSON = register(a_pundit("lawrenson", "Mark Lawrenson"))

#: Chris Sutton's, from 2022/23. 1,512 calls over four Seasons.
SUTTON = register(a_pundit("sutton", "Chris Sutton"))
