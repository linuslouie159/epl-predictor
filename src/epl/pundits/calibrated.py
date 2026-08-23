"""The Calibrated Pundit: a Pundit's own margin map, fitted walk-forward and quoted as a Predictor.

This is the fair reading of a published Scoreline, and the headline half of ADR 0003. A Pundit is
scored as-stated — `[1, 0, 0]`, a claim of certainty nobody made — because the *difference* between
the two readings is the number worth having. This module is the other side of that difference.

**It is a model, not a person, and it is named as one.** A Pundit registers as `lawrenson`; the
Predictor built out of their calls registers as `margin_map_lawrenson`, and its `note` says in
as many words that it is not Mark Lawrenson. ADR 0003 spends its whole Consequences section on
this: a Calibrated Pundit may beat Elo, which is a real finding about the information in pundit
calls, and "Sutton beat the model" is a sentence no output of this project may support. The name on
the scoreboard is the artifact a reader actually receives, so the distinction lives there rather
than in a docstring.

**One map per Pundit, never one shared between them.** A map is a statement about one forecaster's
own calls — "when *this* Pundit says 3-0, how often is it a home win?" — and pooling two people's
calls would correct each of them with the other's habits, which is the same reasoning
:func:`epl.ledger.scoreboard.calibrated_predictions` gives for fitting the shared layer per
Predictor. It costs the start of each record separately, and that cost is visible on the board.

## What makes this a Predictor rather than a scoring step

:mod:`epl.calibration` cannot be a Predictor: it is fitted on the Outcomes of the very Predictions
it corrects, so it runs at scoring time and stores nothing (ADR 0006). This is different in exactly
one way that changes everything. Its map is fitted on the Outcomes of matches that had **already
kicked off** at the As-Of Instant, and its input — the Scoreline — was published before it. So on
any given Friday you really can compute what a 3-0 from this Pundit has been worth and quote it for
Saturday, which means a Calibrated Pundit is a genuine forecaster: it goes through the ledger, it
seals rows, and it audits like everything else.

That is why it reads its history through :class:`epl.predictors.Evidence` rather than off the
corpus. Every stored row carries `inputs_seen` and `latest_input`, so the walk-forward claim in
issue #12's second acceptance criterion is checkable off the file months later rather than asserted
by a test. A Pundit records `inputs_seen = 0` because it consumes no history; this one does not,
and the difference between those two rows is the whole of what the map adds.

**It refits at every Prediction Round rather than folding one map forward**, for the reason
:class:`epl.models.elo.Elo` gives at greater cost: a map carried between calls would have to decide
whether the Evidence it has just been handed extends the one it fitted last time, and getting that
wrong is the one kind of bug this project cannot see — the rates would be built from the wrong
matches while every stored row still audited clean.

## What it does not cover

A Fixture the Pundit never called, exactly as for the Pundit itself. And the opening calls of every
record, because :data:`epl.pundits.margin.MINIMUM_SAMPLE` past calls have to exist before a map is
fitted at all. Both are refused rather than filled in: "a made-up Prediction that scores is worse
than an absent one" (:mod:`epl.pundits.predictor`), and the scoreboard's `fixtures` column is where
the cost shows up.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from epl.metrics import OUTCOMES
from epl.predictors import Evidence, register
from epl.pundits import margin
from epl.pundits.dataset import DIVISION, FIXTURE_KEY, fixture_keys
from epl.pundits.margin import MarginMap, margins_of
from epl.pundits.predictor import LAWRENSON, SUTTON, Pundit, PunditError, map_name
from epl.rounds import as_of_instant

#: What every Calibrated Pundit's note has to say, whatever else it says. Spelled once so two of
#: them cannot drift into carrying different caveats about the same kind of reading.
#:
#: Three things ride onto the scoreboard with this number. That it is a model and whose calls it
#: was built from — ADR 0003's Consequences section, in the one place a reader of the board will
#: see it. That its Fixtures are a fraction of the board's, so the RPS is not comparable to a
#: full-window one (the Ceiling Line's lesson, ADR 0001). And where the as-stated reading is, since
#: the gap between the two is the deliverable rather than either number alone.
NOT_A_PERSON = (
    "a one-feature model fitted on {display}'s published Scorelines — not {display}. It quotes the "
    "Outcome frequencies a call of that predicted goal margin has historically produced, fitted "
    "walk-forward on their past calls only. It may beat a model on this board, which would be a "
    "finding about the information in the calls and never a verdict on the forecaster (ADR 0003)"
)



class CalibratedPundit:
    """One Pundit's calls, read through their own margin map (CONTEXT.md).

    Stateless between rounds: the map is refitted from the Evidence it is handed every time, so one
    instance serves the whole scoreboard and no round can be corrected by another round's sample.

    ``pundit`` is public because the pairing is the point — every number this Predictor produces is
    meant to be read against the as-stated one beside it, and the gap between them is what ADR 0003
    calls the cost of stating certainty.
    """

    def __init__(
        self,
        pundit: Pundit,
        *,
        name: str,
        display_name: str,
        note: str = "",
        minimum: int = margin.MINIMUM_SAMPLE,
    ) -> None:
        self.pundit = pundit
        self.name = name
        self.display_name = display_name
        self.note = note
        self.minimum = minimum
        self._margins: dict[tuple[int, str, str], int] | None = None

    def covers(self, fixtures: pd.DataFrame) -> npt.NDArray[np.bool_]:
        """Which Fixtures this map has both a call for and a sample behind.

        Two conditions, and the second is what a Pundit does not have. A map needs
        :attr:`minimum` past calls before it is fitted at all, so the opening weeks of a record are
        not covered — by this Predictor, while the Pundit beside it covers them and is scored on
        them. The two therefore run over slightly different slates, which is exactly why the
        three-way comparison cuts every Predictor to the Fixtures they all cover
        (:mod:`epl.pundits.report`).

        The count is over call *dates* rather than over the kickoffs :meth:`map_at` actually fits
        on, and on this corpus the two agree exactly. An As-Of Instant is always midnight
        (`epl.rounds`), so a call dated on or after one is a call on a Fixture that had not kicked
        off, whether or not a kickoff time was recorded; and every call in the frozen dataset was
        located against a played match, so every one of them reaches the sample
        (:func:`epl.pundits.dataset._locate`). A call on a Fixture that was never played would
        break the second half of that and make this an over-count — which is a problem for the
        live spike at issue #16 rather than for the backfill, and
        `tests/pundits/test_calibrated_over_the_corpus.py` is what would notice.
        """
        called = self.pundit.covers(fixtures)
        if not len(fixtures):
            return called

        instants = pd.to_datetime(
            pd.to_datetime(fixtures["date"]).dt.date.map(as_of_instant)
        ).to_numpy()
        published = np.sort(pd.to_datetime(self.pundit.calls["date"]).to_numpy())
        behind = np.searchsorted(published, instants, side="left")
        return np.asarray(called & (behind >= self.minimum), dtype=bool)

    def map_at(self, evidence: Evidence) -> MarginMap:
        """The map as it stood at this Evidence's As-Of Instant.

        Public for the reason :meth:`epl.models.elo.Elo.ratings_at` is: this is the model's own
        view of history, and the only honest way to ask what a 3-0 from this Pundit was worth at
        some moment is to ask it through the same cut a Prediction would have been made under.

        The join is to what Evidence holds rather than to the corpus, so a call whose Fixture had
        not kicked off cannot reach the sample — which is the walk-forward guarantee, applied by
        construction rather than by a filter that could be written wrong.
        """
        played = evidence.matches(divisions=(DIVISION,))
        past = self.pundit.calls.merge(
            played[[*FIXTURE_KEY, "outcome"]], on=list(FIXTURE_KEY), how="inner", validate="1:1"
        )
        return margin.fit(
            margins_of(past["pred_home_goals"], past["pred_away_goals"]),
            past["outcome"],
            minimum=self.minimum,
        )

    def predict(
        self, fixtures: pd.DataFrame, evidence: Evidence
    ) -> npt.NDArray[np.float64]:
        """Each call quoted at what that predicted goal margin has been worth, and nothing else.

        Reshaped rather than left to numpy, so a round this Pundit called nothing in comes back as
        ``(0, 3)`` rather than as ``(0,)`` — the backfill filters by :meth:`covers` first and never
        asks for one, but the live loop at issue #17 has no such guarantee.
        """
        if not len(fixtures):
            return np.empty((0, len(OUTCOMES)), dtype=np.float64)
        return self.map_at(evidence).quote(self._margins_for(fixtures))

    def _margins_for(self, fixtures: pd.DataFrame) -> list[int]:
        """The predicted goal margin of each Fixture's call, refusing any Fixture with none.

        An uncovered Fixture is an error, not a fallback — the same rule as
        :meth:`epl.pundits.predictor.Pundit.predict`, and for the same reason. A Prediction quoted
        for a Fixture nobody called would land on this Predictor's record indistinguishable from
        one that came from a call.
        """
        lookup = self._by_fixture()
        keys = fixture_keys(fixtures)
        uncovered = [key for key in keys if key not in lookup]
        if uncovered:
            season, home, away = uncovered[0]
            raise PunditError(
                f"{self.display_name} rests on {self.pundit.display_name}'s calls, and there is "
                f"no call on {len(uncovered)} of these Fixtures — {home} v {away} in {season}. "
                "`covers` is what keeps such a Fixture off the walk"
            )
        return [lookup[key] for key in keys]

    def _by_fixture(self) -> dict[tuple[int, str, str], int]:
        """Every call's predicted goal margin, keyed by Fixture. Built once per Predictor.

        A dictionary rather than a merge because the ledger walks this over hundreds of Prediction
        Rounds and asks twice at each — once for ``covers`` and once for ``predict``.
        """
        if self._margins is None:
            calls = self.pundit.calls
            self._margins = dict(
                zip(
                    fixture_keys(calls),
                    margins_of(calls["pred_home_goals"], calls["pred_away_goals"]).tolist(),
                    strict=True,
                )
            )
        return self._margins


def a_calibrated_pundit(pundit: Pundit) -> CalibratedPundit:
    """The margin map built from one Pundit's calls, named and captioned so it cannot read as them.

    The name and the note are both derived from the Pundit rather than retyped. Written out they
    would be a second copy of who worked when — after the pages, the frozen dataset and the
    Pundit's own note — and the one nothing would catch drifting.
    """
    display = pundit.display_name
    return CalibratedPundit(
        pundit,
        name=map_name(pundit.name),
        display_name=f"Margin Map ({display.split()[-1]}'s calls)",
        note=(
            f"{NOT_A_PERSON.format(display=display)}. Measured over the Fixtures it covers, which "
            f"are neither the board's nor quite {display}'s, so not comparable to a full-window "
            f"RPS. The as-stated reading beside it is `{pundit.name}`, and the gap between the two "
            "is the cost of stating certainty"
        ),
    )


#: The map fitted on Mark Lawrenson's 1,896 calls. Not Mark Lawrenson — see :data:`NOT_A_PERSON`.
MARGIN_MAP_LAWRENSON = register(a_calibrated_pundit(LAWRENSON))

#: The map fitted on Chris Sutton's 1,512. Not Chris Sutton.
MARGIN_MAP_SUTTON = register(a_calibrated_pundit(SUTTON))
