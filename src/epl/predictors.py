"""The Predictor contract — what every Predictor is handed and what it must hand back.

A Predictor is anything that emits Predictions and can be scored: a model, the Market Line, a
Pundit, or the Naive Baseline (CONTEXT.md). One contract serves all of them, which is what lets the
ledger, the audits and the scoreboard be written once and never special-case a Predictor.

The contract is deliberately *not* "here is an As-Of Instant, go and read what you need". A
Predictor is handed :class:`Evidence`: the rows of the corpus that were already in the past at that
instant, and nothing else. Leaving each Predictor to filter for itself would put the project's one
rule — no future data, ever — in as many places as there are Predictors, and a Predictor that got
it wrong would leak silently and score suspiciously well.

:class:`Evidence` also *records* what it handed over, so a stored Prediction can say which input
rows it saw. That record is what makes the leak check an audit of the evidence rather than a claim
about the code: :func:`epl.ledger.schema.audit` re-reads it from the CSV and fails if any Prediction
saw a row timestamped at or after its own As-Of Instant.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd

from epl.rounds import kickoff_instants

#: A Predictor's name keys the ledger and names its file under ``outputs/backtest/``, so it is a
#: slug: lower case, digits and underscores, starting with a letter.
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class PredictorError(Exception):
    """Something claiming to be a Predictor could not be registered or found."""


class Corpus:
    """Every match a Predictor could ever see, with its kickoffs derived once and sorted by them.

    A backtest cuts the same table at every Prediction Round in turn. Re-deriving 52,672 kickoffs
    at each of 1,189 rounds — parsing a date column and a time column each time — is the difference
    between a walk that takes seconds and one that takes minutes, and sorting by kickoff turns each
    cut into a slice rather than a scan.

    Kickoffs are derived here, never accepted from a caller: a supplied ``kickoff`` column would be
    a way to move a row across an As-Of Instant without any audit noticing.

    Rows therefore reach a Predictor in kickoff order rather than the order they were loaded in.
    """

    def __init__(self, matches: pd.DataFrame) -> None:
        kickoffs = (
            kickoff_instants(matches).to_numpy()
            if len(matches)
            else np.empty(0, dtype="datetime64[ns]")
        )
        order = kickoffs.argsort(kind="stable")
        self._matches = matches.iloc[order]
        self._kickoffs = pd.Series(kickoffs[order])

    def before(
        self, as_of: pd.Timestamp
    ) -> tuple[pd.DataFrame, npt.NDArray[np.datetime64]]:
        """The rows that had kicked off strictly before ``as_of``, and their kickoffs."""
        cut = int(self._kickoffs.searchsorted(as_of, side="left"))
        return self._matches.iloc[:cut], self._kickoffs.to_numpy()[:cut]


class Evidence:
    """Everything a Predictor may see at one As-Of Instant, and a record of what it saw.

    Construct with :meth:`before`, never by hand: the classmethod is what applies the cut, so an
    Evidence holding a row from the future is not a thing that can be built.

    A match row is timestamped at its kickoff. That is earlier than the moment its result became
    known, so it is the loose direction in principle — but every As-Of Instant is a midnight and
    the latest kickoff anywhere in the corpus is 20:15, so no match in this project is still being
    played when an As-Of Instant falls. ``tests/ledger/test_the_corpus.py`` re-derives that from
    the data rather than trusting it.
    """

    def __init__(
        self,
        as_of: datetime | pd.Timestamp,
        visible: pd.DataFrame,
        kickoffs: npt.NDArray[np.datetime64],
    ) -> None:
        self.as_of = pd.Timestamp(as_of)
        self._visible = visible
        self._kickoffs = kickoffs
        self._taken = np.zeros(len(visible), dtype=bool)

    @classmethod
    def before(
        cls, matches: pd.DataFrame | Corpus, as_of: datetime | pd.Timestamp
    ) -> Evidence:
        """The matches that had kicked off strictly before ``as_of``.

        Strictly, not on-or-before: the As-Of Instant is midnight at the start of the anchor day,
        and a Fixture played on that day has not been played when its round is predicted.

        Pass a :class:`Corpus` when cutting the same table many times; a plain frame is prepared
        into one for this single cut.
        """
        instant = pd.Timestamp(as_of)
        corpus = matches if isinstance(matches, Corpus) else Corpus(matches)
        return cls(instant, *corpus.before(instant))

    def matches(self, *, divisions: Sequence[str] | None = None) -> pd.DataFrame:
        """The visible matches, optionally narrowed to some tiers — and recorded as seen."""
        if divisions is None:
            self._taken[:] = True
            return self._visible

        wanted = self._visible["division"].isin(list(divisions)).to_numpy()
        self._taken |= wanted
        return self._visible.loc[wanted]

    @property
    def rows_seen(self) -> int:
        """How many input rows this Predictor was actually handed.

        The union over every read, so a Predictor that asks for one tier and then for the whole
        pyramid is recorded as having seen the pyramid rather than the two added together.
        """
        return int(self._taken.sum())

    @property
    def latest_seen(self) -> pd.Timestamp | None:
        """The kickoff of the most recent input row handed over, or ``None`` if there was none.

        This is the number the leak audit turns on: it must fall strictly before
        :attr:`as_of`, and the cut in :meth:`before` is what guarantees it does.
        """
        if not self._taken.any():
            return None
        return pd.Timestamp(self._kickoffs[self._taken].max())


@runtime_checkable
class Predictor(Protocol):
    """Anything that emits Predictions and can be scored.

    A model, the Market Line, a Pundit or the Naive Baseline — all of them are this, and the ledger
    and the scoreboard know nothing else about any of them (CONTEXT.md).

    Structural rather than inherited on purpose. The Market Line is a column of odds and a Pundit is
    a person with a spreadsheet; requiring either to subclass a model base class would be a lie
    about what they are, and the only thing the rest of the system needs is that they answer
    :meth:`predict`.
    """

    #: The Predictor's slug — see :data:`NAME_PATTERN`.
    name: str

    def predict(
        self, fixtures: pd.DataFrame, evidence: Evidence
    ) -> npt.NDArray[np.float64]:
        """One probability distribution over (Home, Draw, Away) per row of ``fixtures``.

        ``fixtures`` is one Prediction Round's Fixtures, in the order the Predictions must come
        back in. ``evidence`` carries the As-Of Instant and everything visible at it; a Predictor
        that reads anything else is reading the future, which is why the corpus is not passed.

        Return an ``(n, 3)`` array. The ledger validates it with :func:`epl.metrics.as_predictions`,
        so a row that does not sum to one is a bug that stops the run rather than a number that
        reaches the scoreboard.
        """
        ...


#: Three attributes a Predictor **may** declare. They are read through the accessors below rather
#: than named on :class:`Predictor`, because a Protocol member is required of everything that
#: claims the Protocol — adding them there would mean every Predictor, and every test double,
#: had to carry three attributes that almost all of them have no use for.
#:
#: ``covers(fixtures) -> Sequence[bool]``
#:     Which Fixtures this Predictor has anything to say about. Declared by the Predictors whose
#:     input does not span the whole Evaluation Window — the Ceiling Line, whose closing odds
#:     begin in 2019/20, and a Pundit, who published in the Seasons they worked (issue #11).
#:     Absent means it covers everything. See :func:`epl.ledger.schema.covered`.
#:
#: ``also_sees: tuple[str, ...]``
#:     Fixture columns this Predictor claims beyond the ledger's allow-list. Only the columns in
#:     :data:`epl.ledger.schema.PRIVILEGED_FIXTURE_COLUMNS` may be claimed, and today only the
#:     Ceiling Line claims any (ADR 0001).
#:
#: ``note: str``
#:     A caveat that must travel with this Predictor's score wherever it is reported. The Ceiling
#:     Line carries one, because a scoreboard line that did not say it knows team news the model
#:     cannot have would be a misleading number rather than an incomplete one.
OPTIONAL_ATTRIBUTES: tuple[str, ...] = ("covers", "also_sees", "note")


def also_sees(predictor: Predictor) -> tuple[str, ...]:
    """The extra Fixture columns this Predictor claims, as it claims them.

    Unvalidated: what may actually be claimed is the ledger's rule, not the Predictor's, and
    :func:`epl.ledger.schema.visible` is where the claim is checked.
    """
    return tuple(getattr(predictor, "also_sees", ()))


def note(predictor: Predictor | str) -> str:
    """The caveat that must be printed beside this Predictor's score, or ``""`` if it has none.

    Takes a name as well as a Predictor, because the scoreboard scores stored rows and a stored
    row carries only a name. A name nobody has registered has no note, which is the right answer
    for a Predictor whose ledger file outlived its code.
    """
    found = _REGISTRY.get(predictor) if isinstance(predictor, str) else predictor
    return str(getattr(found, "note", "")) if found is not None else ""


#: Registered Predictors, in registration order. Module-level and process-wide: importing
#: ``epl.benchmarks`` is what puts the Naive Baseline on the scoreboard.
_REGISTRY: dict[str, Predictor] = {}


def register[PredictorT: Predictor](predictor: PredictorT) -> PredictorT:
    """Put a Predictor on the scoreboard, and hand it back so it can be named where it is defined.

        NAIVE_BASELINE = register(NaiveBaseline())

    Refuses a duplicate name. Names key the ledger and name the backtest file, so two Predictors
    sharing one would merge two track records into a single line and nothing would look wrong.
    """
    name = getattr(predictor, "name", None)
    if not isinstance(name, str) or not NAME_PATTERN.match(name):
        raise PredictorError(
            f"a Predictor's name must match {NAME_PATTERN.pattern}; got {name!r}"
        )
    if name in _REGISTRY:
        raise PredictorError(f"a Predictor named {name!r} is already registered")
    _REGISTRY[name] = predictor
    return predictor


def registered() -> tuple[Predictor, ...]:
    """Every registered Predictor, in registration order."""
    return tuple(_REGISTRY.values())


def by_name(name: str) -> Predictor:
    """The registered Predictor with this name, or a :class:`PredictorError` naming the others."""
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "nothing"
        raise PredictorError(f"no Predictor named {name!r}; registered: {known}") from None
