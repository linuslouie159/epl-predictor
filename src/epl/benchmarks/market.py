"""The Market Line and the Ceiling Line — the market registered as a Predictor.

The **Market Line** is the opponent this project is measured against: the vig-removed
market-average *pre-match* book, `BbAv*` spliced to `Avg*`, one continuous series from 2005/06
(ADR 0001; CONTEXT.md defines the term). It scores ~0.1936 RPS over the Evaluation Window.

The **Ceiling Line** is the identical treatment of the market-average *closing* book, available
from 2019/20. A reference upper bound, never the headline opponent, because it knows team news the
model and the Pundits cannot have (ADR 0001).

They are one class and two instances rather than two classes, because they differ in exactly one
thing — which book they read — and writing them separately would invite the two to drift into
being scored differently. That is also the shape the ticket asks for: the vig removal sits behind
one interface so "a reader can see for themselves that the choice barely matters".

Three things about these two are easy to mistake for bugs:

* **Neither reads the corpus.** Both read the book off the Fixture, so both record
  ``inputs_seen = 0`` and an empty ``latest_input``. That is correct — a Predictor that consumes
  no history has no history to leak, and a Pundit will record the same.
* **The Ceiling Line sees a column the ledger withholds from everything else.** It claims the
  closing odds by name in :data:`CLOSING_COLUMNS` below, and the ledger grants the claim only
  because :data:`epl.ledger.schema.PRIVILEGED_FIXTURE_COLUMNS` permits those three columns and no
  others. The exception is deliberate, bounded and visible in both places.
* **The Ceiling Line's RPS is not comparable to the Market Line's.** It is measured over the
  2,660 Fixtures from 2019/20 onward against the other's 7,980, and over that shorter, harder span
  the Market Line scores 0.1981 rather than 0.1936. The Ceiling Line's 0.1968 beats it — by 0.0013
  RPS, which is what a few hours of team news is worth. Its :attr:`note` carries that caveat onto
  the scoreboard, because a bare number beside the Market Line's would read as the opposite of
  what it means.

Seasons 2000/01 and 2001/02 carry no odds at all, and 2002/03-2004/05 carry only Bet365's, so
neither line covers them (ADR 0001). Those Seasons have no market comparison rather than a market
comparison of zero, which is why each line declares what it covers instead of quoting a floor.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from epl.benchmarks import vig
from epl.predictors import Evidence, register
from epl.windows import season_label

#: The market-average pre-match book, in the ordinal (Home, Draw, Away) order. Already spliced by
#: the ingest: `BbAvH/D/A` for 2005/06-2018/19 and `AvgH/D/A` from 2019/20, which are the same
#: quantity under two spellings. One column here means there is nowhere for a per-era branch to
#: hide, which is what "no change of definition visible at the join" has to mean in code.
PREMATCH_COLUMNS: tuple[str, ...] = (
    "prematch_odds_home",
    "prematch_odds_draw",
    "prematch_odds_away",
)

#: The market-average closing book, 2019/20 onward — the Ceiling Line's input, and the one thing
#: the ledger's allow-list withholds from every other Predictor.
#:
#: Spelled out rather than aliased to `epl.ledger.schema.PRIVILEGED_FIXTURE_COLUMNS`. Importing
#: that tuple would make the Ceiling Line's claim and the ledger's check the same object, so the
#: check would pass by construction and prove nothing about the only Predictor that makes a claim.
#: Written out, the claim is an independent statement the ledger genuinely tests — and
#: `tests/benchmarks/test_market.py` asserts the two lists still match.
CLOSING_COLUMNS: tuple[str, ...] = (
    "closing_odds_home",
    "closing_odds_draw",
    "closing_odds_away",
)


class MarketError(Exception):
    """A Fixture reached a market Predictor without the book it was supposed to be priced at."""


class OddsLine:
    """A Predictor that reads one book of decimal odds off the Fixture and removes the vig.

    Stateless and history-free by construction: :meth:`predict` never touches its Evidence, which
    is what makes ``inputs_seen = 0`` a fact about the Predictor rather than an oversight.
    """

    def __init__(
        self,
        name: str,
        columns: tuple[str, ...],
        *,
        method: str = vig.DEFAULT_METHOD,
        note: str = "",
        privileged: bool = False,
    ) -> None:
        self.name = name
        self.columns = columns
        self.method = method
        self.note = note
        #: The claim the ledger checks against its own list of claimable columns. Empty unless
        #: this line reads a book the allow-list withholds, which today means the closing one.
        self.also_sees: tuple[str, ...] = columns if privileged else ()

    def covers(self, fixtures: pd.DataFrame) -> npt.NDArray[np.bool_]:
        """Which Fixtures carry the book this line reads.

        A Season Football-Data priced is a Season with a market to compare against; one it did not
        is a Season with no comparison at all, not a Season the market priced at nothing. Saying
        so here is what keeps those Seasons off the ledger rather than in it as invented rows.

        It asks :func:`epl.benchmarks.vig.is_book` rather than merely checking for nulls, so that
        this and :meth:`_book` cannot disagree about what a book is — a Fixture claimed here and
        then refused there would stop a backfill on a row nobody meant to walk over.
        """
        return vig.is_book(self._prices(fixtures).to_numpy(dtype=float))

    def overround(self, fixtures: pd.DataFrame) -> npt.NDArray[np.float64]:
        """The margin in each Fixture's book, before it was removed.

        The sanity check issue #8 asks be reported alongside every Market Line rather than
        trusted. ``python -m epl.benchmarks overround`` is where it is reported per Season.
        """
        return vig.overround(self._book(fixtures))

    def predict(
        self, fixtures: pd.DataFrame, evidence: Evidence
    ) -> npt.NDArray[np.float64]:
        """The book, vig removed — and nothing from ``evidence``, deliberately.

        The market's price is a fact about the Fixture, sampled at the As-Of Instant itself
        (ADR 0001), so there is no history for this Predictor to read and none for it to leak.
        """
        return vig.remove(self._book(fixtures), method=self.method)

    def _prices(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """This line's three columns, with any the frame does not carry filled in as absent.

        An era before the column existed and a hole inside one that does are the same thing to a
        Predictor: a Fixture with no book. They are told apart by :meth:`covers`, which keeps the
        first off the ledger, and by :meth:`_book`, which refuses the second.
        """
        return pd.DataFrame(
            {
                name: pd.to_numeric(fixtures[name], errors="coerce").to_numpy(
                    dtype=float, na_value=np.nan
                )
                if name in fixtures.columns
                else np.full(len(fixtures), np.nan)
                for name in self.columns
            },
            index=fixtures.index,
        )

    def _book(self, fixtures: pd.DataFrame) -> npt.NDArray[np.float64]:
        """The three prices as an ``(n, 3)`` array, refusing anything that is not a book.

        Refusing rather than falling back to a floor. A Fixture that reached here unpriced is
        either a Season this line should never have been walked over — which :meth:`covers` is
        supposed to have prevented — or a hole in the ingest. Both are bugs, and a third each
        would hide either one behind a plausible-looking number on the scoreboard.
        """
        try:
            return vig.as_book(self._prices(fixtures).to_numpy(dtype=float))
        except vig.VigError as unpriced:
            raise MarketError(
                f"{self.name}: {self._unpriced(fixtures)} — {unpriced}"
            ) from unpriced

    def _unpriced(self, fixtures: pd.DataFrame) -> str:
        """Name the Seasons whose Fixtures arrived without a book, so the complaint is
        actionable rather than a count."""
        missing = fixtures.loc[~self.covers(fixtures)]
        if missing.empty or "season" not in missing.columns:
            return f"{len(missing)} Fixtures have no book"
        seasons = ", ".join(season_label(int(season)) for season in sorted(set(missing["season"])))
        return f"{len(missing)} Fixtures in {seasons} have no book"


#: The opponent. Pre-match rather than closing, against convention, so that the model, the market
#: and the Pundits are compared on one information set (ADR 0001).
#:
#: Its note points at the margin behind the number. A vig-removed line is a *derived* Prediction,
#: and issue #8 asks that the removal be "sanity-checked rather than trusted" — so a reader meeting
#: 0.1936 on the scoreboard is told, there, where to go and look at what was taken out.
MARKET_LINE = register(
    OddsLine(
        "market_line",
        PREMATCH_COLUMNS,
        note="vig removed by Shin; check the margin with `python -m epl.benchmarks overround`",
    )
)

#: The reference upper bound. Registered like anything else — the scoreboard has no branch per
#: Predictor and must not grow one — but carrying the caveat that has to travel with its score.
CEILING_LINE = register(
    OddsLine(
        "ceiling_line",
        CLOSING_COLUMNS,
        privileged=True,
        note=(
            "reference only: closing odds know team news the model cannot, "
            "and cover 2019/20 onward, so this RPS is not comparable to a full-window one"
        ),
    )
)


#: Canonical column order for :func:`overround_report`. ``note`` carries each line's caveat, so
#: the Ceiling Line is labelled here as it is on the scoreboard rather than sitting as a bare row
#: beside the Market Line.
OVERROUND_COLUMNS: tuple[str, ...] = (
    "predictor",
    "season",
    "season_label",
    "division",
    "fixtures",
    "mean_overround",
    "min_overround",
    "max_overround",
    "note",
)


def overround_report(matches: pd.DataFrame) -> pd.DataFrame:
    """The margin each line's book carried, per Season and per tier — the vig removal's receipt.

    Issue #8 asks that the overround be reported alongside every Market Line "so the vig removal
    can be sanity-checked rather than trusted". This is that report: a method that quietly stopped
    removing anything would show up here as a margin that never fell, and the long decline it does
    show — 9.4% in 2005/06 down to ~4.1% in the early 2020s — is a fact about the market that no
    bug would reproduce by accident.

    Seasons a line does not cover are absent rather than blank, because a Season with no market is
    not a Season whose margin was zero (ADR 0001).
    """
    rows = []
    for line in (MARKET_LINE, CEILING_LINE):
        for (season, division), group in matches.groupby(["season", "division"], sort=True):
            covered = line.covers(group)
            if not covered.any():
                continue
            margin = line.overround(group.loc[covered])
            rows.append(
                {
                    "predictor": line.name,
                    "season": int(season),
                    "season_label": season_label(int(season)),
                    "division": str(division),
                    "fixtures": int(covered.sum()),
                    "mean_overround": float(margin.mean()),
                    "min_overround": float(margin.min()),
                    "max_overround": float(margin.max()),
                    "note": line.note,
                }
            )
    return pd.DataFrame(rows, columns=list(OVERROUND_COLUMNS))
