"""The final league table, and the chain that decides who is above whom when two Clubs are level.

A Season Projection is a distribution over final league tables (CONTEXT.md), so this is where a
simulated Season stops being a heap of Scorelines and becomes a table with a champion in it. It
knows nothing about a posterior, a Predictor or an As-Of Instant: it takes Fixtures and goals and
returns positions, which is what makes the tiebreaker chain unit-testable against hand-built
leagues rather than only against whatever a 10,000-Season run happened to produce.

**Ties are the reason this module exists rather than a `sort_values` call.** Across the 26 Seasons
ingested, 24 had at least one pair of Clubs level on points and the average Season had 3.3 of them.
Goal difference settles nearly all of those, which is exactly why the Season Projection could not
be built on Elo — an Outcome does not imply a Scoreline, and a table needs goals.

The chain is :data:`TIEBREAKERS`, and it is issue #15's, not the Premier League's. The competition's
own regulation runs points, goal difference, goals scored, and then — with no head-to-head step at
all — declares the Clubs to occupy the same position, holding a play-off at a neutral ground only
if the championship, relegation or a European place turns on it. The two head-to-head steps here
sit between goals scored and that play-off, so the chain is a strict *refinement* of the real one:
it changes nothing the regulation decides, and it replaces some coin flips with a rule. Over the
corpus it is measurable rather than arguable, and
``tests/simulate/test_projection_over_the_corpus.py`` measures it.

**Every simulated Season shares the Fixtures already played**, and :class:`Slate` is what makes
that structural. It holds the results of the played Fixtures and takes goals only for the ones
that have not been played, so there is no argument through which a projection could hand itself
the result of a Fixture it is supposed to be forecasting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

#: What a win and a draw are worth. Three points for a win throughout the ingested corpus — the
#: change came in 1981, nineteen Seasons before :data:`epl.windows.FIRST_SEASON`.
POINTS_FOR_A_WIN = 3
POINTS_FOR_A_DRAW = 1

#: The chain, in the order it is applied. Stated as data rather than left implicit in the sort, so
#: that a report can print what it used and a reader can check the order without reading
#: :meth:`Slate.positions`. See the module docstring on how this differs from the competition's own
#: regulation, and why the difference is a refinement rather than a disagreement.
TIEBREAKERS: tuple[str, ...] = (
    "points",
    "goal difference",
    "goals scored",
    "head-to-head points, over the matches among the tied Clubs only",
    "head-to-head away goals, over the same matches",
    "a play-off at a neutral ground, taken here as a coin flip",
)

#: Canonical column order for a readable table.
TABLE_COLUMNS: tuple[str, ...] = (
    "position",
    "club",
    "played",
    "won",
    "drawn",
    "lost",
    "goals_for",
    "goals_against",
    "goal_difference",
    "points",
)


class TableError(Exception):
    """A league table was asked for from Fixtures and goals that do not line up."""


@dataclass(frozen=True)
class Standings:
    """What every Club took out of a Season, one row per simulated Season.

    Every array is ``(seasons, clubs)`` and indexed by :attr:`clubs`. Held as arrays rather than as
    a frame because a projection builds this ten thousand times and reads it as numbers; the frame
    a person reads is :meth:`Slate.table`.
    """

    clubs: tuple[str, ...]
    played: npt.NDArray[np.int64]
    won: npt.NDArray[np.int64]
    drawn: npt.NDArray[np.int64]
    lost: npt.NDArray[np.int64]
    goals_for: npt.NDArray[np.int64]
    goals_against: npt.NDArray[np.int64]
    points: npt.NDArray[np.int64]

    @property
    def goal_difference(self) -> npt.NDArray[np.int64]:
        return self.goals_for - self.goals_against


@dataclass(frozen=True)
class Finish:
    """One batch of simulated Seasons played out to a final table.

    :attr:`level_pairs` is the count nobody would think to ask for and everybody should: how many
    adjacent pairs, across these Seasons, were still level after points, goal difference and goals
    scored — which is exactly how often the two head-to-head steps of :data:`TIEBREAKERS` had
    anything to decide. Over the 26 real Seasons in the corpus it is **zero**; the chain's lower
    half exists for the simulated tables, and this is what lets a projection say how often it was
    needed rather than assume.
    """

    standings: Standings
    #: ``(seasons, clubs)``, one-based, in :attr:`Standings.clubs` order.
    positions: npt.NDArray[np.intp]
    level_pairs: int


@dataclass(frozen=True)
class Slate:
    """One Season's Fixtures with Clubs reduced to indices, the played ones first.

    The ordering is not cosmetic. A projection hands back goals for the Fixtures that have not been
    played, and those goals line up with :attr:`home` and :attr:`away` positionally — putting the
    played Fixtures first is what makes "the rest" a slice rather than a mask that could be built
    two different ways in two places.

    :attr:`results` holds the goals of the played Fixtures and nothing else. A Slate therefore has
    no way to express "the result of a Fixture that has not kicked off", which is the project's one
    rule (CLAUDE.md) expressed in a type rather than in a check.

    Build with :meth:`of` or :meth:`finished`. Clubs are ordered by name so that the same Season
    always produces the same table and a re-run reproduces a published projection exactly.
    """

    clubs: tuple[str, ...]
    #: Club index of the home and the away Club of every Fixture, played first. ``(fixtures,)``.
    home: npt.NDArray[np.intp]
    away: npt.NDArray[np.intp]
    #: ``(played, 2)`` — home goals and away goals of the Fixtures behind the As-Of Instant.
    results: npt.NDArray[np.int64]

    @classmethod
    def of(cls, played: pd.DataFrame, remaining: pd.DataFrame) -> Slate:
        """A Season split at an As-Of Instant: what is known, then what is to be simulated.

        ``played`` needs ``home_club``, ``away_club``, ``home_goals`` and ``away_goals``;
        ``remaining`` needs only the two Club columns, **and only those two are read**. A caller
        that hands over a historical Season's remaining Fixtures with their results still attached
        — which is exactly what validation does — cannot leak them through here.
        """
        clubs = tuple(
            sorted(
                set(_column(played, "home_club"))
                | set(_column(played, "away_club"))
                | set(_column(remaining, "home_club"))
                | set(_column(remaining, "away_club"))
            )
        )
        position = {club: index for index, club in enumerate(clubs)}
        return cls(
            clubs=clubs,
            home=_coded(played, remaining, "home_club", position),
            away=_coded(played, remaining, "away_club", position),
            results=_results(played),
        )

    @classmethod
    def finished(cls, matches: pd.DataFrame) -> Slate:
        """A Season with nothing left to simulate — a real final table, or a test's league."""
        return cls.of(matches, matches.iloc[:0])

    @property
    def played(self) -> int:
        """How many Fixtures have been played, which is where the simulated ones begin."""
        return int(self.results.shape[0])

    @property
    def remaining(self) -> int:
        """How many Fixtures a projection has to simulate."""
        return len(self) - self.played

    @property
    def club_count(self) -> int:
        return len(self.clubs)

    def so_far(self) -> Slate:
        """The same Season with the Fixtures still to come dropped — the table as it stands.

        Every Club keeps its place in :attr:`clubs`, including one that has somehow not played
        yet, so the columns of a projection line up with the columns of the table it started from.
        """
        return Slate(
            clubs=self.clubs,
            home=self.home[: self.played],
            away=self.away[: self.played],
            results=self.results,
        )

    def __len__(self) -> int:
        return len(self.home)

    def goals(self, simulated: npt.ArrayLike | None) -> npt.NDArray[np.int64]:
        """Every Fixture's Scoreline in every simulated Season: ``(seasons, fixtures, 2)``.

        ``simulated`` is ``(seasons, remaining, 2)`` — one Scoreline per unplayed Fixture per
        simulated Season, in :attr:`home`'s order. ``None`` means there is nothing left to
        simulate, and is refused if there is.
        """
        if simulated is None:
            if self.remaining:
                raise TableError(
                    f"this Season has {self.remaining} Fixture(s) left to play, so a table "
                    "cannot be built without simulating them"
                )
            drawn = np.empty((1, 0, 2), dtype=np.int64)
        else:
            drawn = np.asarray(simulated, dtype=np.int64)

        if drawn.ndim != 3 or drawn.shape[1:] != (self.remaining, 2):
            raise TableError(
                f"expected simulated goals shaped (seasons, {self.remaining}, 2) for the "
                f"Fixtures that have not been played; got {drawn.shape}"
            )

        seasons = int(drawn.shape[0])
        already = np.broadcast_to(self.results, (seasons, self.played, 2))
        return np.concatenate([already, drawn], axis=1)

    def finish(
        self, simulated: npt.ArrayLike | None, rng: np.random.Generator
    ) -> Finish:
        """Play these simulated Seasons out to a final table.

        One call rather than :meth:`standings` and :meth:`positions` in turn, because the second is
        computed from the first and a projection wants both: asking separately would fold every
        Fixture twice, ten thousand Seasons over.

        ``rng`` is required rather than defaulted. The last step of :data:`TIEBREAKERS` is a coin
        flip, and a projection that reached for an unseeded generator would be one whose published
        table could not be reproduced — which is an acceptance criterion, not a nicety.
        """
        goals = self.goals(simulated)
        standings = self._standings(goals)

        # Ascending by goals scored, then goal difference, then points — `lexsort` takes its
        # primary key last — and reversed, so the champion is column 0. The three that sort here
        # are exactly the three steps of the chain that need no head-to-head record.
        order = np.lexsort(
            (standings.goals_for, standings.goal_difference, standings.points), axis=-1
        )[:, ::-1]

        level = self._level_with_the_club_above(standings, order)
        for season in np.flatnonzero(level.any(axis=1)):
            self._settle_ties(order[season], level[season], goals[season], rng)

        places = np.tile(np.arange(1, self.club_count + 1, dtype=np.intp), (len(order), 1))
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, places, axis=1)
        return Finish(standings=standings, positions=ranks, level_pairs=int(level.sum()))

    def standings(self, simulated: npt.ArrayLike | None) -> Standings:
        """Points, goals and the win/draw/loss split, per Club and per simulated Season."""
        return self._standings(self.goals(simulated))

    def positions(
        self, simulated: npt.ArrayLike | None, rng: np.random.Generator
    ) -> npt.NDArray[np.intp]:
        """Where every Club finished, ``(seasons, clubs)`` and one-based, in :attr:`clubs` order."""
        return self.finish(simulated, rng).positions

    def table(
        self,
        simulated: npt.ArrayLike | None,
        rng: np.random.Generator,
        *,
        season: int = 0,
        clubs: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        """One simulated Season as a league table a person can read. :data:`TABLE_COLUMNS`.

        ``season`` picks which simulated Season; the default is the first, which is the only one
        when the Slate is :meth:`finished`.
        """
        played_out = self.finish(simulated, rng)
        standings, ranks = played_out.standings, played_out.positions[season]
        frame = pd.DataFrame(
            {
                "position": ranks,
                "club": list(clubs if clubs is not None else self.clubs),
                "played": standings.played[season],
                "won": standings.won[season],
                "drawn": standings.drawn[season],
                "lost": standings.lost[season],
                "goals_for": standings.goals_for[season],
                "goals_against": standings.goals_against[season],
                "goal_difference": standings.goal_difference[season],
                "points": standings.points[season],
            }
        )
        return frame.sort_values("position").reset_index(drop=True)[list(TABLE_COLUMNS)]

    def _standings(self, goals: npt.NDArray[np.int64]) -> Standings:
        """The fold from Scorelines to columns, done once for both Clubs of every Fixture.

        Two ``bincount`` calls per column over a flattened ``(season, club)`` cell index, rather
        than a groupby: a projection calls this once per posterior draw and pandas would make the
        Monte Carlo walk slower than the posterior fit in front of it — minutes to its seconds.
        """
        seasons, fixtures, _ = goals.shape
        cells = self.club_count
        home_goals = goals[:, :, 0].astype(np.float64)
        away_goals = goals[:, :, 1].astype(np.float64)

        season_offset = np.repeat(np.arange(seasons, dtype=np.intp) * cells, fixtures)
        home_cell = season_offset + np.tile(self.home, seasons)
        away_cell = season_offset + np.tile(self.away, seasons)

        def per_club(
            cell: npt.NDArray[np.intp], weights: npt.NDArray[np.generic]
        ) -> npt.NDArray[np.int64]:
            counted = np.bincount(
                cell,
                weights=np.asarray(weights, dtype=np.float64).reshape(-1),
                minlength=seasons * cells,
            )
            return np.rint(counted).astype(np.int64).reshape(seasons, cells)

        home_win = home_goals > away_goals
        away_win = away_goals > home_goals
        level = home_goals == away_goals
        ones = np.ones_like(home_goals)

        played = per_club(home_cell, ones) + per_club(away_cell, ones)
        won = per_club(home_cell, home_win) + per_club(away_cell, away_win)
        drawn = per_club(home_cell, level) + per_club(away_cell, level)
        return Standings(
            clubs=self.clubs,
            played=played,
            won=won,
            drawn=drawn,
            lost=played - won - drawn,
            goals_for=per_club(home_cell, home_goals) + per_club(away_cell, away_goals),
            goals_against=per_club(home_cell, away_goals) + per_club(away_cell, home_goals),
            points=POINTS_FOR_A_WIN * won + POINTS_FOR_A_DRAW * drawn,
        )

    @staticmethod
    def _level_with_the_club_above(
        standings: Standings, order: npt.NDArray[np.intp]
    ) -> npt.NDArray[np.bool_]:
        """``(seasons, clubs - 1)``: whether each Club is level with the one placed above it.

        Level on all three of the sorted steps, which is what makes the head-to-head record the
        next question. Compared as integers rather than through a packed key, because a key wide
        enough to hold points, goal difference and goals scored is a key that has to be argued
        about every time one of them changes range.
        """
        def ordered(values: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
            return np.take_along_axis(values, order, axis=1)

        points = ordered(standings.points)
        difference = ordered(standings.goal_difference)
        scored = ordered(standings.goals_for)
        return (
            (points[:, 1:] == points[:, :-1])
            & (difference[:, 1:] == difference[:, :-1])
            & (scored[:, 1:] == scored[:, :-1])
        )

    def _settle_ties(
        self,
        order: npt.NDArray[np.intp],
        level: npt.NDArray[np.bool_],
        goals: npt.NDArray[np.int64],
        rng: np.random.Generator,
    ) -> None:
        """Re-order each run of level Clubs in one simulated Season, in place.

        A run rather than a pair: three Clubs level on points, goal difference and goals scored is
        one mini-league, and settling them pairwise would let the answer depend on which pair was
        looked at first.
        """
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and level[stop - 1]:
                stop += 1
            if stop - start > 1:
                order[start:stop] = self._mini_league(order[start:stop], goals, rng)
            start = stop

    def _mini_league(
        self,
        tied: npt.NDArray[np.intp],
        goals: npt.NDArray[np.int64],
        rng: np.random.Generator,
    ) -> npt.NDArray[np.intp]:
        """The tied Clubs re-ordered by the matches among *themselves*, then by a coin flip.

        Only those matches. A Club that reached the tie by beating everyone else 5-0 arrives on
        equal points, and that 5-0 has already had its say through goal difference — letting it
        speak again here would be counting it twice.
        """
        seat = np.full(self.club_count, -1, dtype=np.intp)
        seat[tied] = np.arange(len(tied), dtype=np.intp)
        among = (seat[self.home] >= 0) & (seat[self.away] >= 0)

        home_seat = seat[self.home[among]]
        away_seat = seat[self.away[among]]
        home_goals = goals[among, 0].astype(np.float64)
        away_goals = goals[among, 1].astype(np.float64)
        level = home_goals == away_goals
        size = len(tied)

        points = np.bincount(
            home_seat,
            weights=POINTS_FOR_A_WIN * (home_goals > away_goals) + POINTS_FOR_A_DRAW * level,
            minlength=size,
        ) + np.bincount(
            away_seat,
            weights=POINTS_FOR_A_WIN * (away_goals > home_goals) + POINTS_FOR_A_DRAW * level,
            minlength=size,
        )
        away_scored = np.bincount(away_seat, weights=away_goals, minlength=size)

        # One uniform per tied Club, drawn whether or not the steps above have already settled
        # them: the play-off is the last key of the sort, so it decides only a Club still level on
        # everything, and drawing it unconditionally keeps the generator's consumption a function
        # of the tie's size rather than of its shape.
        coin = rng.random(size)
        return tied[np.lexsort((coin, -away_scored, -points))]


def _column(frame: pd.DataFrame, name: str) -> list[str]:
    """One Club column as plain strings, refusing a frame that does not carry it."""
    if name not in frame.columns:
        raise TableError(f"a Season's Fixtures need a {name!r} column; got {sorted(frame.columns)}")
    return [str(club) for club in frame[name].to_numpy(dtype=object)]


def _coded(
    played: pd.DataFrame, remaining: pd.DataFrame, name: str, position: dict[str, int]
) -> npt.NDArray[np.intp]:
    """One Club column of both frames, played first, as indices into the Club table."""
    return np.asarray(
        [position[club] for club in (*_column(played, name), *_column(remaining, name))],
        dtype=np.intp,
    )


def _results(played: pd.DataFrame) -> npt.NDArray[np.int64]:
    """The goals of the played Fixtures, ``(played, 2)``."""
    if not len(played):
        return np.empty((0, 2), dtype=np.int64)
    missing = {"home_goals", "away_goals"} - set(played.columns)
    if missing:
        raise TableError(
            f"a played Fixture needs {sorted(missing)}; got {sorted(played.columns)}"
        )
    return np.column_stack(
        [
            played["home_goals"].to_numpy(dtype=np.int64),
            played["away_goals"].to_numpy(dtype=np.int64),
        ]
    )


__all__ = [
    "POINTS_FOR_A_DRAW",
    "POINTS_FOR_A_WIN",
    "TABLE_COLUMNS",
    "TIEBREAKERS",
    "Finish",
    "Slate",
    "Standings",
    "TableError",
]
