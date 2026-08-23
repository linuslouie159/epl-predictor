"""One Elo rating pool across all four English tiers, folded forward in kickoff order.

A reader of an *EPL* predictor will wonder why League Two results move ratings. The answer is
promoted Clubs (ADR 0004). Over 2001/02-2025/26 promoted Clubs averaged 36.9 points against a
league average of 52.0, but the penalty swings from 44.0 points for the 2010/11 intake to 19.7 for
2024/25 — so any fixed starting rating is wrong by a drifting amount, and any Premier-League-only
prior gives all three promoted Clubs the same rating on day one. Rating the whole pyramid lets a
Club arrive with a rating it earned, keeps a relegated Club updating instead of freezing, and
removes the special case for yo-yo Clubs entirely.

There is therefore **one pool and no per-tier bookkeeping inside it**. Neither :class:`Ratings` nor
:class:`Elo` knows what a division is. A Club that is promoted is a Club whose next opponents
happen to be better, which is the whole mechanism: Elo is zero-sum, so ratings stay comparable
across tiers that are connected only by promotion and relegation.

:func:`newcomers` is the one thing here that reads a division, and it is not part of the pool — it
is how the corpus is asked *who came up*, which is the question ADR 0004 exists to answer well and
the one the ratings above are checked against.

Three things this deliberately does not do:

* **No regression to the mean between Seasons.** Issue #9 names three hyperparameters — K, the
  home-advantage constant and any cross-tier offset — and a summer decay is a fourth that would
  have to be fitted like the others. It is not here because it was not fitted, not because it was
  judged worthless.
* **No cross-tier offset**, and none was fitted or measured. Issue #9 allows for one; the pool has
  one scale by construction, because rating points only ever move between Clubs, so a tier's level
  is already expressed by the Clubs promoted into and relegated out of it. That is an argument
  rather than a measurement, and it is the reason nothing here fits an offset.
* **No calibration.** That is one shared step across every Predictor and lives in
  :mod:`epl.calibration` (ADR 0006, issue #10), which is what lets every metric be reported pre-
  and post-calibration. Measured over the Evaluation Window it costs Elo 0.0009 RPS rather than
  buying any, so what this module emits is also what the scoreboard's headline is read off.

The hyperparameters are fitted inside the Burn-In Window and frozen (ADR 0008). See
:data:`FROZEN_SETTINGS`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

from epl.models.ordered_logit import ModelError, OrderedLogit
from epl.predictors import Evidence, register
from epl.rounds import kickoff_instants

#: How many rating points make one factor of ten in expected score — Elo's own unit, and the one
#: constant here that is a convention rather than a fitted parameter. Changing it would rescale
#: every rating and every fitted K in step, so it is fixed and the scale is fitted instead.
DECADE = 400.0

#: What a Club nobody has seen is worth. Every edge is a *difference* of two ratings, so this
#: cancels for any pair that both started here and is a convention rather than a hyperparameter.
START_RATING = 1500.0

#: What the home Club scored, as Elo reads it. A Draw is half a point to each, which is what makes
#: the update indifferent between two equal Clubs drawing.
HOME_SCORE: dict[str, float] = {"H": 1.0, "D": 0.5, "A": 0.0}


@dataclass(frozen=True)
class Settings:
    """Elo's fitted constants.

    ``k`` is how far one match may move a rating; ``home_advantage`` is what playing at home is
    worth, in rating points, inside the expectation.

    Frozen, because a hyperparameter that can be reassigned after the fact is one that can be
    tuned outside the Burn-In Window without anything looking wrong (ADR 0008).
    """

    k: float
    home_advantage: float
    start_rating: float = START_RATING

    def __post_init__(self) -> None:
        if not self.k > 0:
            raise ModelError(
                f"the K-factor must be positive; got {self.k}. A K of zero is a pool that never "
                "learns anything and still reports ratings"
            )

    def expected(self, edge: float) -> float:
        """What the home Club is expected to score, given the edge it takes into the match.

        Elo's logistic in base ten. Note that this is *not* the ordered logit — this maps an edge
        onto one expected number of points, which is what the rating update needs; ADR 0006's
        mapping onto three probabilities is :mod:`epl.models.ordered_logit`, and it is fitted
        separately because the two answer different questions from the same input.
        """
        return 1.0 / (1.0 + 10.0 ** (-edge / DECADE))


@dataclass
class Ratings:
    """The pool: one rating per Club, folded over matches in kickoff order.

    Mutable and order-dependent on purpose — Elo *is* an online algorithm, and a pool that could
    be rebuilt from a set of matches without their order would not be one. :meth:`update` sorts
    what it is handed, so a caller cannot change the ratings by handing over a frame it happened to
    have sorted by something else.
    """

    settings: Settings
    start: Mapping[str, float] | None = None
    _rating: dict[str, float] = field(init=False, repr=False)
    _played: dict[str, int] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._rating = dict(self.start or {})

    @classmethod
    def through(cls, matches: pd.DataFrame, settings: Settings) -> Ratings:
        """A pool folded over ``matches`` from cold. What a fresh walk starts with."""
        ratings = cls(settings)
        ratings.update(matches)
        return ratings

    @property
    def clubs(self) -> tuple[str, ...]:
        """Every Club the pool holds a rating for, in name order."""
        return tuple(sorted(self._rating))

    def rating(self, club: str) -> float:
        """This Club's rating, or the conventional starting one if it has never been seen.

        A Club promoted into League Two from the National League starts cold here, which ADR 0004
        accepts: it is four tiers away from anything this project predicts.
        """
        return self._rating.get(club, self.settings.start_rating)

    def played(self, club: str) -> int:
        """How many matches this Club's rating rests on.

        Public because it is what open risk 4 is checked with: cross-tier ratings have no burn-in
        at the very start of the corpus, and the claim that this never reaches a scored result is
        a claim about this number at the first scored Prediction Round.
        """
        return self._played.get(club, 0)

    def edge(self, home: str, away: str) -> float:
        """How far ahead the home Club is, counting what being at home is worth.

        **One definition, used twice**: this is the quantity Elo's expectation is taken over when a
        rating is updated, and it is the quantity the ordered logit turns into three probabilities.
        Two definitions would let a Prediction be built from a Supremacy the ratings had never been
        learned through, so both routes go through :func:`_edge` — this one from Club names, and
        :meth:`_fold` from the two ratings it already has in hand.
        """
        return _edge(self.rating(home), self.rating(away), self.settings.home_advantage)

    def edges(
        self, home_clubs: Iterable[str], away_clubs: Iterable[str]
    ) -> npt.NDArray[np.float64]:
        """One edge per Fixture, in the order the Fixtures came in."""
        return np.asarray(
            [
                self.edge(home, away)
                for home, away in zip(home_clubs, away_clubs, strict=True)
            ],
            dtype=np.float64,
        ).reshape(-1)

    def update(self, matches: pd.DataFrame) -> None:
        """Fold more matches in, oldest kickoff first.

        Folding in two goes gives the same pool as folding in one, because a rating only ever
        depends on the matches before it. That is a property of the algorithm rather than a thing
        the backtest exploits: :class:`Elo` deliberately rebuilds from cold at every Prediction
        Round, for reasons written there.
        """
        self._fold(_in_kickoff_order(matches))

    def walk(self, matches: pd.DataFrame) -> pd.DataFrame:
        """:meth:`update`, plus what the model thought on the way through.

        Returns ``matches`` in kickoff order with an ``edge`` column: what the home Club's edge was
        **before** that match's own Outcome moved anything. That is the project's one rule stated
        inside the model rather than around it — a match fitted against a rating its own result
        helped produce would leak without a single stored row looking wrong.

        The frame handed in is not modified. Ordered rather than restored to the caller's order, so
        that an edge and the match it belongs to cannot come apart.
        """
        ordered = _in_kickoff_order(matches)
        return ordered.assign(edge=self._fold(ordered))

    def _fold(self, ordered: pd.DataFrame) -> npt.NDArray[np.float64]:
        """The online update itself, over matches already in kickoff order.

        Returns the pre-match edge of each, which is the only thing :meth:`walk` needs that
        :meth:`update` does not — so there is one loop and no second implementation to drift.
        """
        if ordered.empty:
            return np.empty(0, dtype=np.float64)

        scored = _home_scores(ordered["outcome"])
        edges = np.empty(len(ordered), dtype=np.float64)
        k = self.settings.k
        # Bound once rather than looked up per match: this loop runs tens of millions of times over
        # a backfill, and both are the shared definitions rather than copies of them.
        expected = self.settings.expected
        home_advantage = self.settings.home_advantage
        start = self.settings.start_rating
        rating = self._rating
        played = self._played

        for index, (home, away, points) in enumerate(
            zip(
                ordered["home_club"].to_numpy(dtype=object),
                ordered["away_club"].to_numpy(dtype=object),
                scored,
                strict=True,
            )
        ):
            before_home = rating.get(home, start)
            before_away = rating.get(away, start)
            edge = _edge(before_home, before_away, home_advantage)
            edges[index] = edge
            move = k * (points - expected(edge))
            rating[home] = before_home + move
            rating[away] = before_away - move
            played[home] = played.get(home, 0) + 1
            played[away] = played.get(away, 0) + 1
        return edges


def _edge(home_rating: float, away_rating: float, home_advantage: float) -> float:
    """How far ahead the home Club is — the one definition, in one place.

    Written here rather than twice because the two callers hold different things:
    :meth:`Ratings.edge` has two Club names and looks their ratings up, while :meth:`Ratings._fold`
    already has both ratings in hand and must not look them up again. Both must mean the same
    quantity, or a Prediction could be built from a Supremacy the ratings were never learned
    through.
    """
    return home_rating - away_rating + home_advantage


def _in_kickoff_order(matches: pd.DataFrame) -> pd.DataFrame:
    """``matches`` sorted by kickoff, stably.

    Sorted here rather than required of the caller. Elo is path-dependent, so a frame that arrived
    sorted by Club or by division would quietly produce a different pool from the same matches —
    a discrepancy nothing downstream could see.
    """
    if matches.empty:
        return matches
    return matches.iloc[kickoff_instants(matches).to_numpy().argsort(kind="stable")]


#: Fitted on Seasons 2001/02-2004/05 and frozen (ADR 0008), on 22 Aug 2026, by
#: ``python -m epl.models fit``. 2000/01 warms the ratings up and is not fitted on.
#:
#: 28.5 is a brisk K by the standards of international ratings and an ordinary one for club form.
#: 80 rating points of home advantage sits inside the range football Elo ratings are usually quoted
#: at — and it is only interpretable at all because the draw band is not allowed to absorb it (see
#: :func:`epl.models.burn_in.fit_logit`).
#:
#: Literals rather than a fit at import: this is what the stored backtest was produced from, so a
#: rebuilt file is byte-identical to the last one (ADR 0005), and no Prediction ever depends on
#: whether ``data/raw/`` happens to be populated.
#: ``tests/models/test_elo_over_the_corpus.py`` re-runs the fit and checks these are still what it
#: finds.
FROZEN_SETTINGS = Settings(k=28.5, home_advantage=80.0)

#: The mapping fitted alongside them. A scale of 186.9 rating points to the logit, and a draw band
#: of +/- 0.6232 — which puts the draw probability at 30.2% between evenly matched Clubs, falling
#: away from there on its own as Supremacy grows (ADR 0006).
FROZEN_LOGIT = OrderedLogit(
    scale=186.92494716987136,
    cutpoints=(-0.6231771622238356, 0.6231771622238356),
)


class Elo:
    """Pyramid-wide Elo, registered as a Predictor.

    Handed one Prediction Round's Fixtures and the Evidence visible at its As-Of Instant, it folds
    a rating pool over **everything** that Evidence holds — all four tiers (ADR 0004) — and turns
    each Fixture's edge into three probabilities through the frozen ordered logit (ADR 0006).

    **It rebuilds the pool at every round rather than carrying one forward.** That costs one pass
    over the visible corpus per round instead of one over the whole corpus per backfill, and it is
    deliberate. A pool carried between calls would have to decide whether the Evidence it has just
    been handed extends the one it folded last time, and getting that wrong is the one kind of bug
    this project cannot see: the ratings would be built from the wrong matches while every stored
    row still audited clean, because ``inputs_seen`` and ``latest_input`` are a receipt from
    :class:`~epl.predictors.Evidence` and not from the model. Stateless is worth the minute.
    """

    def __init__(
        self,
        settings: Settings = FROZEN_SETTINGS,
        logit: OrderedLogit = FROZEN_LOGIT,
        *,
        name: str = "elo",
    ) -> None:
        self.name = name
        self.settings = settings
        self.logit = logit

    def ratings_at(self, evidence: Evidence) -> Ratings:
        """The pool as it stood at this Evidence's As-Of Instant.

        Public because it is the model's own view of history and the only way to ask a question
        about it — how warmed up a Club's rating was at the first scored round (open risk 4), or
        which Clubs a Season's promoted three arrived with. Asking through Evidence rather than
        through the corpus keeps even a diagnostic on the right side of the project's one rule.
        """
        return Ratings.through(evidence.matches(), self.settings)

    def predict(
        self, fixtures: pd.DataFrame, evidence: Evidence
    ) -> npt.NDArray[np.float64]:
        """One Prediction per Fixture, from ratings earned strictly before the As-Of Instant."""
        ratings = self.ratings_at(evidence)
        return self.logit.probabilities(
            ratings.edges(fixtures["home_club"], fixtures["away_club"])
        )


#: The registered instance, on the frozen parameters. One instance serves the whole scoreboard
#: because it holds no state between rounds — see :class:`Elo`.
ELO = register(Elo())


def newcomers(matches: pd.DataFrame, season: int, division: str = "E0") -> tuple[str, ...]:
    """The Clubs playing in this tier this Season that were not in it the Season before.

    Elo's business rather than the ingest's, because these Clubs are the entire reason the pyramid
    is rated as one (ADR 0004): a promoted Club has to arrive with a rating it earned, and issue #9
    asks that the three who come up "arrive with distinct ratings rather than a single shared
    starting value". This is how that question is asked of a corpus.

    The first ingested Season has no Season before it, so everyone in it is a newcomer — which is
    true, and is the cold start ADR 0004 accepts and the Burn-In Window absorbs.
    """
    tier = matches.loc[matches["division"] == division]
    return tuple(sorted(_playing(tier, season) - _playing(tier, season - 1)))


def _playing(tier: pd.DataFrame, season: int) -> set[str]:
    """Every Club that appeared in this tier that Season, at home or away.

    Both, not just at home. Over a full Season the two sets are the same, but a Season the corpus
    holds only part of — the current one, mid-way through — can have a Club that has played away
    and not yet at home, and calling it a newcomer the following Season would be wrong twice.
    """
    played = tier.loc[tier["season"] == season]
    return set(played["home_club"]) | set(played["away_club"])


def _home_scores(outcomes: pd.Series) -> npt.NDArray[np.float64]:
    """What the home Club scored in each match, refusing anything that is not an Outcome.

    Refusing rather than skipping. A skipped row would leave the pool quietly different from the
    one the same corpus produced yesterday, and every rating downstream of it would still look
    perfectly reasonable. Evidence holds matches that have already kicked off, so a missing Outcome
    here means the corpus was cut wrong rather than that a Fixture is still pending.
    """
    scores = outcomes.map(HOME_SCORE)
    unreadable = scores.isna()
    if unreadable.any():
        seen = sorted({str(value) for value in outcomes[unreadable].tolist()})
        raise ModelError(
            f"{int(unreadable.sum())} matches carry no outcome Elo can read: {seen}. "
            f"An Outcome is one of {sorted(HOME_SCORE)}"
        )
    return np.asarray(scores.to_numpy(dtype=np.float64), dtype=np.float64)
