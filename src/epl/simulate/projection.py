"""The Monte Carlo Season Projection: ten thousand Seasons, played out to a final table.

The last modelling stage (issue #15), and the one that turns everything before it into an answer a
person asked for: not "what will happen in this Fixture" but "who wins the league". Every Fixture
that has not been played is simulated to a Scoreline, the Season is resolved through
:data:`epl.simulate.table.TIEBREAKERS`, and the ten thousand tables that come back are counted into
a probability of the title, of a European place and of relegation for every Club.

**Strengths are drawn from the posterior on every simulated Season, never fixed at a point
estimate.** That is the whole reason ADR 0007 pays minutes for a Bayesian fit where the MLE
takes 0.22 seconds: parameter uncertainty barely moves a single Fixture, "but that same uncertainty
compounds across 380 simulated Fixtures into a final table, and ignoring it is what makes naive
season simulators report a title probability of 48% where the honest answer is 34%". A projection
that called :meth:`epl.simulate.posterior.Posterior.mean` would be that naive simulator, would run
faster, and would look entirely reasonable — which is why
``tests/simulate/test_projection.py`` builds a posterior that disagrees with itself and checks the
disagreement survives.

**Two seeds, and both are recorded.** The sampler's is on every
:class:`epl.simulate.posterior.Diagnostics`; the walk's is on :class:`Simulation`. Between them
they are what makes "re-running with the recorded seed reproduces a published projection exactly"
a property rather than a hope, and :meth:`Projection.describe` prints both.

Within-Season strength drift is deliberately **not** modelled — one draw of
:class:`epl.models.likelihood.Strengths` plays out a whole simulated Season. Measured across 520
club-Seasons, first-half to second-half variation was indistinguishable from sampling noise
(ADR 0007), so a drift term would be fitting noise and would widen every interval here for nothing.

Nothing in this module writes to either Prediction store. A Season Projection is a distribution over
final tables, not a Prediction about a Fixture; the ledger's audits are about Predictions and would
have nothing to say about one of these (ADR 0005).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from epl.models.dixon_coles import DIXON_COLES, DixonColes
from epl.models.likelihood import scorelines
from epl.predictors import Corpus, Evidence
from epl.rounds import kickoff_instants
from epl.simulate.checkpoints import PROJECTED_DIVISION, season_fixtures
from epl.simulate.posterior import PRIORS, SAMPLING, Diagnostics, Posterior, Priors, Sampling
from epl.simulate.posterior import fit as fit_posterior
from epl.simulate.table import Slate
from epl.windows import season_label

#: How many Scoreline probabilities a single chunk of the walk may hold at once. The sampling step
#: compares one uniform against a whole Fixture's cumulative Scoreline grid, which is 289 cells, so
#: a Season of 380 Fixtures costs 110,000 cells and this allows about three dozen Seasons per
#: chunk. Chunking does **not** change the answer: numpy fills a ``(seasons, fixtures)`` block of
#: uniforms row by row, so the same seed produces the same draws whatever the chunk size is.
CHUNK_CELLS = 4_000_000

#: Canonical column order for the table a person reads.
PROJECTION_COLUMNS: tuple[str, ...] = (
    "club",
    "played",
    "points",
    "mean_points",
    "mean_position",
    "title",
    "european",
    "relegation",
)

#: What a *published* projection carries in front of :data:`PROJECTION_COLUMNS`, and behind it.
#:
#: Issue #15 asks for "a fixed deterministic seed recorded in the output" and for "re-running with
#: the recorded seed reproduces a published projection exactly", and a file that names neither the
#: Season it is of nor the two seeds it was produced with satisfies neither. So the identity goes
#: in front and the provenance behind: with these ten values and the corpus, the twenty rows
#: between them can be reproduced and nothing else has to be remembered.
PROJECTION_IDENTITY: tuple[str, ...] = ("season", "as_of", "prediction_round")
PROJECTION_PROVENANCE: tuple[str, ...] = (
    "remaining",
    "simulated_seasons",
    "seed",
    "posterior_draws",
    "sampler_seed",
)


class ProjectionError(Exception):
    """A Season Projection was asked for from a Season, a posterior or bands that do not fit."""


@dataclass(frozen=True)
class Bands:
    """Which finishing positions get a probability of their own, stated rather than derived.

    Issue #15 asks for "the title, European places and relegation", and exactly one of those three
    is a fact about the league rather than a choice. **Relegation is the bottom three** and has
    been since the Premier League went to twenty Clubs in 1995/96, one Season before the corpus
    starts. **The title is first place.** European places are the choice.

    :attr:`european` is the top four — the Champions League places for most of the Evaluation
    Window — and it is a simplification in two directions that a reader should know about rather
    than discover. England has had a fifth place in some recent Seasons through UEFA's coefficient,
    and a European place can also be won by lifting a cup, which is not a league position at all
    and which nothing in this project models. So this is "the top four of the league table", named
    honestly, and a caller who wants a different band passes one.
    """

    title: int = 1
    european: int = 4
    relegation: int = 3


#: The bands every projection reports unless a caller says otherwise.
BANDS = Bands()


@dataclass(frozen=True)
class Simulation:
    """How many Seasons the walk plays out, and the seed that makes the walk reproducible.

    The second of the two places randomness enters a projection — :class:`Sampling` is the first —
    and issue #15's last acceptance criterion is that re-running with the recorded seed reproduces
    a published projection exactly. Both are recorded on :class:`Projection` for that reason.

    Ten thousand Seasons is the ticket's figure and is a resolution rather than a convergence
    threshold: it puts a Monte Carlo standard error of about 0.5 percentage points on a probability
    near a half, and well under that on the small ones, which is finer than a projection is worth
    reporting to.
    """

    seasons: int = 10_000
    #: The day this stage was built, and deliberately *not* :attr:`Sampling.seed`. Two identical
    #: seeds in one output invite a reader to think one of them was reused.
    seed: int = 20260826

    def __post_init__(self) -> None:
        if self.seasons < 1:
            raise ProjectionError(
                f"a Season Projection simulates at least one Season; got {self.seasons}"
            )


#: The walk every projection uses unless a caller says otherwise.
SIMULATION = Simulation()


@dataclass(frozen=True)
class At:
    """The Prediction Round a projection was taken at.

    Carried so that a published table says what it is a projection *of* and when it was taken from.
    A projection is only meaningful against an As-Of Instant: the same Season projected a fortnight
    later is a different distribution, and the whole validation exercise is watching it tighten.
    """

    season: int
    as_of: pd.Timestamp
    prediction_round: str

    @classmethod
    def of(cls, season: int, round_row: pd.Series) -> At:
        """From one row of :func:`epl.simulate.checkpoints.projection_rounds`."""
        return cls(
            season=season,
            as_of=pd.Timestamp(round_row["as_of_instant"]),
            prediction_round=str(round_row["prediction_round"]),
        )

    def describe(self) -> str:
        return (
            f"{season_label(self.season)} as of {self.as_of.date()} "
            f"(round {self.prediction_round})"
        )


@dataclass(frozen=True)
class Projection:
    """A distribution over final league tables (CONTEXT.md), held as counts rather than as tables.

    :attr:`finishes` is ``(clubs, positions)``: how many of the simulated Seasons finished Club *i*
    in position *j + 1*. Counts rather than the ten thousand tables themselves because the tables
    are 200 MB and the counts are 3 kB, and every question issue #15 asks — the title, a European
    place, relegation, where the real champion landed — is a sum over a slice of this.
    """

    clubs: tuple[str, ...]
    finishes: npt.NDArray[np.int64]
    #: Points already on the board at the As-Of Instant, and matches already played. Per Club.
    points: npt.NDArray[np.int64]
    played: npt.NDArray[np.int64]
    #: Final points summed over every simulated Season, which :meth:`table` turns into a mean.
    final_points: npt.NDArray[np.int64]
    #: How many Fixtures the walk had to simulate, and how many posterior draws it had to do it on.
    remaining: int
    draws: int
    #: How often, across every simulated Season, two Clubs finished level on points, goal
    #: difference *and* goals scored — the count of times the head-to-head steps of
    #: :data:`epl.simulate.table.TIEBREAKERS` had anything to decide. Reported rather than assumed:
    #: over the 26 real Seasons in the corpus it is zero, so the only evidence that the lower half
    #: of the chain is ever exercised is this number.
    level_pairs: int
    simulation: Simulation
    bands: Bands
    diagnostics: Diagnostics
    at: At | None = None

    @property
    def seasons(self) -> int:
        """How many Seasons were simulated."""
        return self.simulation.seasons

    @property
    def fixtures_played(self) -> int:
        """How much of the Season is behind this projection — Fixtures, not Club-matches."""
        return int(self.played.sum()) // 2

    def probability(self, first: int, last: int) -> npt.NDArray[np.float64]:
        """The share of simulated Seasons that finished each Club between two positions.

        One-based and inclusive at both ends, so ``probability(1, 1)`` is the title and
        ``probability(18, 20)`` is relegation from a twenty-Club league.
        """
        places = len(self.clubs)
        if not 1 <= first <= last <= places:
            raise ProjectionError(
                f"positions {first}-{last} are not inside a league of {places} Clubs"
            )
        return self.finishes[:, first - 1 : last].sum(axis=1) / self.seasons

    @property
    def title(self) -> npt.NDArray[np.float64]:
        return self.probability(1, self.bands.title)

    @property
    def european(self) -> npt.NDArray[np.float64]:
        return self.probability(1, self.bands.european)

    @property
    def relegation(self) -> npt.NDArray[np.float64]:
        places = len(self.clubs)
        return self.probability(places - self.bands.relegation + 1, places)

    @property
    def mean_position(self) -> npt.NDArray[np.float64]:
        places = np.arange(1, len(self.clubs) + 1, dtype=np.float64)
        return (self.finishes * places).sum(axis=1) / self.seasons

    def table(self) -> pd.DataFrame:
        """The projection as a table a person reads, best expected finish first.

        :data:`PROJECTION_COLUMNS`. ``points`` is what is already on the board and ``mean_points``
        is where the walk expects the Club to end up, so the two together say how much of the
        answer is history and how much is forecast.
        """
        frame = pd.DataFrame(
            {
                "club": list(self.clubs),
                "played": self.played,
                "points": self.points,
                "mean_points": self.final_points / self.seasons,
                "mean_position": self.mean_position,
                "title": self.title,
                "european": self.european,
                "relegation": self.relegation,
            }
        )
        return frame.sort_values("mean_position").reset_index(drop=True)[
            list(PROJECTION_COLUMNS)
        ]

    def published(self) -> pd.DataFrame:
        """:meth:`table`, with what it is a projection of in front and how it was produced behind.

        What gets written to disk, where :meth:`table` is what gets printed. The two differ because
        a terminal and a file want different things: a reader looking at fifteen columns of which
        five are the same on every row is worse off, and a file that does not carry those five
        cannot be re-run (:data:`PROJECTION_PROVENANCE`).

        Refuses an unattributed projection rather than writing one. A published table that does not
        say which Season and which instant it is of is not a Season Projection, it is twenty
        numbers.
        """
        if self.at is None:
            raise ProjectionError(
                "this projection does not say which Season or As-Of Instant it is of, so it "
                "cannot be published; build it through `project`, which attributes one"
            )
        table = self.table()
        return table.assign(
            season=self.at.season,
            as_of=self.at.as_of,
            prediction_round=self.at.prediction_round,
            remaining=self.remaining,
            simulated_seasons=self.seasons,
            seed=self.simulation.seed,
            posterior_draws=self.draws,
            sampler_seed=self.diagnostics.seed,
        )[[*PROJECTION_IDENTITY, *PROJECTION_COLUMNS, *PROJECTION_PROVENANCE]]

    @property
    def where(self) -> str:
        """Which Season and instant this is a projection of, in a few words.

        Its own property because a caller that wants only this half should not have to take
        :meth:`describe` apart to get it.
        """
        return self.at.describe() if self.at is not None else "an unattributed Season"

    def describe(self) -> str:
        """One line, carrying both seeds — see the module docstring on why both."""
        return (
            f"{self.where}: {self.seasons:,} simulated Seasons over {self.draws:,} draws, "
            f"{self.remaining} Fixtures still to play; "
            f"walk seed {self.simulation.seed}, sampler seed {self.diagnostics.seed}; "
            f"{self.level_pairs:,} tables needed the head-to-head steps"
        )


def slate_at(
    matches: pd.DataFrame,
    season: int,
    as_of: pd.Timestamp,
    *,
    division: str = PROJECTED_DIVISION,
) -> Slate:
    """One Season's Fixtures split at an As-Of Instant, ready to be simulated.

    The same cut :class:`epl.predictors.Evidence` applies to the corpus, and applied here for the
    same reason: a Fixture that kicked off strictly before the instant is history, and every other
    Fixture of the Season is what the projection is about. A historical Season's unplayed Fixtures
    still carry their results in the corpus, and :meth:`epl.simulate.table.Slate.of` reads only
    their Club columns — which is what stops validation from being a lookup.

    For the live Season the corpus holds only what has been played, so the Fixtures still to come
    have to be handed in from ``fixtures.csv`` (:mod:`epl.ingest.fixtures`) rather than found here.
    Stage 13 measured that file and it **cannot supply them**: its forward horizon is a couple of
    days, and a projection needs every remaining Fixture of the campaign. So a live projection is
    blocked on a source of upcoming Fixtures rather than on code here — see docs/DECISIONS.md,
    "The live loop, and the input it is still waiting for".

    A Season the corpus does not hold raises :class:`epl.simulate.checkpoints.CheckpointError`
    rather than :class:`ProjectionError`, because the cut is
    :func:`epl.simulate.checkpoints.season_fixtures` and that is exactly what its error means: a
    Season Projection asked for where no Season Projection can be taken.
    """
    within = season_fixtures(matches, season, division=division)
    kickoffs = kickoff_instants(within).to_numpy()
    order = np.argsort(kickoffs, kind="stable")
    within, kickoffs = within.iloc[order], kickoffs[order]
    behind = kickoffs < np.datetime64(pd.Timestamp(as_of))
    return Slate.of(within.iloc[np.flatnonzero(behind)], within.iloc[np.flatnonzero(~behind)])


def project(
    matches: pd.DataFrame,
    season: int,
    at: pd.Series,
    *,
    corpus: Corpus | None = None,
    posterior: Posterior | None = None,
    model: DixonColes = DIXON_COLES,
    simulation: Simulation = SIMULATION,
    sampling: Sampling = SAMPLING,
    priors: Priors = PRIORS,
    bands: Bands = BANDS,
) -> Projection:
    """Fit the posterior at one Prediction Round and walk the Season from it.

    ``at`` is one row of :func:`epl.simulate.checkpoints.projection_rounds`, which is what decides
    where a projection may be taken at all. ``corpus`` is the same ``matches`` with its kickoffs
    already derived and sorted, which is worth passing when many projections are taken over one
    table — validation cuts it 126 times, and each cut would otherwise re-parse 52,672 kickoffs.

    ``posterior`` is the escape hatch, and the reason the expensive half and the cheap half are two
    functions: a fit is minutes — 537 s measured at 2015/16's mid-Season checkpoint — where the
    walk is 2.7 s, so anything re-walking one fit (a different seed, different bands, a sensitivity
    check) passes the fit back in rather than paying a thousandfold for it again.

    The sample is built through ``model``'s own :meth:`epl.models.dixon_coles.DixonColes.sample_at`
    rather than assembled here. ADR 0007's split holds only if both fits see the same matches with
    the same weights, and a second copy of "all four tiers, this decay" in this file would be a
    place for that to stop being true.
    """
    instant = pd.Timestamp(at["as_of_instant"])
    slate = slate_at(matches, season, instant)

    if posterior is None:
        evidence = Evidence.before(corpus if corpus is not None else matches, instant)
        sample = model.sample_at(evidence, also=slate.clubs)
        posterior = fit_posterior(sample, priors=priors, sampling=sampling)

    return simulate(
        slate, posterior, simulation=simulation, bands=bands, at=At.of(season, at)
    )


def simulate(
    slate: Slate,
    posterior: Posterior,
    *,
    simulation: Simulation = SIMULATION,
    bands: Bands = BANDS,
    at: At | None = None,
) -> Projection:
    """Walk ``simulation.seasons`` Seasons over ``slate``, one posterior draw at a time.

    The loop is over *draws* rather than over simulated Seasons, and the difference is worth a
    sentence: every Season assigned to one draw shares its Scoreline grid, so the grid — 289
    probabilities for each of up to 380 Fixtures — is built four thousand times rather than ten
    thousand. Nothing about the answer changes, because which draw a Season runs on was decided
    before the loop by :func:`draw_order`.
    """
    if not len(posterior):
        raise ProjectionError(
            "a Season Projection is drawn from a posterior and this one has no draws in it"
        )
    _check_bands(bands, slate.club_count)
    strength_of = _strength_index(slate, posterior)

    rng = np.random.default_rng(simulation.seed)
    clubs = slate.club_count
    finishes = np.zeros((clubs, clubs), dtype=np.int64)
    final_points = np.zeros(clubs, dtype=np.int64)
    level_pairs = 0

    home = strength_of[slate.home[slate.played :]]
    away = strength_of[slate.away[slate.played :]]
    per_draw = np.bincount(
        draw_order(rng, len(posterior), simulation.seasons), minlength=len(posterior)
    )
    cell_of = np.arange(clubs, dtype=np.intp) * clubs

    for index in np.flatnonzero(per_draw):
        drawn = posterior.draw(int(index))
        home_rate, away_rate = drawn.rates(home, away)
        grid = scorelines(home_rate, away_rate, drawn.correction)
        goals = _sample_scorelines(grid, int(per_draw[index]), rng)

        played_out = slate.finish(goals, rng)
        finishes += np.bincount(
            (cell_of[None, :] + played_out.positions - 1).reshape(-1), minlength=clubs * clubs
        ).reshape(clubs, clubs)
        final_points += played_out.standings.points.sum(axis=0)
        level_pairs += played_out.level_pairs

    so_far = slate.so_far().standings(None)
    return Projection(
        clubs=slate.clubs,
        finishes=finishes,
        points=so_far.points[0],
        played=so_far.played[0],
        final_points=final_points,
        remaining=slate.remaining,
        draws=len(posterior),
        level_pairs=level_pairs,
        simulation=simulation,
        bands=bands,
        diagnostics=posterior.diagnostics,
        at=at,
    )


def draw_order(
    rng: np.random.Generator, draws: int, seasons: int
) -> npt.NDArray[np.intp]:
    """Which posterior draw each simulated Season runs on.

    Every draw used the same number of times give or take one, with the remainder handed out at
    random. Not sampled with replacement, which would add Monte Carlo noise on top of the parameter
    uncertainty this exists to represent — and emphatically not walked in order, because
    :func:`epl.simulate.posterior.fit` concatenates its chains, so the first quarter of a
    posterior's draws are one chain's and a projection over the first ten thousand of them would be
    running on a quarter of the exploration it paid nine minutes for.

    Whole copies first and a random subset for the remainder, rather than a permutation of enough
    whole copies truncated to length: truncating drops draws that were already unevenly spread
    through the permutation, which over 4,000 draws and 10,000 Seasons leaves some used three times
    and others once.
    """
    if draws < 1:
        raise ProjectionError(f"a posterior needs at least one draw; got {draws}")
    whole, remainder = divmod(seasons, draws)
    pool = np.tile(np.arange(draws, dtype=np.intp), whole)
    if remainder:
        pool = np.concatenate([pool, rng.permutation(draws)[:remainder]])
    return np.asarray(rng.permutation(pool), dtype=np.intp)


def _sample_scorelines(
    grid: npt.NDArray[np.float64], seasons: int, rng: np.random.Generator
) -> npt.NDArray[np.int64]:
    """``seasons`` Scorelines per Fixture, drawn from the Scoreline grid: ``(seasons, n, 2)``.

    From the grid rather than from two Poissons, and the difference is Dixon-Coles itself: the
    low-score correction lives in exactly four cells of it, so a walk that sampled the two rates
    independently would be simulating the model with its own correction taken out.
    """
    fixtures = len(grid)
    if not fixtures:
        return np.zeros((seasons, 0, 2), dtype=np.int64)

    size = grid.shape[1]
    cumulative = np.cumsum(grid.reshape(fixtures, -1), axis=1)
    # The grid is normalised, but a float sum of 289 terms lands a few ulps short of one and a
    # uniform above that last edge would index off the end of the grid.
    cumulative[:, -1] = 1.0

    picked = np.empty((seasons, fixtures), dtype=np.int64)
    step = max(1, CHUNK_CELLS // (fixtures * size * size))
    for start in range(0, seasons, step):
        stop = min(start + step, seasons)
        uniform = rng.random((stop - start, fixtures))
        picked[start:stop] = (uniform[:, :, None] > cumulative[None, :, :]).sum(axis=2)

    return np.stack([picked // size, picked % size], axis=-1)


def _strength_index(slate: Slate, posterior: Posterior) -> npt.NDArray[np.intp]:
    """Where each of the Season's Clubs sits in the posterior's parameter vector."""
    position = {club: index for index, club in enumerate(posterior.clubs)}
    missing = sorted(set(slate.clubs) - position.keys())
    if missing:
        raise ProjectionError(
            f"{len(missing)} Club(s) of this Season have no strength in this posterior: "
            f"{missing[:5]}. Name them in `also` when the Sample is built, so a promoted Club is "
            "uncertain rather than unanswerable"
        )
    return np.asarray([position[club] for club in slate.clubs], dtype=np.intp)


def _check_bands(bands: Bands, clubs: int) -> None:
    """The reported bands have to fit inside the league, and not overlap each other."""
    if bands.title < 1 or bands.european < bands.title or bands.relegation < 1:
        raise ProjectionError(
            f"bands must run title <= european and relegation >= 1; got {bands}"
        )
    if bands.european + bands.relegation > clubs:
        raise ProjectionError(
            f"a European band of {bands.european} and a relegation band of {bands.relegation} "
            f"overlap in a league of {clubs} Clubs"
        )


__all__ = [
    "BANDS",
    "CHUNK_CELLS",
    "PROJECTION_COLUMNS",
    "PROJECTION_IDENTITY",
    "PROJECTION_PROVENANCE",
    "SIMULATION",
    "At",
    "Bands",
    "Projection",
    "ProjectionError",
    "Simulation",
    "draw_order",
    "project",
    "simulate",
    "slate_at",
]
