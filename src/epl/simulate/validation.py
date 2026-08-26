"""Where the real champion landed — the check that a projection is calibrated, not just plausible.

Issue #15's least skippable acceptance criterion, and the only one that cannot be satisfied by
reading the code: "validated by simulating from mid-Season across historical Seasons and checking
where the real champion landed, so the projection is known to be calibrated and not merely
plausible". A Season Projection is a distribution over final tables, and a distribution that is
merely plausible — the strong Clubs near the top, the promoted ones near the bottom — is exactly
what an overconfident simulator produces. The way to tell them apart is to run the thing across
completed Seasons and ask whether what it called a 30% chance happened about three times in ten.

Each historical Season is projected from :data:`epl.simulate.checkpoints.CHECKPOINTS_PER_SEASON`
Prediction Rounds, so the same Season appears several times with progressively less football left
to simulate. That is deliberate: the interesting property is not one number but the *tightening*,
and a projection that were merely plausible would tighten just as smoothly while being wrong.

**These points are not independent, and no summary here should be read as though they were.**
Within one projection the twenty Clubs' title probabilities sum to one; across the six checkpoints
of a Season they concern the same eventual champion. So a reliability diagram over 2,520
Club-projections carries far less information than 2,520 independent forecasts would, and its bins
should be read as a shape rather than as a significance test. It is stated here rather than
implied, in the same spirit as the Ceiling Line's note (ADR 0001).

**The real final table is the one the results give**, and the corpus carries no column for an
administrative points deduction — Portsmouth's nine in 2009/10, Everton's and Nottingham Forest's
in 2023/24. Both sides of this comparison are therefore the table as the football alone produced
it, which is self-consistent: the projection is not being marked against a table it was never
asked to predict. It does mean the relegation column of those two Seasons describes a table that
differs from the published one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from epl.metrics.calibration import diagram, error_of
from epl.predictors import Corpus
from epl.simulate.checkpoints import (
    CHECKPOINTS_PER_SEASON,
    PROJECTED_DIVISION,
    projection_rounds,
    season_fixtures,
)
from epl.simulate.posterior import PRIORS, SAMPLING, Priors, Sampling
from epl.simulate.projection import (
    BANDS,
    SIMULATION,
    Bands,
    Projection,
    ProjectionError,
    Simulation,
    project,
)
from epl.simulate.table import Slate
from epl.windows import season_label

#: The three things a projection says about a Club, and the three things that then either happened
#: or did not. Each names two columns of :data:`VALIDATION_COLUMNS`: the probability, and
#: ``<event>_happened``.
EVENTS: tuple[str, ...] = ("title", "european", "relegation")

#: Canonical column order for the raw evidence: one row per Club per projection.
VALIDATION_COLUMNS: tuple[str, ...] = (
    "season",
    "checkpoint",
    "checkpoints",
    "as_of",
    "prediction_round",
    "fixtures_played",
    "remaining",
    "club",
    "position",
    "title",
    "title_happened",
    "european",
    "european_happened",
    "relegation",
    "relegation_happened",
    "seed",
    "sampler_seed",
)

#: Canonical column order for the headline: what each projection said about the Club that went on
#: to win the league, and about the Club it thought would.
CHAMPION_COLUMNS: tuple[str, ...] = (
    "season",
    "checkpoint",
    "as_of",
    "fixtures_played",
    "remaining",
    "champion",
    "title",
    "rank",
    "favourite",
    "favourite_title",
)

#: What a projection's reliability diagram counts, where a Predictor's counts ``predictions``
#: (:data:`epl.metrics.COUNTED_AS`). Named apart for the reason `f714c28` renamed a clashing column
#: at stage 8: a reader with `reliability.csv` and `projection_validation.csv` open together should
#: not find one word meaning two things. A Club-projection is not a Prediction (CONTEXT.md).
COUNTED_AS = "projections"

#: Canonical column order for a reliability diagram over a projection's events.
PROJECTION_RELIABILITY_COLUMNS: tuple[str, ...] = (
    "lower",
    "upper",
    COUNTED_AS,
    "mean_predicted",
    "observed",
    "gap",
)


@dataclass(frozen=True)
class Validation:
    """Every projection taken over the historical Seasons, and what actually happened.

    :attr:`rows` is the evidence — one row per Club per projection,
    :data:`VALIDATION_COLUMNS` — and everything else here is a view over it. Held rather than
    summarised because the summaries are cheap and the run that produced the rows is not: a fit is
    minutes and a full validation is a hundred and twenty-six of them.
    """

    rows: pd.DataFrame
    bands: Bands
    simulation: Simulation
    sampling: Sampling

    @property
    def projections(self) -> int:
        """How many Season Projections — and therefore how many posterior fits — this covers."""
        if self.rows.empty:
            return 0
        return int(self.rows[["season", "checkpoint"]].drop_duplicates().shape[0])

    def champions(self) -> pd.DataFrame:
        """One row per projection: what it said about the Club that went on to win the league.

        :data:`CHAMPION_COLUMNS`. ``rank`` is where the eventual champion stood in that
        projection's own title ordering, so ``1`` means the projection's favourite went on to win.
        ``title`` is the probability it gave them, which is the number the whole exercise is about:
        a projection that named the right favourite every time and gave them 95% would be badly
        calibrated, and one that named them half the time at 40% would be doing well.
        """
        if self.rows.empty:
            return pd.DataFrame(columns=list(CHAMPION_COLUMNS))

        ordered = self.rows.sort_values(
            ["season", "checkpoint", "title", "club"], ascending=[True, True, False, True]
        )
        by_projection = ordered.groupby(["season", "checkpoint"], sort=False)
        ordered = ordered.assign(rank=by_projection.cumcount() + 1)

        favourite = (
            ordered.loc[ordered["rank"] == 1, ["season", "checkpoint", "club", "title"]]
            .rename(columns={"club": "favourite", "title": "favourite_title"})
        )
        champions = (
            ordered.loc[ordered["title_happened"]]
            .rename(columns={"club": "champion"})
            .merge(favourite, on=["season", "checkpoint"], how="left")
        )
        return champions.sort_values(["season", "checkpoint"]).reset_index(drop=True)[
            list(CHAMPION_COLUMNS)
        ]

    def reliability(self, event: str | None = None) -> pd.DataFrame:
        """What was promised in each probability band against how often it happened.

        ``event`` is one of :data:`EVENTS`, or ``None`` to pool all three — which is what makes the
        diagram worth reading at all, since one Season produces one champion and twenty-one of them
        would not fill ten bins.

        Binned by :func:`epl.metrics.diagram`, which is the same ten bins, the same right-open
        edges and the same rounding a Predictor's diagram gets — so the 0.008 below and the 0.006 on
        the scoreboard are the same measurement of two different things rather than two
        measurements. Only the front differs, and it has to: a Prediction is three ordinal Outcomes
        that need one-hotting, and these three events arrive already paired with what happened.
        """
        promised, happened = self._points(event)
        return diagram(promised, happened, counted_as=COUNTED_AS)[
            list(PROJECTION_RELIABILITY_COLUMNS)
        ]

    def calibration_error(self, event: str | None = None) -> float:
        """The reliability diagram as one number: the count-weighted mean absolute gap."""
        table = self.reliability(event)
        if table[COUNTED_AS].sum() == 0:
            raise ProjectionError("no projections to validate")
        return error_of(table, counted_as=COUNTED_AS)

    def describe(self) -> str:
        """The headline, in the four lines a reader needs before looking at any table."""
        if self.rows.empty:
            return "no projections"
        champions = self.champions()
        seasons = sorted(self.rows["season"].unique())
        errors = ", ".join(f"{event} {self.calibration_error(event):.3f}" for event in EVENTS)
        return "\n".join(
            [
                f"{self.projections} projections over {len(seasons)} Seasons "
                f"({season_label(int(seasons[0]))}-{season_label(int(seasons[-1]))}), "
                f"{self.simulation.seasons:,} simulated Seasons each",
                f"the eventual champion was the projection's favourite "
                f"{(champions['rank'] == 1).mean():.0%} of the time, "
                f"in its top three {(champions['rank'] <= 3).mean():.0%}",
                f"it gave the eventual champion a mean title probability of "
                f"{champions['title'].mean():.3f}, and named a favourite at "
                f"{champions['favourite_title'].mean():.3f}",
                f"ten-bin calibration error: {errors}, pooled "
                f"{self.calibration_error():.3f}",
            ]
        )

    def _points(self, event: str | None) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The (promised, happened) pairs a diagram is built over."""
        wanted = EVENTS if event is None else (event,)
        unknown = set(wanted) - set(EVENTS)
        if unknown:
            raise ProjectionError(f"no such event {sorted(unknown)}; a projection has {EVENTS}")
        promised = np.concatenate(
            [self.rows[name].to_numpy(dtype=np.float64) for name in wanted]
        )
        happened = np.concatenate(
            [self.rows[f"{name}_happened"].to_numpy(dtype=np.float64) for name in wanted]
        )
        return promised, happened


def final_positions(
    matches: pd.DataFrame,
    season: int,
    *,
    seed: int,
    division: str = PROJECTED_DIVISION,
) -> pd.Series:
    """Where every Club actually finished, indexed by Club.

    Built through the same :class:`epl.simulate.table.Slate` a projection is scored from, so the
    real table and the ten thousand simulated ones are settled by one chain (:data:`TIEBREAKERS`)
    rather than by two implementations that could disagree about a tie.

    Refuses a Season in which the Clubs have not all played the same number of matches, which is
    what an unfinished Season looks like from here: validating against a partial table would mark
    a projection against a result that had not happened yet.

    A Season the corpus does not hold raises :class:`epl.simulate.checkpoints.CheckpointError`
    instead, for the reason :func:`epl.simulate.projection.slate_at` gives: the cut is
    :func:`epl.simulate.checkpoints.season_fixtures` and both errors are that module's.
    """
    slate = Slate.finished(season_fixtures(matches, season, division=division))
    standings = slate.standings(None)
    played = standings.played[0]
    if len(set(played.tolist())) != 1:
        raise ProjectionError(
            f"{season_label(season)} is not finished — its Clubs have played between "
            f"{played.min()} and {played.max()} matches, so it has no final table to validate "
            "against"
        )

    positions = slate.positions(None, np.random.default_rng(seed))[0]
    return pd.Series(positions, index=list(slate.clubs), name="position")


def validate(
    matches: pd.DataFrame,
    seasons: Iterable[int],
    *,
    checkpoints: int = CHECKPOINTS_PER_SEASON,
    division: str = PROJECTED_DIVISION,
    simulation: Simulation = SIMULATION,
    sampling: Sampling = SAMPLING,
    priors: Priors = PRIORS,
    bands: Bands = BANDS,
    on_projection: Callable[[Projection, pd.DataFrame], None] | None = None,
) -> Validation:
    """Project every named Season from its checkpoints and record what actually happened.

    The expensive one: a posterior fit per checkpoint, about nine minutes each at the shipped
    sampler settings, so twenty-one Seasons at six checkpoints is the overnight job ADR 0007
    confines to exactly this.

    ``on_projection`` is handed each projection **and the rows it produced**, as soon as they
    exist. That is not only for progress printing: a run this long that returned nothing until the
    end would lose eight hours to one interruption, so the caller is given every result at the
    moment it can be saved. ``python -m epl.simulate validate`` writes the file on every call for
    exactly that reason.

    The corpus is prepared once and cut per checkpoint (:class:`epl.predictors.Corpus`), because
    re-deriving 52,672 kickoffs at each of a hundred and twenty-six cuts is minutes of the run
    spent parsing dates.
    """
    corpus = Corpus(matches)
    collected: list[pd.DataFrame] = []

    for season in seasons:
        chosen = projection_rounds(matches, season, checkpoints=checkpoints, division=division)
        finished = final_positions(matches, season, seed=simulation.seed, division=division)
        for index in range(len(chosen)):
            projection = project(
                matches,
                season,
                chosen.iloc[index],
                corpus=corpus,
                simulation=simulation,
                sampling=sampling,
                priors=priors,
                bands=bands,
            )
            rows = rows_for(projection, finished, index + 1, len(chosen))
            collected.append(rows)
            if on_projection is not None:
                on_projection(projection, rows)

    every = (
        pd.concat(collected, ignore_index=True)
        if collected
        else pd.DataFrame(columns=list(VALIDATION_COLUMNS))
    )
    return Validation(rows=every, bands=bands, simulation=simulation, sampling=sampling)


def rows_for(
    projection: Projection, finished: pd.Series, checkpoint: int, checkpoints: int
) -> pd.DataFrame:
    """One projection, joined to the table its Season actually produced.

    Public because it is the whole comparison: everything :class:`Validation` reports is a fold
    over these rows, and a join that could only be exercised by an eight-hour run is a join nobody
    checks. ``finished`` is :func:`final_positions`, indexed by Club.
    """
    if projection.at is None:  # pragma: no cover - `project` always attributes one
        raise ProjectionError("a projection has to say which Season it is of to be validated")

    missing = sorted(set(projection.clubs) - set(finished.index))
    if missing:
        raise ProjectionError(
            f"{len(missing)} Club(s) were projected but are not in the final table: {missing[:5]}"
        )

    places = len(projection.clubs)
    position = finished.reindex(list(projection.clubs)).to_numpy(dtype=np.int64)
    return pd.DataFrame(
        {
            "season": projection.at.season,
            "checkpoint": checkpoint,
            "checkpoints": checkpoints,
            "as_of": projection.at.as_of,
            "prediction_round": projection.at.prediction_round,
            "fixtures_played": projection.fixtures_played,
            "remaining": projection.remaining,
            "club": list(projection.clubs),
            "position": position,
            "title": projection.title,
            "title_happened": position <= projection.bands.title,
            "european": projection.european,
            "european_happened": position <= projection.bands.european,
            "relegation": projection.relegation,
            "relegation_happened": position > places - projection.bands.relegation,
            # Both halves of the randomness, on every row. A validation file is the evidence for a
            # claim about calibration, and a reader who cannot re-run the projections it is made of
            # has to take the claim on trust.
            "seed": projection.simulation.seed,
            "sampler_seed": projection.diagnostics.seed,
        }
    )[list(VALIDATION_COLUMNS)]


__all__ = [
    "CHAMPION_COLUMNS",
    "EVENTS",
    "PROJECTION_RELIABILITY_COLUMNS",
    "VALIDATION_COLUMNS",
    "Validation",
    "final_positions",
    "rows_for",
    "validate",
]
