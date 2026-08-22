"""Where every hyperparameter is fitted, and the only place one can be.

Elo's K-factor, its home-advantage constant and the ordered logit's scale and cutpoints are chosen
here, on Seasons 2000/01-2004/05, and then frozen into :data:`epl.models.elo.FROZEN_SETTINGS` and
:data:`epl.models.elo.FROZEN_LOGIT` as literals (ADR 0008). Nothing fits at predict time and
nothing fits at import.

Issue #9 asks that "tuning against data outside the Burn-In Window is not possible by accident",
and a convention someone remembers is not that. So the confinement is structural and doubled:

* :func:`fit` **cuts the corpus to the Burn-In Window before it walks anything**. Handing it all
  26 ingested Seasons is indistinguishable from handing it the five — an Evaluation Window match
  cannot move a rating here, let alone be scored.
* the ``seasons`` argument is then checked against :data:`epl.windows.BURN_IN_WINDOW`, so a caller
  who *meant* to reach outside is told rather than quietly given the intersection.

**2000/01 warms up and is not fitted on.** Every Club starts the corpus at the same conventional
rating, so a Season in which the model knows nothing about anybody would mostly teach the fit to
learn fast. That is open risk 4 — cross-tier ratings have no burn-in before 2000/01 — showing up
inside the Burn-In Window itself. The choice is made here on that reasoning alone and is never
compared against an Evaluation Window score, which would be the very leak ADR 0008 exists to stop.

**RPS is what is minimised**, because RPS is what the scoreboard reports and what the project is
judged on (CLAUDE.md). It is a strictly proper scoring rule for ordinal Outcomes, so minimising it
is not a shortcut; log loss is reported beside it so a reader can see the two agree.

    python -m epl.models fit      re-derive the frozen numbers and print them
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize

from epl import metrics
from epl.models.elo import Ratings, Settings
from epl.models.ordered_logit import ModelError, OrderedLogit
from epl.windows import BURN_IN_WINDOW, season_label

#: The tier the parameters are scored on. All four move ratings (ADR 0004); only the Premier
#: League is ever predicted, so only the Premier League is fitted against.
FITTED_DIVISION = "E0"

#: Seasons the parameters are scored on: the Burn-In Window minus its first Season, which is spent
#: warming up. See the module docstring — this is a decision about cold ratings, not a tuned knob.
FITTING_SEASONS = range(min(BURN_IN_WINDOW) + 1, max(BURN_IN_WINDOW) + 1)

#: Candidate K-factors. Wide enough to contain both the sluggish values used for international
#: ratings and the brisk ones used for club form, so the fit is choosing rather than confirming.
K_GRID: tuple[float, ...] = tuple(float(k) for k in range(6, 41, 2))

#: Candidate home-advantage constants, in rating points. Starts at zero so that "home advantage is
#: worth nothing" stays on the table as an answer the data could give, and runs to 200 — twice the
#: top of the range football Elo ratings are usually quoted at — so that the answer is somewhere
#: inside it rather than against its ceiling.
HOME_ADVANTAGE_GRID: tuple[float, ...] = tuple(float(points) for points in range(0, 201, 10))

#: How many times the grid is halved and re-centred on its own winner. One pass takes the K
#: resolution from 2 points to 0.5 and the home-advantage resolution from 10 points to 2.5, which
#: is far finer than these values are meaningful to.
REFINEMENTS = 1

#: Where the ordered logit search starts: a scale of one Elo decade, and a draw band wide enough
#: to hold roughly a quarter of the distribution at even Supremacy. A fixed start rather than a
#: data-dependent one, so the same corpus gives the same parameters (ADR 0005).
LOGIT_START: tuple[float, float] = (math.log(400.0), math.log(0.55))

#: Nelder-Mead's convergence tolerances. Tight enough that the fitted parameters are stable to
#: more places than they are ever quoted to, and fixed rather than defaulted so a scipy release
#: cannot silently move the frozen numbers.
LOGIT_TOLERANCE = 1e-10
LOGIT_MAX_ITERATIONS = 4000


@dataclass(frozen=True)
class Fit:
    """What one fitting run found, and what it was allowed to look at while finding it.

    ``fixtures`` and ``matches_seen`` are the receipt. They say the parameters were scored on
    Premier League Fixtures inside :data:`FITTING_SEASONS` while the ratings behind them warmed on
    every tier — which is ADR 0004 and ADR 0008 in two integers.
    """

    settings: Settings
    logit: OrderedLogit
    fixtures: int
    matches_seen: int
    rps: float
    log_loss: float

    def describe(self) -> str:
        """The fit as a line a human can compare against the frozen literals."""
        lower, upper = self.logit.cutpoints
        return (
            f"k={self.settings.k:g} home_advantage={self.settings.home_advantage:g} "
            f"scale={self.logit.scale:.4f} cutpoints=({lower:.6f}, {upper:.6f}) "
            f"-> {self.rps:.5f} RPS, {self.log_loss:.5f} log loss "
            f"over {self.fixtures} Fixtures from {self.matches_seen} matches"
        )


def fit(
    matches: pd.DataFrame,
    *,
    seasons: Iterable[int] = FITTING_SEASONS,
    k: Sequence[float] = K_GRID,
    home_advantage: Sequence[float] = HOME_ADVANTAGE_GRID,
    refine: int = REFINEMENTS,
) -> Fit:
    """Fit every hyperparameter on the Burn-In Window, and refuse to look anywhere else.

    ``matches`` may be the whole corpus; only its Burn-In Window rows are walked over. The search
    is a grid on Elo's two constants — both of which have interpretable ranges, so a grid is
    readable where an optimiser's path is not — with the ordered logit fitted exactly at each
    candidate, then re-centred and halved :data:`REFINEMENTS` times.
    """
    wanted = [int(season) for season in seasons]
    outside = sorted(season for season in wanted if season not in BURN_IN_WINDOW)
    if outside:
        raise ModelError(
            f"hyperparameters are fitted in the Burn-In Window and nowhere else (ADR 0008); "
            f"{', '.join(season_label(season) for season in outside)} "
            f"{'is' if len(outside) == 1 else 'are'} outside "
            f"{season_label(min(BURN_IN_WINDOW))}-{season_label(max(BURN_IN_WINDOW))}"
        )

    visible = matches.loc[matches["season"].isin(list(BURN_IN_WINDOW))]
    scored = visible["season"].isin(wanted) & (visible["division"] == FITTED_DIVISION)
    if not scored.any():
        raise ModelError(
            "nothing to fit on: no Burn-In Window Fixtures in "
            f"{FITTED_DIVISION} for the Seasons asked for"
        )

    best = _search(visible, scored.to_numpy(), tuple(k), tuple(home_advantage), refine)
    return best


def _search(
    visible: pd.DataFrame,
    scored: npt.NDArray[np.bool_],
    k_grid: tuple[float, ...],
    home_grid: tuple[float, ...],
    refine: int,
) -> Fit:
    """The grid, re-centred on its own winner ``refine`` times.

    Ties are broken by the first candidate in grid order rather than by whichever the float
    comparison happens to prefer, so the same corpus gives the same parameters every time.
    """
    best = _fit_at(visible, scored, Settings(k_grid[0], home_grid[0]))
    for pass_number in range(refine + 1):
        for k in k_grid:
            for home_advantage in home_grid:
                candidate = _fit_at(visible, scored, Settings(k, home_advantage))
                if candidate.rps < best.rps:
                    best = candidate
        if pass_number == 0:
            _check_inside(best.settings, k_grid, home_grid)
        k_grid = _around(best.settings.k, k_grid, floor=1e-3)
        home_grid = _around(best.settings.home_advantage, home_grid, floor=0.0)
    return best


def _check_inside(
    settings: Settings, k_grid: tuple[float, ...], home_grid: tuple[float, ...]
) -> None:
    """Complain if the coarse grid's winner sits against one of its own walls.

    A search that stops at its boundary has not found an optimum, it has found the edge of what it
    was allowed to consider — and the refinement passes then re-centre on that edge and report it
    as though it were a fitted value. This caught exactly that: the home-advantage grid originally
    stopped at 140 points and the winner sat on it.

    A home advantage of zero is the one boundary that is a real answer rather than a wall, so only
    the top of that grid is checked. And a grid of two candidates has no interior for a winner to
    sit in, so there the check has nothing to say rather than something to complain about — which
    is what lets a test hand over a grid small enough to afford.
    """
    against_the_wall = [
        f"{name}={value:g} sits at the {end} of its grid"
        for name, value, end, wall, grid in (
            ("k", settings.k, "bottom", min(k_grid), k_grid),
            ("k", settings.k, "top", max(k_grid), k_grid),
            ("home_advantage", settings.home_advantage, "top", max(home_grid), home_grid),
        )
        if len(grid) > 2 and value == wall
    ]
    if against_the_wall:
        raise ModelError(
            f"the fit ran to the edge of its own search: {'; '.join(against_the_wall)}. "
            "Widen the grid — a boundary is not an optimum"
        )


def _around(
    centre: float, grid: tuple[float, ...], *, floor: float
) -> tuple[float, ...]:
    """A grid of the same length, half as wide, centred on the winner.

    Clipped at ``floor`` so a refinement cannot propose a K of zero or a negative home advantage —
    the first is a pool that never learns and the second is a claim nothing in the data supports.
    """
    step = (max(grid) - min(grid)) / (len(grid) - 1) / 2 if len(grid) > 1 else 0.0
    reach = step * (len(grid) - 1) / 2
    return tuple(
        max(floor, centre - reach + step * index) for index in range(len(grid))
    )


def _fit_at(
    visible: pd.DataFrame, scored: npt.NDArray[np.bool_], settings: Settings
) -> Fit:
    """Walk the whole Burn-In pyramid under these Elo constants, then fit the logit to what it saw.

    The two halves are fitted this way round because they are not symmetric: the ratings do not
    depend on the logit at all, so one walk serves every logit the search might consider, while
    every logit has to be fitted against the ratings it will actually be used with.
    """
    replayed = Ratings(settings).walk(visible)
    kept = scored[_walk_order(visible, replayed)]
    edges = replayed["edge"].to_numpy(dtype=float)[kept]
    outcomes = replayed["outcome"].to_numpy(dtype=object)[kept]

    logit = fit_logit(edges, outcomes)
    predictions = logit.probabilities(edges)
    return Fit(
        settings=settings,
        logit=logit,
        fixtures=int(kept.sum()),
        matches_seen=len(visible),
        rps=metrics.rps(predictions, outcomes),
        log_loss=metrics.log_loss(predictions, outcomes),
    )


def _walk_order(
    visible: pd.DataFrame, replayed: pd.DataFrame
) -> npt.NDArray[np.intp]:
    """Where each walked row sat in the frame the caller handed over.

    :meth:`epl.models.elo.Ratings.walk` returns its matches in kickoff order, which is not the
    order ``visible`` arrived in, so the mask that says which rows are scored has to be reordered
    to match rather than assumed to line up.
    """
    positions = pd.Series(np.arange(len(visible)), index=visible.index)
    return np.asarray(positions.loc[replayed.index].to_numpy(), dtype=np.intp)


def fit_logit(edges: object, outcomes: object) -> OrderedLogit:
    """The ordered logit that scores best over these edges and the Outcomes that followed them.

    Searched over an unconstrained parameterisation — log scale and log half-width — so that a
    scale of zero and a crossed pair of cutpoints are not points the search can reach. A crossed
    pair would emit negative draw probabilities, which is an ordinal model that has lost the order
    RPS depends on.

    **The draw band is centred on zero, and that is a modelling decision rather than a
    convenience.** An edge already carries what playing at home is worth (:meth:`Ratings.edge`), so
    an edge of zero is a genuinely even contest — and at an even contest Home and Away must be
    equally likely, or the model is claiming a home advantage it has already counted. Letting the
    centre float makes it a second home-advantage parameter pointing the other way, and the two are
    then only weakly told apart.

    Measured on the Burn-In Window, two ways. Held at one arbitrary pair of Elo constants
    (K 30, home advantage 100) the free band scores 0.20552 against the centred band's 0.20583 —
    0.0003 RPS. Searched end to end, where K and the home advantage move too, the free band reaches
    0.20550 and the centred one 0.20554: **0.00004 RPS**, for a home-advantage constant that goes
    from 80 rating points to 155 with the band shifted back to cancel most of it. One parameter, in
    the place where it also moves the ratings, is the honest version of the same model.
    """
    # Validated through the metrics module, then scored as the labels it was handed: every score
    # in this project goes through one definition of what an Outcome is (spec, user story 16).
    observed = np.asarray(outcomes, dtype=object).ravel()
    if len(set(metrics.as_outcomes(observed).tolist())) < 2:
        raise ModelError(
            "an ordered logit cannot be fitted to one Outcome; at least two must have happened"
        )
    edge_array = np.asarray(edges, dtype=np.float64).ravel()

    def objective(free: npt.NDArray[np.float64]) -> float:
        return metrics.rps(_logit_from(free).probabilities(edge_array), observed)

    found = minimize(
        objective,
        _free_from(_logit_from(np.asarray(LOGIT_START, dtype=np.float64))),
        method="Nelder-Mead",
        options={
            "xatol": LOGIT_TOLERANCE,
            "fatol": LOGIT_TOLERANCE,
            "maxiter": LOGIT_MAX_ITERATIONS,
            "maxfev": LOGIT_MAX_ITERATIONS,
        },
    )
    return _logit_from(np.asarray(found.x, dtype=np.float64))


def _logit_from(free: npt.NDArray[np.float64]) -> OrderedLogit:
    """(log scale, log half-width) -> a valid :class:`OrderedLogit` centred on zero."""
    log_scale, log_half_width = (float(value) for value in free)
    half_width = math.exp(log_half_width)
    return OrderedLogit(scale=math.exp(log_scale), cutpoints=(-half_width, half_width))


def _free_from(logit: OrderedLogit) -> npt.NDArray[np.float64]:
    """The inverse of :func:`_logit_from`, so the search can be started from a stated logit."""
    return np.array([math.log(logit.scale), math.log(logit.band / 2)], dtype=np.float64)


def base_rate_rps(matches: pd.DataFrame, *, seasons: Iterable[int] = FITTING_SEASONS) -> float:
    """What the Outcome base rates alone score over the fitting sample.

    The floor a fit has to clear before it is worth reading at all — the same idea as the Naive
    Baseline on the scoreboard, measured here on the fitting sample rather than on the Evaluation
    Window, because comparing a fitted number against a held-out one would flatter it.
    """
    sample = matches.loc[
        matches["season"].isin([int(season) for season in seasons])
        & (matches["division"] == FITTED_DIVISION)
    ]
    outcomes = sample["outcome"]
    rates = np.array(
        [float((outcomes == outcome).sum()) for outcome in metrics.OUTCOMES], dtype=np.float64
    )
    rates = rates / rates.sum()
    return metrics.rps(np.tile(rates, (len(sample), 1)), outcomes.tolist())
