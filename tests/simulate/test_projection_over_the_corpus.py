"""The Season Projection over real football, where the claims about it actually live.

``test_league_table.py`` and ``test_projection.py`` make their points on hand-built leagues of
three and four Clubs, which is the right place for "goal difference comes before the head-to-head
record". Three things cannot be said there at all:

* what the tiebreaker chain is *for*. Over the 26 real Seasons in the corpus, 24 had at least one
  pair of Clubs level on points and 85 pairs in all — and **not one pair in 26 years was still
  level after goals scored**. The two head-to-head steps and the coin flip beneath them have never
  been needed by a real Premier League table. They exist for the simulated ones, and the only
  evidence that they are ever exercised is a projection's own ``level_pairs``.
* that the walk is affordable at the size it is actually run at. Ten thousand Seasons over a real
  half-Season of Fixtures and four thousand posterior draws is the thing ADR 0007 buys, and if it
  took an hour the design would be wrong.
* that a historical Season does not leak. The corpus knows Leicester won 2015/16; a projection
  taken at Christmas that same Season must not.

The posterior fitted here is **deliberately too short to converge**, and that is not a
compromise: nothing in this file is a claim about the fit. ``test_posterior_over_the_corpus.py`` is
where the fit is checked, at the settings it actually ships with, and paying twenty minutes again
here would buy the walk nothing.

Needs a populated ``data/raw/``, so it skips when that is absent:

    python -m epl.ingest fetch
    python -m epl.ingest build
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from epl.ingest import load_matches
from epl.models.dixon_coles import DIXON_COLES
from epl.predictors import Evidence
from epl.simulate.checkpoints import projection_rounds
from epl.simulate.posterior import Posterior, Sampling
from epl.simulate.posterior import fit as fit_posterior
from epl.simulate.projection import SIMULATION, Simulation, simulate, slate_at
from epl.simulate.table import Slate
from epl.simulate.validation import final_positions
from epl.windows import FIRST_SEASON, LAST_SEASON

pytestmark = pytest.mark.cache

#: The Season every projection here is taken of: 2015/16, in the middle of the Evaluation Window
#: and the one Season whose answer everybody already knows.
SEASON = 2015

#: Which of the Season's six checkpoints. The third is Christmas, with the Season half played —
#: "simulating from mid-Season", in the acceptance criterion's own words.
CHECKPOINT = 2

#: A posterior short enough to fit inside a test run. See the module docstring: no claim here is
#: about the fit, so it is sized to be cheap rather than to converge.
SHORT = Sampling(draws=40, tune=150, chains=1)

#: How long the ten-thousand-Season walk may take. A ceiling with room in it rather than a
#: benchmark, but low enough to catch the walk becoming the expensive half.
WALK_SECONDS_CEILING = 60


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    return load_matches()


@pytest.fixture(scope="module")
def slate(matches: pd.DataFrame) -> Slate:
    at = projection_rounds(matches, SEASON).iloc[CHECKPOINT]
    return slate_at(matches, SEASON, pd.Timestamp(at["as_of_instant"]))


@pytest.fixture(scope="module")
def posterior(matches: pd.DataFrame, slate: Slate) -> Posterior:
    at = projection_rounds(matches, SEASON).iloc[CHECKPOINT]
    evidence = Evidence.before(matches, pd.Timestamp(at["as_of_instant"]))
    return fit_posterior(DIXON_COLES.sample_at(evidence, also=slate.clubs), sampling=SHORT)


class TestWhatTheChainIsFor:
    def test_ties_on_points_are_routine(self, matches: pd.DataFrame) -> None:
        """24 of 26 Seasons, 85 pairs, 3.3 a Season — the reason a table needs goals."""
        seasons, with_a_tie, pairs = _points_ties(matches)

        assert seasons == 26
        assert with_a_tie == 24
        assert pairs == 85

    def test_no_real_final_table_has_ever_needed_the_head_to_head_steps(
        self, matches: pd.DataFrame
    ) -> None:
        """Goal difference and goals scored settle every points tie the corpus holds.

        So the lower half of the chain is not there for the real tables. It is there because a
        projection produces ten thousand of them at a time — see the test below.
        """
        rng = np.random.default_rng(0)
        deep = sum(
            _finished(matches, season).finish(None, rng).level_pairs
            for season in range(FIRST_SEASON, LAST_SEASON)
        )

        assert deep == 0

    def test_a_ten_thousand_season_walk_does_need_them(
        self, slate: Slate, posterior: Posterior
    ) -> None:
        projection = simulate(slate, posterior, simulation=SIMULATION)

        assert projection.level_pairs > 0


class TestTheRealTableAProjectionIsMarkedAgainst:
    def test_the_season_everyone_remembers_comes_out_the_way_it_did(
        self, matches: pd.DataFrame
    ) -> None:
        positions = final_positions(matches, SEASON, seed=SIMULATION.seed)

        assert positions["leicester"] == 1
        assert sorted(positions[positions > 17].index) == [
            "aston_villa",
            "newcastle",
            "norwich",
        ]

    def test_a_checkpoint_splits_the_season_without_losing_a_fixture(self, slate: Slate) -> None:
        assert slate.club_count == 20
        assert slate.played + slate.remaining == 380
        assert 100 < slate.played < 280


class TestTheWalkAtTheSizeItIsRunAt:
    def test_every_simulated_season_promotes_exactly_one_champion_and_relegates_three(
        self, slate: Slate, posterior: Posterior
    ) -> None:
        """The invariant nothing else checks: these are positions in one table, not three coins.

        Exactly one Club finishes first in every simulated Season, exactly four finish in the top
        four and exactly three go down — so the three columns sum to 1, to 4 and to 3 whatever the
        model believes.
        """
        projection = simulate(slate, posterior, simulation=SIMULATION)

        assert projection.title.sum() == pytest.approx(1.0)
        assert projection.european.sum() == pytest.approx(4.0)
        assert projection.relegation.sum() == pytest.approx(3.0)
        assert projection.finishes.sum() == 20 * SIMULATION.seasons

    def test_the_corpus_knows_who_won_and_the_projection_does_not(
        self, slate: Slate, posterior: Posterior
    ) -> None:
        """The leak test. Leicester led at Christmas and went on to win; every row of the Season
        is in the corpus, including the ones this projection is supposed to be forecasting."""
        projection = simulate(slate, posterior, simulation=SIMULATION)

        title = dict(zip(projection.clubs, projection.title, strict=True))

        assert 0.0 < title["leicester"] < 0.5

    def test_ten_thousand_seasons_is_seconds_not_minutes(
        self, slate: Slate, posterior: Posterior
    ) -> None:
        clock = time.perf_counter()
        simulate(slate, posterior, simulation=SIMULATION)
        elapsed = time.perf_counter() - clock

        assert elapsed < WALK_SECONDS_CEILING

    def test_the_recorded_seed_reproduces_the_walk_exactly(
        self, slate: Slate, posterior: Posterior
    ) -> None:
        """Issue #15's last acceptance criterion, over a real Season rather than a toy one."""
        settings = Simulation(seasons=2_000, seed=SIMULATION.seed)

        first = simulate(slate, posterior, simulation=settings)
        again = simulate(slate, posterior, simulation=settings)

        assert np.array_equal(first.finishes, again.finishes)
        assert first.describe() == again.describe()


def _finished(matches: pd.DataFrame, season: int) -> Slate:
    within = matches.loc[(matches["division"] == "E0") & (matches["season"] == season)]
    return Slate.finished(within)


def _points_ties(matches: pd.DataFrame) -> tuple[int, int, int]:
    """(Seasons, Seasons with at least one pair level on points, pairs in all)."""
    seasons = with_a_tie = pairs = 0
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        within = matches.loc[(matches["division"] == "E0") & (matches["season"] == season)]
        if within.empty:
            continue
        seasons += 1
        points = Slate.finished(within).standings(None).points[0]
        level = int(((points[:, None] == points[None, :]).sum() - len(points)) // 2)
        pairs += level
        with_a_tie += bool(level)
    return seasons, with_a_tie, pairs
