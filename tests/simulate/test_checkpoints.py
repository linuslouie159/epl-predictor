"""Where a Season Projection is taken, and — more to the point — where it is not.

Issue #14's sharpest acceptance criterion is a negative one: the posterior is fitted "only at
Season Projection points — weekly during the live Season, roughly six checkpoints per historical
Season for validation — never at all 1,332 rounds". A full posterior at every Prediction Round is
what ADR 0007 refuses, because it turns a backtest into an overnight job and buys nothing: with
~10,000 observations, parameter uncertainty barely moves a single Fixture's probability.

(The ticket's 1,332 is the design's figure. Stage 2 measured **1,189** and corrected it —
docs/DECISIONS.md — and the corrected number is the one used everywhere else here.)

So this is a test about restraint. It uses a synthetic Season rather than the corpus, so it says
what the rule is rather than what one Season happened to produce.
"""

from __future__ import annotations

import pandas as pd
import pytest

from epl.rounds import prediction_rounds
from epl.simulate.checkpoints import (
    CHECKPOINTS_PER_SEASON,
    CheckpointError,
    projection_rounds,
)


def season_of(fixtures: int = 380, clubs: int = 20) -> pd.DataFrame:
    """A Season shaped like a real one: ten Fixtures a week over 38 weeks."""
    rows = []
    start = pd.Timestamp("2024-08-17")
    for week in range(fixtures // (clubs // 2)):
        kickoff = start + pd.Timedelta(weeks=week)
        for match in range(clubs // 2):
            rows.append(
                {
                    "season": 2024,
                    "division": "E0",
                    "date": (kickoff + pd.Timedelta(days=match % 2)).date(),
                    "time": pd.NA,
                    "home_club": f"club_{(2 * match) % clubs}",
                    "away_club": f"club_{(2 * match + 1) % clubs}",
                    "home_goals": 1,
                    "away_goals": 0,
                    "outcome": "H",
                }
            )
    return pd.DataFrame(rows)


class TestHowManyFitsASeasonCosts:
    def test_a_historical_season_gets_a_handful_of_checkpoints_not_every_round(self) -> None:
        matches = season_of()
        every_round = prediction_rounds(matches)

        chosen = projection_rounds(matches, season=2024)

        assert len(chosen) == CHECKPOINTS_PER_SEASON
        assert len(every_round) > 4 * CHECKPOINTS_PER_SEASON

    def test_the_live_season_is_projected_every_round(self) -> None:
        """Weekly during the live Season, which is the other half of the criterion.

        A live projection is published as the Season unfolds, so it is taken at every round rather
        than at six of them — the cost that is unacceptable 21 times over is fine once.
        """
        matches = season_of()

        chosen = projection_rounds(matches, season=2024, live=True)

        assert len(chosen) == len(prediction_rounds(matches))

    def test_every_checkpoint_is_a_real_prediction_round(self) -> None:
        """Chosen from the rounds, never invented — so the As-Of Instant rule still holds."""
        matches = season_of()
        every_round = prediction_rounds(matches)

        chosen = projection_rounds(matches, season=2024)

        assert set(chosen["prediction_round"]) <= set(every_round["prediction_round"])
        assert chosen["as_of_instant"].is_monotonic_increasing


class TestWhatKeepsItOffEveryRound:
    def test_the_posterior_is_not_a_registered_predictor(self) -> None:
        """The actual guard against a four-day backfill, and it is structural rather than a check.

        `fit` takes a Sample and cannot tell which round it came from, so it could not refuse an
        all-rounds run even in principle. What prevents one is that nothing in `epl.simulate`
        implements the Predictor contract — so `python -m epl.ledger backfill`, the one thing that
        walks all 1,189 rounds, has no way to reach it. Registering one here would silently turn a
        four-minute fit into the overnight job ADR 0007 splits the two fits to avoid.
        """
        import epl.simulate  # noqa: F401 - imported for its side effects, which must be none
        from epl.predictors import registered

        from_simulate = [
            predictor
            for predictor in registered()
            if type(predictor).__module__.startswith("epl.simulate")
        ]

        assert from_simulate == []


class TestWhichDivisionsRoundsCount:
    def test_a_lower_league_round_is_not_a_checkpoint(self) -> None:
        """A Season Projection is of the Premier League table, so its rounds are E0's rounds.

        Dixon-Coles is fitted across all four tiers (ADR 0004) and always will be here too — this
        narrows which *instants* are candidates, not what a fit at one of them sees. Handed the
        whole pyramid instead, 2015/16 offers 70 Prediction Rounds rather than 46, and a checkpoint
        could land on a midweek League Two round at which no Premier League Fixture had moved.
        """
        premier_league = season_of()
        league_two = season_of().assign(division="E3")
        league_two["date"] = pd.to_datetime(league_two["date"]) + pd.Timedelta(days=3)
        league_two["date"] = league_two["date"].dt.date

        both = pd.concat([premier_league, league_two], ignore_index=True)

        assert len(projection_rounds(both, season=2024, live=True)) == len(
            projection_rounds(premier_league, season=2024, live=True)
        )


class TestWhereTheCheckpointsFall:
    def test_they_are_spread_across_the_season_rather_than_bunched(self) -> None:
        """Six checkpoints at the same end of a Season would validate one thing six times.

        The point of projecting from mid-Season repeatedly is to see the distribution tighten as
        Fixtures are played, which needs the checkpoints spread over the campaign.
        """
        matches = season_of()

        chosen = projection_rounds(matches, season=2024)
        gaps = chosen["as_of_instant"].diff().dropna()

        assert gaps.min() >= pd.Timedelta(days=21)

    def test_the_first_checkpoint_waits_for_the_season_to_say_something(self) -> None:
        """Projecting a final table before a ball is kicked is a fit with no Season in it.

        The strengths would be entirely last Season's, and the projection would be measuring the
        decay horizon rather than the campaign.
        """
        matches = season_of()
        every_round = prediction_rounds(matches)

        chosen = projection_rounds(matches, season=2024)

        assert chosen["as_of_instant"].iloc[0] > every_round["as_of_instant"].iloc[0]

    def test_a_season_shorter_than_its_checkpoints_gets_what_it_has(self) -> None:
        """Asking for six checkpoints from four rounds should give four, not raise."""
        matches = season_of(fixtures=40)

        chosen = projection_rounds(matches, season=2024)

        assert 0 < len(chosen) <= len(prediction_rounds(matches))

    def test_a_season_with_no_fixtures_is_refused_rather_than_returned_empty(self) -> None:
        with pytest.raises(CheckpointError, match="2019"):
            projection_rounds(season_of(), season=2019)
