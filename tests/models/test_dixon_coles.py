"""The maximum-likelihood fit, and the Predictor over it.

The likelihood is tested in ``tests/models/test_likelihood.py`` and the walk over real football in
``tests/models/test_dixon_coles_over_the_corpus.py``. What is left for here is whether the fit finds
the right answer and whether the Predictor stays inside the contract:

* strengths recovered from matches generated at strengths a reader can check by eye
* the flat direction removed, so two fits of the same model are comparable
* the time decay actually decaying — the same two Clubs, with the recent half of their history
  reversed, must come out the other way round
* the Predictor reading its history through Evidence, so ``inputs_seen`` is a receipt

Samples here are small and lopsided on purpose. A fit that only works on 9,000 matches is one whose
failures nobody can see.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from epl.models import ModelError
from epl.models.dixon_coles import FROZEN_DECAY, DixonColes, fit
from epl.models.likelihood import MAX_GOALS, Decay, Sample
from epl.predictors import Evidence

AS_OF = pd.Timestamp("2024-08-20")

#: Long enough that nothing in these fixtures is meaningfully discounted, so a test about the fit
#: is not also a test about the decay.
FLAT = Decay(half_life_days=1e6)


def round_robin(
    make_matches: Callable[..., pd.DataFrame],
    scorelines: dict[tuple[str, str], tuple[int, int]],
    *,
    repeats: int = 12,
    date: str = "2024-08-19",
) -> pd.DataFrame:
    """A frame in which each pairing played the same Scoreline ``repeats`` times.

    Repetition rather than random draws, so the maximum-likelihood answer is the arithmetic of the
    table above and a failing test says which number moved.
    """
    return make_matches(
        *[
            {
                "date": date,
                "home_club": home,
                "away_club": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
            }
            for (home, away), (home_goals, away_goals) in scorelines.items()
            for _ in range(repeats)
        ]
    )


def sample_of(matches: pd.DataFrame, decay: Decay = FLAT, **kwargs: object) -> Sample:
    return Sample.of(matches, AS_OF, decay, **kwargs)  # type: ignore[arg-type]


class TestWhatTheFitFinds:
    def test_two_clubs_that_always_draw_one_one_get_the_same_strengths(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        found = fit(
            sample_of(
                round_robin(
                    make_matches,
                    {("arsenal", "chelsea"): (1, 1), ("chelsea", "arsenal"): (1, 1)},
                )
            )
        )

        assert found.attack == pytest.approx(found.attack[::-1], abs=1e-4)
        assert found.defence == pytest.approx(found.defence[::-1], abs=1e-4)
        assert found.home_advantage == pytest.approx(0.0, abs=1e-4)

    def test_it_expects_each_club_to_score_exactly_what_it_scored(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """Maximum likelihood for a log-linear Poisson matches the sufficient statistics exactly:
        every Club's fitted goals scored and conceded equal the ones in the table.

        Fitted with the low-score correction held at zero, because it is the one term that is not a
        Poisson and so the one that puts the moment condition slightly out.
        """
        matches = round_robin(
            make_matches,
            {
                ("arsenal", "chelsea"): (3, 1),
                ("chelsea", "arsenal"): (1, 2),
                ("arsenal", "luton"): (4, 0),
                ("luton", "arsenal"): (1, 3),
                ("chelsea", "luton"): (2, 0),
                ("luton", "chelsea"): (0, 1),
            },
        )
        sample = sample_of(matches)
        found = fit(sample, correction=0.0)
        home_rate, away_rate = found.rates(sample.home, sample.away)

        for club in sample.clubs:
            at_home = sample.home == sample.clubs.index(club)
            away = sample.away == sample.clubs.index(club)
            scored = sample.home_goals[at_home].sum() + sample.away_goals[away].sum()
            expected = home_rate[at_home].sum() + away_rate[away].sum()

            assert expected == pytest.approx(scored, rel=1e-4), club

    def test_the_stronger_club_is_expected_to_score_more(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """The one pairing that is lopsided in the table has to be lopsided in the fit."""
        matches = round_robin(
            make_matches,
            {
                ("arsenal", "chelsea"): (3, 1),
                ("chelsea", "arsenal"): (1, 2),
                ("arsenal", "luton"): (4, 0),
                ("luton", "arsenal"): (1, 3),
                ("chelsea", "luton"): (2, 0),
                ("luton", "chelsea"): (0, 1),
            },
        )
        sample = sample_of(matches)
        home_rate, away_rate = fit(sample).rates(sample.home, sample.away)
        arsenal_at_home_to_luton = (sample.home == sample.clubs.index("arsenal")) & (
            sample.away == sample.clubs.index("luton")
        )

        assert home_rate[arsenal_at_home_to_luton].mean() > 2.5
        assert away_rate[arsenal_at_home_to_luton].mean() < 1.0

    def test_the_best_attack_belongs_to_the_club_that_scores_most(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        found = fit(
            sample_of(
                round_robin(
                    make_matches,
                    {
                        ("arsenal", "chelsea"): (4, 0),
                        ("chelsea", "arsenal"): (0, 4),
                        ("arsenal", "luton"): (4, 0),
                        ("luton", "arsenal"): (0, 4),
                        ("chelsea", "luton"): (1, 1),
                        ("luton", "chelsea"): (1, 1),
                    },
                )
            )
        )

        assert found.table()["club"].iloc[0] == "arsenal"

    def test_home_advantage_is_what_the_same_pairings_score_more_of_at_home(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """Two goals at home and one away in every match, from every Club: log 2."""
        found = fit(
            sample_of(
                round_robin(
                    make_matches,
                    {
                        ("arsenal", "chelsea"): (2, 1),
                        ("chelsea", "arsenal"): (2, 1),
                        ("arsenal", "luton"): (2, 1),
                        ("luton", "arsenal"): (2, 1),
                    },
                )
            )
        )

        assert found.home_advantage == pytest.approx(np.log(2.0), abs=1e-3)

    def test_it_comes_back_in_a_gauge(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """The likelihood is flat along "add a constant to every strength", so an uncentred fit is
        an arbitrary point on a line and not comparable to any other fit."""
        found = fit(
            sample_of(
                round_robin(
                    make_matches,
                    {("arsenal", "chelsea"): (3, 1), ("chelsea", "arsenal"): (0, 2)},
                )
            )
        )

        assert found.attack.mean() == pytest.approx(0.0, abs=1e-9)

    def test_a_club_with_no_matches_keeps_the_neutral_strengths_it_started_at(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """Zero weight is zero gradient. The alternative — dropping the Club — makes a Fixture
        unanswerable rather than uncertain."""
        sample = sample_of(
            round_robin(
                make_matches,
                {("arsenal", "chelsea"): (2, 1), ("chelsea", "arsenal"): (2, 1)},
            ),
            also=["luton"],
        )
        found = fit(sample)
        position = found.clubs.index("luton")

        assert found.attack[position] == pytest.approx(0.0, abs=1e-6)
        assert found.defence[position] == pytest.approx(0.0, abs=1e-6)

    def test_the_same_sample_gives_the_same_fit(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """A rebuilt backtest has to be byte-identical to the last one (ADR 0005)."""
        matches = round_robin(
            make_matches,
            {("arsenal", "chelsea"): (3, 1), ("chelsea", "arsenal"): (0, 2)},
        )
        once, twice = fit(sample_of(matches)), fit(sample_of(matches))

        assert once.attack.tolist() == twice.attack.tolist()
        assert once.correction == twice.correction

    def test_an_empty_sample_fits_nothing_rather_than_failing(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        found = fit(sample_of(make_matches()))

        assert found.clubs == ()

    def test_a_fit_that_runs_out_of_iterations_says_so(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """Loud rather than approximate: an unconverged fit produces a Prediction that looks
        exactly like a converged one on the scoreboard."""
        matches = round_robin(
            make_matches,
            {("arsenal", "chelsea"): (3, 1), ("chelsea", "arsenal"): (0, 2)},
        )

        with pytest.raises(ModelError, match="did not converge"):
            fit(sample_of(matches), max_iterations=1)


class TestTheTimeDecayInsideTheFit:
    def test_recent_form_outweighs_old_form(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """The same two Clubs, thrashing each other in turn. Whoever won recently is stronger."""

        def strongest(recent_winner: str, other: str) -> str:
            matches = make_matches(
                *[
                    {
                        "date": "2023-08-20",
                        "home_club": other,
                        "away_club": recent_winner,
                        "home_goals": 4,
                        "away_goals": 0,
                    }
                    for _ in range(20)
                ],
                *[
                    {
                        "date": "2024-08-19",
                        "home_club": recent_winner,
                        "away_club": other,
                        "home_goals": 4,
                        "away_goals": 0,
                    }
                    for _ in range(20)
                ],
            )
            found = fit(sample_of(matches, Decay(half_life_days=90.0)))
            return str(found.table()["club"].iloc[0])

        assert strongest("arsenal", "chelsea") == "arsenal"
        assert strongest("chelsea", "arsenal") == "chelsea"


class TestThePredictor:
    @pytest.fixture
    def league(self, make_matches: Callable[..., pd.DataFrame]) -> pd.DataFrame:
        """A four-Club league in which one Club is much the best and one much the worst."""
        goals = {"arsenal": 3, "chelsea": 2, "luton": 1, "wolves": 0}
        return make_matches(
            *[
                {
                    "date": "2024-08-19",
                    "home_club": home,
                    "away_club": away,
                    "home_goals": goals[home],
                    "away_goals": goals[away],
                }
                for home, away in itertools.permutations(goals, 2)
                for _ in range(6)
            ]
        )

    @pytest.fixture
    def fixtures(self, make_matches: Callable[..., pd.DataFrame]) -> pd.DataFrame:
        return make_matches(
            {"date": "2024-08-24", "home_club": "arsenal", "away_club": "wolves"},
            {"date": "2024-08-24", "home_club": "wolves", "away_club": "arsenal"},
        )

    def test_it_predicts_one_distribution_per_fixture(
        self, league: pd.DataFrame, fixtures: pd.DataFrame
    ) -> None:
        predictions = DixonColes(FLAT).predict(fixtures, Evidence.before(league, AS_OF))

        assert predictions.shape == (2, 3)
        assert predictions.sum(axis=1) == pytest.approx([1.0, 1.0])

    def test_the_stronger_club_is_favourite_either_way_round(
        self, league: pd.DataFrame, fixtures: pd.DataFrame
    ) -> None:
        home_win, away_win = DixonColes(FLAT).predict(fixtures, Evidence.before(league, AS_OF))

        assert home_win[0] > home_win[1] > home_win[2]
        assert away_win[2] > away_win[1] > away_win[0]

    def test_it_reads_its_history_through_evidence(
        self, league: pd.DataFrame, fixtures: pd.DataFrame
    ) -> None:
        """So ``inputs_seen`` and ``latest_input`` on every stored row are a receipt from Evidence
        rather than a claim by the model (:mod:`epl.predictors`)."""
        evidence = Evidence.before(league, AS_OF)
        DixonColes(FLAT).predict(fixtures, evidence)

        assert evidence.rows_seen == len(league)
        assert evidence.latest_seen is not None
        assert evidence.latest_seen < AS_OF

    def test_it_reads_only_the_tiers_it_was_told_to(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        matches = make_matches(
            {"date": "2024-08-19", "division": "E0"},
            {"date": "2024-08-19", "division": "E1", "home_club": "luton", "away_club": "wolves"},
        )
        evidence = Evidence.before(matches, AS_OF)
        DixonColes(FLAT, divisions=("E0",)).predict(
            make_matches({"date": "2024-08-24"}), evidence
        )

        assert evidence.rows_seen == 1

    def test_a_round_with_no_fixtures_gives_no_predictions(
        self, league: pd.DataFrame, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        predictions = DixonColes(FLAT).predict(
            make_matches().iloc[:0], Evidence.before(league, AS_OF)
        )

        assert predictions.shape == (0, 3)

    def test_it_gives_scorelines_as_well_as_outcomes(
        self, league: pd.DataFrame, fixtures: pd.DataFrame
    ) -> None:
        """"A Scoreline implies an Outcome; an Outcome does not imply a Scoreline" (CONTEXT.md),
        and the Season Projection needs the goals (ADR 0007)."""
        model = DixonColes(FLAT)
        evidence = Evidence.before(league, AS_OF)
        grid = model.scorelines(fixtures, evidence)

        assert grid.shape == (2, MAX_GOALS + 1, MAX_GOALS + 1)
        assert grid.sum(axis=(1, 2)) == pytest.approx([1.0, 1.0])

    def test_a_fixture_between_clubs_it_has_never_seen_is_uncertain_not_unanswerable(
        self, league: pd.DataFrame, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """Two Clubs promoted from outside the corpus meet: the model has nothing to say and says
        so with a Prediction near the neutral one, rather than raising."""
        predictions = DixonColes(FLAT).predict(
            make_matches({"date": "2024-08-24", "home_club": "yeovil", "away_club": "barnet"}),
            Evidence.before(league, AS_OF),
        )

        assert predictions.shape == (1, 3)
        assert predictions[0, 0] > predictions[0, 2]

    def test_its_strengths_can_be_asked_for_at_an_instant(
        self, league: pd.DataFrame
    ) -> None:
        found = DixonColes(FLAT).strengths_at(Evidence.before(league, AS_OF))

        assert found.table()["club"].iloc[0] == "arsenal"
        assert found.table()["club"].iloc[-1] == "wolves"

    def test_its_sample_says_what_reached_the_fit(self, league: pd.DataFrame) -> None:
        sample = DixonColes(FLAT).sample_at(Evidence.before(league, AS_OF))

        assert len(sample) == len(league)
        assert sample.clubs == ("arsenal", "chelsea", "luton", "wolves")


class TestWhatIsFrozen:
    def test_the_frozen_decay_is_a_literal_and_not_a_fit_at_import(self) -> None:
        """So a rebuilt backtest is byte-identical to the last one and no Prediction depends on
        whether ``data/raw/`` is populated (ADR 0005, ADR 0008)."""
        assert isinstance(FROZEN_DECAY.half_life_days, float)
        assert FROZEN_DECAY.half_life_days > 0

    def test_it_is_registered_under_its_own_name(self) -> None:
        from epl.models import DIXON_COLES

        assert DIXON_COLES.name == "dixon_coles"
        assert DIXON_COLES.decay == FROZEN_DECAY
