"""The Dixon-Coles likelihood, its gradient, and the Scoreline grid it produces.

ADR 0007 makes this module the one thing the MLE fit and the Bayesian fit share, so a defect here
reaches both and a test here is worth two. Everything is checked against arithmetic worked by hand
or against a property that has to hold whatever the parameters are:

* the rates, the correction and the Scoreline grid against numbers computed on paper
* the analytic gradient against central differences of the very function it belongs to, which is
  the only honest way to keep a hand-written derivative true
* the gauge freedom — adding a constant to every attack and every defence — against everything the
  model produces, because a fit that did not have it would be a different model

Nothing here fits anything or reads the corpus. ``tests/models/test_dixon_coles.py`` is the fit and
``tests/models/test_dixon_coles_over_the_corpus.py`` is the walk over real football.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from epl.models import ModelError
from epl.models.likelihood import (
    Decay,
    Sample,
    Strengths,
    low_score_factor,
    negative_log_likelihood,
    outcomes,
    pack,
    scorelines,
    start,
    unpack,
)

#: An As-Of Instant every sample in this file is weighted against.
AS_OF = pd.Timestamp("2024-08-20")

#: A decay that halves every ten days, so a hand-worked weight is a power of two.
TEN_DAYS = Decay(half_life_days=10.0)


def strengths(
    clubs: tuple[str, ...] = ("arsenal", "chelsea"),
    attack: tuple[float, ...] = (0.0, 0.0),
    defence: tuple[float, ...] = (0.0, 0.0),
    home_advantage: float = 0.0,
    correction_value: float = 0.0,
) -> Strengths:
    return Strengths(
        clubs=clubs,
        attack=np.array(attack, dtype=float),
        defence=np.array(defence, dtype=float),
        home_advantage=home_advantage,
        correction=correction_value,
    )


class TestTheTimeDecay:
    def test_a_match_played_at_the_instant_counts_in_full(self) -> None:
        assert TEN_DAYS.weights([0.0])[0] == pytest.approx(1.0)

    def test_a_match_one_half_life_old_counts_half(self) -> None:
        assert TEN_DAYS.weights([10.0])[0] == pytest.approx(0.5)

    def test_it_keeps_halving(self) -> None:
        assert TEN_DAYS.weights([20.0, 30.0]) == pytest.approx([0.25, 0.125])

    def test_the_horizon_is_where_the_weight_reaches_the_floor(self) -> None:
        """A floor of one in a hundred is between six and seven half-lives back."""
        decay = Decay(half_life_days=10.0, floor=0.01)

        assert decay.horizon == pytest.approx(10.0 * math.log2(100.0))
        assert decay.weights([decay.horizon])[0] == pytest.approx(0.01)

    def test_a_half_life_of_zero_is_a_model_that_has_seen_nothing(self) -> None:
        with pytest.raises(ModelError, match="half-life must be positive"):
            Decay(half_life_days=0.0)

    def test_a_floor_outside_zero_to_one_is_refused(self) -> None:
        with pytest.raises(ModelError, match="weight floor"):
            Decay(half_life_days=10.0, floor=1.0)


class TestBuildingASample:
    def test_it_weights_each_match_by_how_long_ago_it_was(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        sample = Sample.of(
            make_matches(
                {"date": "2024-08-20"},
                {"date": "2024-08-10"},
                {"date": "2024-07-31"},
            ),
            AS_OF + pd.Timedelta(days=1),
            TEN_DAYS,
        )

        assert sample.weight == pytest.approx([0.5**0.1, 0.5**1.1, 0.5**2.1])

    def test_matches_past_the_horizon_are_dropped(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        decay = Decay(half_life_days=10.0, floor=0.25)
        sample = Sample.of(
            make_matches({"date": "2024-08-10"}, {"date": "2024-07-01"}),
            AS_OF,
            decay,
        )

        assert len(sample) == 1

    def test_clubs_are_indexed_in_name_order(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """Name order rather than order of appearance, so a rebuild is byte-identical (ADR 0005)."""
        sample = Sample.of(
            make_matches(
                {"home_club": "wolves", "away_club": "arsenal", "date": "2024-08-19"},
                {"home_club": "chelsea", "away_club": "wolves", "date": "2024-08-19"},
            ),
            AS_OF,
            TEN_DAYS,
        )

        assert sample.clubs == ("arsenal", "chelsea", "wolves")
        assert sample.home.tolist() == [2, 1]
        assert sample.away.tolist() == [0, 2]

    def test_a_club_named_in_also_gets_a_slot_it_has_no_matches_for(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """The Clubs of the Fixtures about to be predicted, so a Fixture is uncertain rather than
        unanswerable."""
        sample = Sample.of(
            make_matches({"date": "2024-08-19"}), AS_OF, TEN_DAYS, also=["luton"]
        )

        assert "luton" in sample.clubs
        assert sample.index_of(["luton"]).tolist() == [sample.clubs.index("luton")]

    def test_a_club_with_no_slot_is_refused_rather_than_guessed_at(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        sample = Sample.of(make_matches({"date": "2024-08-19"}), AS_OF, TEN_DAYS)

        with pytest.raises(ModelError, match="no slot in this sample"):
            sample.index_of(["luton"])

    def test_a_match_at_or_after_the_as_of_instant_is_refused(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """Evidence is what cuts the corpus; this refuses to be the second place that rule lives."""
        with pytest.raises(ModelError, match="at or after"):
            Sample.of(make_matches({"date": "2024-08-20"}), AS_OF, TEN_DAYS)

    def test_an_empty_corpus_gives_an_empty_sample(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        sample = Sample.of(make_matches(), AS_OF, TEN_DAYS)

        assert len(sample) == 0
        assert sample.clubs == ()


class TestTheRates:
    def test_two_average_clubs_with_no_home_advantage_both_score_one(self) -> None:
        home, away = strengths().rates(np.array([0]), np.array([1]))

        assert home == pytest.approx([1.0])
        assert away == pytest.approx([1.0])

    def test_home_advantage_lifts_only_the_home_rate(self) -> None:
        home, away = strengths(home_advantage=math.log(2.0)).rates(np.array([0]), np.array([1]))

        assert home == pytest.approx([2.0])
        assert away == pytest.approx([1.0])

    def test_a_strong_attack_against_a_weak_defence_multiplies(self) -> None:
        """log 3 of attack against log 2 of defence conceded is 3 x 2 = 6 goals expected."""
        model = strengths(attack=(math.log(3.0), 0.0), defence=(0.0, -math.log(2.0)))
        home, _ = model.rates(np.array([0]), np.array([1]))

        assert home == pytest.approx([6.0])

    def test_adding_a_constant_to_every_strength_changes_nothing(self) -> None:
        """The model's one flat direction, which is why a fit has to be put in a gauge."""
        plain = strengths(attack=(0.3, -0.1), defence=(0.2, 0.4), home_advantage=0.25)
        shifted = strengths(
            attack=(1.3, 0.9), defence=(1.2, 1.4), home_advantage=0.25
        )
        home, away = plain.rates(np.array([0]), np.array([1]))
        shifted_home, shifted_away = shifted.rates(np.array([0]), np.array([1]))

        assert home == pytest.approx(shifted_home)
        assert away == pytest.approx(shifted_away)

    def test_centring_puts_the_average_attack_at_zero_without_moving_a_rate(self) -> None:
        model = strengths(attack=(1.3, 0.9), defence=(1.2, 1.4), home_advantage=0.25)
        centred = model.centred()
        home, away = model.rates(np.array([0]), np.array([1]))
        centred_home, centred_away = centred.rates(np.array([0]), np.array([1]))

        assert centred.attack.mean() == pytest.approx(0.0)
        assert centred_home == pytest.approx(home)
        assert centred_away == pytest.approx(away)

    def test_a_strengths_needs_one_attack_and_one_defence_per_club(self) -> None:
        with pytest.raises(ModelError, match="one per Club"):
            Strengths(("arsenal", "chelsea"), np.zeros(2), np.zeros(1), 0.0, 0.0)


class TestTheLowScoreCorrection:
    def test_it_is_one_everywhere_the_correction_is_zero(self) -> None:
        """Rho of zero is two independent Poissons, which is the model without Dixon-Coles."""
        factor = low_score_factor(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 3.0]),
            np.array([1.4, 1.4, 1.4]),
            np.array([1.1, 1.1, 1.1]),
            0.0,
        )

        assert factor == pytest.approx([1.0, 1.0, 1.0])

    def test_it_is_one_on_every_scoreline_but_the_four(self) -> None:
        factor = low_score_factor(
            np.array([2.0, 0.0, 1.0, 3.0]),
            np.array([0.0, 2.0, 2.0, 3.0]),
            np.full(4, 1.5),
            np.full(4, 1.2),
            -0.1,
        )

        assert factor == pytest.approx([1.0, 1.0, 1.0, 1.0])

    def test_a_negative_correction_lifts_the_two_low_draws_and_lowers_the_two_wins(self) -> None:
        """The direction the data has always given: more 0-0 and 1-1, fewer 1-0 and 0-1."""
        factor = low_score_factor(
            np.array([0.0, 1.0, 1.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 1.0]),
            np.full(4, 1.5),
            np.full(4, 1.2),
            -0.1,
        )

        assert factor[0] > 1.0 and factor[1] > 1.0
        assert factor[2] < 1.0 and factor[3] < 1.0

    def test_each_of_the_four_is_the_published_expression(self) -> None:
        """1 - lambda.mu.rho, 1 + lambda.rho, 1 + mu.rho, 1 - rho (Dixon and Coles 1997)."""
        home_rate, away_rate, rho = 1.5, 1.2, -0.1
        factor = low_score_factor(
            np.array([0.0, 0.0, 1.0, 1.0]),
            np.array([0.0, 1.0, 0.0, 1.0]),
            np.full(4, home_rate),
            np.full(4, away_rate),
            rho,
        )

        assert factor == pytest.approx(
            [
                1.0 - home_rate * away_rate * rho,
                1.0 + home_rate * rho,
                1.0 + away_rate * rho,
                1.0 - rho,
            ]
        )


class TestTheScorelineGrid:
    def test_every_fixture_sums_to_one(self) -> None:
        grid = scorelines(np.array([1.4, 3.0]), np.array([1.1, 0.4]), -0.1)

        assert grid.sum(axis=(1, 2)) == pytest.approx([1.0, 1.0])

    def test_with_no_correction_it_is_two_independent_poissons(self) -> None:
        """2-1 at rates 1.5 and 1.2 is e^-1.5 1.5^2/2 x e^-1.2 1.2."""
        grid = scorelines(np.array([1.5]), np.array([1.2]), 0.0)
        expected = (math.exp(-1.5) * 1.5**2 / 2) * (math.exp(-1.2) * 1.2)

        assert grid[0, 2, 1] == pytest.approx(expected, rel=1e-9)

    def test_the_correction_moves_probability_without_creating_any(self) -> None:
        """Its four adjustments cancel exactly, so only the truncated tail is renormalised away."""
        plain = scorelines(np.array([1.5]), np.array([1.2]), 0.0, max_goals=40)
        corrected = scorelines(np.array([1.5]), np.array([1.2]), -0.15, max_goals=40)

        assert corrected.sum() == pytest.approx(plain.sum(), rel=1e-12)
        assert corrected[0, 0, 0] > plain[0, 0, 0]
        assert corrected[0, 1, 0] < plain[0, 1, 0]

    def test_truncating_at_fifteen_goals_loses_almost_nothing(self) -> None:
        coarse = scorelines(np.array([2.5]), np.array([2.0]), -0.1, max_goals=15)
        fine = scorelines(np.array([2.5]), np.array([2.0]), -0.1, max_goals=40)

        assert coarse[0, :16, :16] == pytest.approx(fine[0, :16, :16], abs=1e-9)

    def test_no_fixture_gives_an_empty_grid(self) -> None:
        grid = scorelines(np.empty(0), np.empty(0), -0.1)

        assert grid.shape == (0, 16, 16)


class TestCollapsingOntoOutcomes:
    def test_it_is_the_grid_partitioned_by_the_diagonal(self) -> None:
        grid = scorelines(np.array([1.6]), np.array([1.1]), -0.1)
        three = outcomes(grid)

        above = sum(
            grid[0, home, away]
            for home in range(grid.shape[1])
            for away in range(grid.shape[2])
            if home > away
        )
        assert three[0, 0] == pytest.approx(above)
        assert three.sum() == pytest.approx(1.0)

    def test_evenly_matched_clubs_are_as_likely_to_win_as_to_lose(self) -> None:
        three = outcomes(scorelines(np.array([1.3]), np.array([1.3]), -0.1))

        assert three[0, 0] == pytest.approx(three[0, 2])

    def test_the_stronger_home_club_is_favourite(self) -> None:
        three = outcomes(scorelines(np.array([2.2]), np.array([0.8]), -0.1))

        assert three[0, 0] > three[0, 1] > three[0, 2]

    def test_no_fixture_gives_no_predictions(self) -> None:
        assert outcomes(scorelines(np.empty(0), np.empty(0), 0.0)).shape == (0, 3)


class TestTheLikelihood:
    def sample(self, make_matches: Callable[..., pd.DataFrame], **decay: object) -> Sample:
        return Sample.of(
            make_matches(
                {"date": "2024-08-19", "home_goals": 2, "away_goals": 1},
                {
                    "date": "2024-08-13",
                    "home_club": "chelsea",
                    "away_club": "arsenal",
                    "home_goals": 0,
                    "away_goals": 0,
                },
                {"date": "2024-08-06", "home_goals": 1, "away_goals": 1},
            ),
            AS_OF,
            Decay(half_life_days=float(decay.get("half_life_days", 10.0))),
        )

    def test_at_average_strengths_it_is_the_weighted_poisson_sum(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """Every rate is one, so each match contributes w(x.0 - 1 + y.0 - 1) = -2w."""
        sample = self.sample(make_matches)
        value, _ = negative_log_likelihood(pack(strengths()), sample)

        assert value == pytest.approx(2.0 * sample.weight.sum())

    def test_the_factorial_terms_are_left_out(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """Stated in the docstring, and worth pinning: the value is not comparable to another
        implementation's without adding log(x!) + log(y!) back.

        One 2-1 at rates of one apiece. The full log-likelihood is -2 - log(2!); this drops the
        second term, so a value of exactly 2 is the constant being left out.
        """
        sample = Sample.of(
            make_matches({"date": "2024-08-20", "home_goals": 2, "away_goals": 1}),
            AS_OF + pd.Timedelta(days=1),
            Decay(half_life_days=1e9),
        )
        value, _ = negative_log_likelihood(pack(strengths()), sample)

        assert value == pytest.approx(2.0)
        assert value != pytest.approx(2.0 + math.log(2.0))

    def test_weights_scale_it(self, make_matches: Callable[..., pd.DataFrame]) -> None:
        near = self.sample(make_matches, half_life_days=1e6)
        far = self.sample(make_matches, half_life_days=10.0)
        flat = pack(strengths(home_advantage=0.2, correction_value=-0.1))

        assert negative_log_likelihood(flat, near)[0] > negative_log_likelihood(flat, far)[0]

    @pytest.mark.parametrize(
        "model",
        [
            strengths(),
            strengths(attack=(0.4, -0.2), defence=(0.1, -0.3), home_advantage=0.25),
            strengths(
                attack=(0.4, -0.2),
                defence=(0.1, -0.3),
                home_advantage=0.25,
                correction_value=-0.12,
            ),
            strengths(
                attack=(-0.6, 0.7),
                defence=(0.5, -0.4),
                home_advantage=-0.1,
                correction_value=0.08,
            ),
        ],
        ids=["neutral", "asymmetric", "corrected", "corrected the other way"],
    )
    def test_the_analytic_gradient_matches_central_differences(
        self, make_matches: Callable[..., pd.DataFrame], model: Strengths
    ) -> None:
        """The only honest way to keep a hand-written derivative true."""
        sample = self.sample(make_matches)
        free = pack(model)
        _, analytic = negative_log_likelihood(free, sample)

        step = 1e-6
        numeric = np.empty_like(free)
        for index in range(len(free)):
            up, down = free.copy(), free.copy()
            up[index] += step
            down[index] -= step
            numeric[index] = (
                negative_log_likelihood(up, sample)[0]
                - negative_log_likelihood(down, sample)[0]
            ) / (2 * step)

        assert analytic == pytest.approx(numeric, abs=1e-6)

    def test_the_gauge_direction_is_flat(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """Adding a constant to every attack and every defence must not change the likelihood —
        which is exactly why a fitted Strengths is meaningless until it is centred."""
        sample = self.sample(make_matches)
        plain = strengths(attack=(0.4, -0.2), defence=(0.1, -0.3), home_advantage=0.25)
        shifted = strengths(attack=(1.4, 0.8), defence=(1.1, 0.7), home_advantage=0.25)

        assert negative_log_likelihood(pack(plain), sample)[0] == pytest.approx(
            negative_log_likelihood(pack(shifted), sample)[0]
        )

    def test_the_true_strengths_beat_wrong_ones(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """A sample of goalless draws is better explained by weak attacks than by strong ones."""
        sample = Sample.of(
            make_matches(
                *[
                    {"date": "2024-08-19", "home_goals": 0, "away_goals": 0}
                    for _ in range(10)
                ]
            ),
            AS_OF,
            TEN_DAYS,
        )
        weak = pack(strengths(attack=(-1.0, -1.0)))
        strong = pack(strengths(attack=(1.0, 1.0)))

        assert negative_log_likelihood(weak, sample)[0] < negative_log_likelihood(strong, sample)[0]


class TestWhereAFitStarts:
    def test_it_puts_every_club_at_the_samples_own_scoring_rates(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        """Two goals at home and one away every time, so the start should quote exactly that."""
        sample = Sample.of(
            make_matches(
                *[
                    {"date": "2024-08-19", "home_goals": 2, "away_goals": 1}
                    for _ in range(4)
                ]
            ),
            AS_OF,
            TEN_DAYS,
        )
        home, away = start(sample).rates(np.array([0]), np.array([1]))

        assert home == pytest.approx([2.0])
        assert away == pytest.approx([1.0])

    def test_it_starts_with_no_correction_so_every_scoreline_is_positive(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        sample = Sample.of(make_matches({"date": "2024-08-19"}), AS_OF, TEN_DAYS)

        assert start(sample).correction == 0.0

    def test_an_empty_sample_starts_neutral(
        self, make_matches: Callable[..., pd.DataFrame]
    ) -> None:
        assert start(Sample.of(make_matches(), AS_OF, TEN_DAYS)).clubs == ()


class TestPacking:
    def test_it_round_trips(self) -> None:
        model = strengths(
            attack=(0.4, -0.2), defence=(0.1, -0.3), home_advantage=0.25, correction_value=-0.12
        )
        back = unpack(pack(model), model.clubs)

        assert back.attack == pytest.approx(model.attack)
        assert back.defence == pytest.approx(model.defence)
        assert back.home_advantage == pytest.approx(model.home_advantage)
        assert back.correction == pytest.approx(model.correction)
