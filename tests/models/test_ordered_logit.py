"""The ordered logit: one number in, three probabilities out.

ADR 0006's whole claim is that the draw band comes for free. A latent match margin slides along a
line with two fitted cutpoints; the band between them has fixed width, so the share of the
distribution falling inside it narrows as Supremacy grows, with no hand-coded taper. These tests
are about that claim rather than about any particular fitted number.

Expected values are worked by hand from the logistic function, in the same spirit as
``tests/metrics/test_scores.py`` — a mapping verified against the code that produces it only proves
the code agrees with itself.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from epl.models import (
    DRAW_CURVE_COLUMNS,
    ModelError,
    OrderedLogit,
    draw_curve,
    sigmoid,
    supremacy,
)

#: Cutpoints at +/- ln 2 put sigma exactly on 2/3 and 1/3, so an evenly matched Fixture comes out
#: at a third each and the arithmetic can be checked without a calculator.
LN2 = math.log(2)

#: A logit whose scale is 100 rating points, so an edge of 100 is one unit of latent margin.
EVEN = OrderedLogit(scale=100.0, cutpoints=(-LN2, LN2))


class TestTheThreeProbabilities:
    def test_evenly_matched_clubs_split_the_band_symmetrically(self) -> None:
        """sigma(ln 2) = 2/3, so the draw band holds 2/3 - 1/3 and each tail holds 1/3."""
        home, draw, away = EVEN.probabilities([0.0])[0]

        assert (home, draw, away) == pytest.approx((1 / 3, 1 / 3, 1 / 3))

    def test_one_unit_of_margin_moves_the_line_along(self) -> None:
        """Worked by hand at edge 100, scale 100 — so the latent margin is exactly 1.

        P(Away) = sigma(-ln2 - 1) = 1 / (1 + 2e)         = 0.1553624
        P(Home) = 1 - sigma(ln2 - 1) = sigma(1 - ln2)    = 0.5761169
        P(Draw) = the rest                               = 0.2685207
        """
        home, draw, away = EVEN.probabilities([100.0])[0]

        assert (home, draw, away) == pytest.approx(
            (0.5761169, 0.2685207, 0.1553624), abs=1e-7
        )

    def test_a_negative_edge_mirrors_a_positive_one(self) -> None:
        """The away Club is stronger by exactly as much, so the distribution reverses."""
        home, draw, away = EVEN.probabilities([-100.0])[0]

        assert (home, draw, away) == pytest.approx((0.1553624, 0.2685207, 0.5761169), abs=1e-7)

    def test_every_prediction_sums_to_one(self) -> None:
        for probabilities in EVEN.probabilities([-800.0, -100.0, 0.0, 100.0, 800.0]):
            assert sum(probabilities) == pytest.approx(1.0)

    def test_it_answers_one_prediction_per_edge(self) -> None:
        assert EVEN.probabilities([0.0, 100.0, 200.0]).shape == (3, 3)

    def test_no_edges_is_no_predictions_rather_than_an_error(self) -> None:
        """A Prediction Round a Predictor covers nothing in still has to return a shape."""
        assert EVEN.probabilities([]).shape == (0, 3)


class TestTheDrawBand:
    def test_the_draw_probability_falls_as_supremacy_grows(self) -> None:
        """ADR 0006's claim, as a property rather than as a fitted number. How far it falls over a
        real corpus depends on the Predictor and is measured in
        ``tests/models/test_elo_over_the_corpus.py``; that it falls at all is structural, and this
        is the structural half."""
        predictions = EVEN.probabilities([0.0, 100.0, 200.0, 400.0, 800.0])
        draws = list(predictions[:, 1])

        assert draws == sorted(draws, reverse=True)
        assert all(earlier > later for earlier, later in itertools.pairwise(draws))

    def test_supremacy_grows_with_the_edge(self) -> None:
        """Supremacy is the gap between the Home and Away probabilities (CONTEXT.md), not the
        rating difference — so it has to be read off the Prediction."""
        gaps = list(supremacy(EVEN.probabilities([0.0, 100.0, 200.0, 400.0, 800.0])))

        assert gaps[0] == pytest.approx(0.0)
        assert gaps == sorted(gaps)

    def test_a_wider_band_draws_more_often(self) -> None:
        """The cutpoints are what set the draw rate, which is what fitting them is for."""
        narrow = OrderedLogit(scale=100.0, cutpoints=(-0.3, 0.3)).probabilities([0.0])[0][1]
        wide = OrderedLogit(scale=100.0, cutpoints=(-1.0, 1.0)).probabilities([0.0])[0][1]

        assert wide > narrow

    def test_a_larger_scale_flattens_the_response(self) -> None:
        """Scale is the rating points one unit of latent margin is worth. A bigger one makes the
        same rating gap say less, which is the parameter the Burn-In fit is really choosing."""
        steep = OrderedLogit(scale=50.0, cutpoints=(-LN2, LN2)).probabilities([100.0])[0][0]
        flat = OrderedLogit(scale=400.0, cutpoints=(-LN2, LN2)).probabilities([100.0])[0][0]

        assert steep > flat


class TestTheDrawCurve:
    """The receipt for ADR 0006, measurable over any Predictor's stored Predictions.

    It is what turns "the draw band narrows as Supremacy grows" from a claim about the arithmetic
    into a number a reader can check — against what the Predictor said *and* against what actually
    happened, which are different questions and are reported side by side.
    """

    def test_it_buckets_by_how_far_apart_the_clubs_are(self) -> None:
        """Four Fixtures into two buckets, by the magnitude of the gap — so a strong away Club
        lands beside an equally strong home one rather than at the opposite end."""
        predictions = EVEN.probabilities([0.0, 20.0, -600.0, 600.0])

        curve = draw_curve(predictions, ["D", "H", "A", "H"], buckets=2)

        assert list(curve["fixtures"]) == [2, 2]
        assert curve["mean_supremacy"].iloc[0] < curve["mean_supremacy"].iloc[1]

    def test_it_reports_what_was_predicted_beside_what_happened(self) -> None:
        """Both, because a taper that is predicted and not observed is a miscalibration and a
        taper that is observed and not predicted is a model with nothing to say."""
        predictions = EVEN.probabilities([0.0, 0.0, 900.0, 900.0])

        curve = draw_curve(predictions, ["D", "H", "D", "A"], buckets=2)

        assert list(curve["observed_draw"]) == pytest.approx([0.5, 0.5])
        assert curve["predicted_draw"].iloc[0] > curve["predicted_draw"].iloc[1]

    def test_the_predicted_draw_rate_falls_across_the_buckets(self) -> None:
        edges = [float(edge) for edge in range(-800, 801, 40)]
        outcomes = ["D" if index % 3 else "H" for index in range(len(edges))]

        curve = draw_curve(EVEN.probabilities(edges), outcomes)

        assert list(curve["predicted_draw"]) == sorted(curve["predicted_draw"], reverse=True)

    def test_a_predictor_that_says_one_thing_gets_one_bucket(self) -> None:
        """The Naive Baseline says the same thing about every Fixture, so there is nothing to
        order it by — one bucket is the honest answer rather than ten identical ones or a crash."""
        flat = np.tile([0.45, 0.25, 0.30], (20, 1))

        curve = draw_curve(flat, ["H"] * 10 + ["D"] * 10)

        assert len(curve) == 1
        assert curve["fixtures"].iloc[0] == 20

    def test_nothing_to_curve_is_an_empty_table_rather_than_an_error(self) -> None:
        curve = draw_curve(np.empty((0, 3)), [])

        assert len(curve) == 0
        assert list(curve.columns) == list(DRAW_CURVE_COLUMNS)


class TestTheLogisticItself:
    def test_it_is_a_half_at_zero(self) -> None:
        assert sigmoid([0.0])[0] == pytest.approx(0.5)

    def test_a_margin_too_large_to_exponentiate_does_not_overflow(self) -> None:
        """exp(710) is inf and inf/inf is nan, which would reach the ledger as a Prediction that
        does not sum to one. The Burn-In fit searches over exp(log scale) and can put a candidate
        scale near zero on its way past, which is what makes this reachable rather than academic.
        """
        saturated = sigmoid([-2000.0, -800.0, 800.0, 2000.0])

        assert np.isfinite(saturated).all()
        assert list(saturated) == pytest.approx([0.0, 0.0, 1.0, 1.0])

    def test_a_saturated_edge_still_gives_a_prediction_that_sums_to_one(self) -> None:
        tiny_scale = OrderedLogit(scale=1e-6, cutpoints=(-LN2, LN2))

        assert tiny_scale.probabilities([-5.0, 0.0, 5.0]).sum(axis=1) == pytest.approx(
            [1.0, 1.0, 1.0]
        )


class TestItRefusesToBeBuiltWrong:
    def test_cutpoints_must_be_ordered(self) -> None:
        """Reversed cutpoints give a negative draw probability — an ordinal model that has lost
        the ordinal structure RPS depends on."""
        with pytest.raises(ModelError, match="cutpoints"):
            OrderedLogit(scale=100.0, cutpoints=(LN2, -LN2))

    def test_cutpoints_must_differ(self) -> None:
        with pytest.raises(ModelError, match="cutpoints"):
            OrderedLogit(scale=100.0, cutpoints=(LN2, LN2))

    def test_the_scale_must_be_positive(self) -> None:
        with pytest.raises(ModelError, match="scale"):
            OrderedLogit(scale=0.0, cutpoints=(-LN2, LN2))
