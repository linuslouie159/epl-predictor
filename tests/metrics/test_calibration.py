"""The reliability diagram: when a Predictor says 60%, does it happen 60% of the time?

Every expected value below is worked out by hand from the pooled (probability, what-happened)
pairs, never from the implementation.
"""

from __future__ import annotations

import pytest

from epl import metrics

CERTAIN_HOME = [1.0, 0.0, 0.0]


class TestReliabilityShape:
    def test_uses_ten_bins(self) -> None:
        """Ten, per the spec. Diagrams from different Predictors must be directly comparable."""
        assert len(metrics.reliability([CERTAIN_HOME], ["H"])) == 10
        assert metrics.BINS == 10

    def test_keeps_empty_bins_so_every_diagram_has_the_same_shape(self) -> None:
        table = metrics.reliability([CERTAIN_HOME], ["H"])
        assert (table["predictions"] == 0).any()
        assert len(table) == 10

    def test_bins_span_zero_to_one_in_equal_steps(self) -> None:
        table = metrics.reliability([CERTAIN_HOME], ["H"])
        assert list(table["lower"]) == pytest.approx([i / 10 for i in range(10)])
        assert list(table["upper"]) == pytest.approx([(i + 1) / 10 for i in range(10)])

    def test_names_the_columns_a_diagram_needs(self) -> None:
        table = metrics.reliability([CERTAIN_HOME], ["H"])
        assert list(table.columns) == [
            "lower",
            "upper",
            "predictions",
            "mean_predicted",
            "observed",
            "gap",
        ]

    def test_pools_all_three_outcomes_of_every_prediction(self) -> None:
        """Each Prediction contributes three points, one per Outcome."""
        assert metrics.reliability([[0.5, 0.3, 0.2]], ["H"])["predictions"].sum() == 3


class TestReliabilityByHand:
    """Four identical Predictions of (0.5, 0.3, 0.2) on Outcomes H, H, D, A.

    Pooled into twelve points:
        p = 0.5 (Home)  x 4, Home happened twice        -> bin [0.5, 0.6), observed 0.50
        p = 0.3 (Draw)  x 4, a Draw happened once       -> bin [0.3, 0.4), observed 0.25
        p = 0.2 (Away)  x 4, an Away win happened once  -> bin [0.2, 0.3), observed 0.25
    """

    @pytest.fixture
    def table(self):
        return metrics.reliability([[0.5, 0.3, 0.2]] * 4, ["H", "H", "D", "A"])

    def test_counts_four_points_in_each_of_the_three_occupied_bins(self, table) -> None:
        assert list(table.loc[[2, 3, 5], "predictions"]) == [4, 4, 4]

    def test_every_other_bin_is_empty(self, table) -> None:
        assert table["predictions"].sum() == 12
        assert list(table.drop(index=[2, 3, 5])["predictions"]) == [0] * 7

    def test_reports_the_mean_probability_promised_in_each_bin(self, table) -> None:
        assert list(table.loc[[2, 3, 5], "mean_predicted"]) == pytest.approx([0.2, 0.3, 0.5])

    def test_reports_how_often_it_actually_happened(self, table) -> None:
        assert list(table.loc[[2, 3, 5], "observed"]) == pytest.approx([0.25, 0.25, 0.5])

    def test_the_gap_is_observed_minus_promised(self, table) -> None:
        assert list(table.loc[[2, 3, 5], "gap"]) == pytest.approx([0.05, -0.05, 0.0])

    def test_an_empty_bin_reports_no_gap_rather_than_a_zero_one(self, table) -> None:
        """A zero gap in an unoccupied bin would read as perfect calibration there."""
        assert table.loc[0, "observed"] != table.loc[0, "observed"]  # NaN
        assert table.loc[0, "gap"] != table.loc[0, "gap"]


class TestExpectedCalibrationError:
    """The reliability diagram as one number: the count-weighted mean absolute gap."""

    def test_a_perfectly_calibrated_predictor_scores_zero(self) -> None:
        """(0.5, 0.3, 0.2) ten times, and Home/Draw/Away come in exactly 5/3/2."""
        outcomes = ["H"] * 5 + ["D"] * 3 + ["A"] * 2
        error = metrics.expected_calibration_error([[0.5, 0.3, 0.2]] * 10, outcomes)
        assert error == pytest.approx(0.0, abs=1e-12)

    def test_the_four_fixture_example_by_hand(self) -> None:
        """(4 x 0.05 + 4 x 0.05 + 4 x 0.00) / 12 = 0.4 / 12."""
        error = metrics.expected_calibration_error([[0.5, 0.3, 0.2]] * 4, ["H", "H", "D", "A"])
        assert error == pytest.approx(0.4 / 12)

    def test_an_overconfident_predictor_by_hand(self) -> None:
        """Certain Home ten times; Home comes in six.

        Bin [0.9, 1.0]: 10 points promised 1.00, observed 0.60 -> gap 0.40
        Bin [0.0, 0.1): 20 points promised 0.00 (the Draws and Away wins), observed 0.20
        ECE = (10 x 0.40 + 20 x 0.20) / 30 = 8 / 30
        """
        outcomes = ["H"] * 6 + ["A"] * 4
        error = metrics.expected_calibration_error([CERTAIN_HOME] * 10, outcomes)
        assert error == pytest.approx(8 / 30)

    def test_it_ignores_empty_bins_rather_than_counting_them_as_perfect(self) -> None:
        outcomes = ["H"] * 5 + ["D"] * 3 + ["A"] * 2
        table = metrics.reliability([[0.5, 0.3, 0.2]] * 10, outcomes)
        assert (table["predictions"] == 0).sum() == 7
        error = metrics.expected_calibration_error([[0.5, 0.3, 0.2]] * 10, outcomes)
        assert error == pytest.approx(0.0, abs=1e-12)

    def test_it_refuses_an_empty_slate(self) -> None:
        with pytest.raises(metrics.MetricsError, match="no Fixtures"):
            metrics.expected_calibration_error([], [])


class TestBinEdges:
    def test_a_probability_of_one_falls_in_the_top_bin_not_off_the_end(self) -> None:
        table = metrics.reliability([CERTAIN_HOME], ["H"])
        assert table.loc[9, "predictions"] == 1

    def test_a_probability_of_zero_falls_in_the_bottom_bin(self) -> None:
        table = metrics.reliability([CERTAIN_HOME], ["H"])
        assert table.loc[0, "predictions"] == 2

    def test_a_probability_on_a_boundary_falls_in_the_bin_it_opens(self) -> None:
        """0.3 belongs to [0.3, 0.4), not to [0.2, 0.3)."""
        table = metrics.reliability([[0.3, 0.3, 0.4]], ["H"])
        assert table.loc[3, "predictions"] == 2
        assert table.loc[2, "predictions"] == 0
