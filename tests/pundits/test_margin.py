"""The margin map: what a Scoreline of a given goal margin has historically been worth.

Every expected rate here is worked by hand from the calls the test names, in the style
`tests/metrics/` established — a map verified against the code that produced it only proves the
code agrees with itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from epl.pundits import margin
from epl.pundits.margin import MarginMapError


def a_map(calls: dict[int, str], *, minimum: int = 4) -> margin.MarginMap:
    """A map fitted on calls named as ``{margin: outcomes}`` — ``{2: "HHDA"}`` is four 2-0 calls.

    The shorthand is what makes a hand-worked rate readable: the test names the sample and the
    assertion names the rate it implies, with nothing in between.
    """
    margins = [value for value, letters in calls.items() for _ in letters]
    outcomes = [letter for letters in calls.values() for letter in letters]
    return margin.fit(margins, outcomes, minimum=minimum)


class TestOneBucketsRate:
    def test_a_bucket_quotes_how_often_that_margin_turned_out_each_way(self) -> None:
        """Four calls of the same margin, two Home, one Draw, one Away — 0.5, 0.25, 0.25."""
        fitted = a_map({1: "HHDA"})

        assert fitted.quote([1]).tolist() == [[0.5, 0.25, 0.25]]

    def test_every_quote_sums_to_one(self) -> None:
        fitted = a_map({-1: "AAAH", 0: "DDHA", 1: "HHHD"})

        assert np.allclose(fitted.quote([-1, 0, 1]).sum(axis=1), 1.0)

    def test_a_margin_nobody_ever_called_wrong_is_quoted_at_certainty(self) -> None:
        """Not clipped. A bucket of four calls that all went Home really did go Home every time,
        and softening it here would be a prior nobody fitted. It is thin samples that make this
        possible, which is what :data:`epl.pundits.margin.MINIMUM_SAMPLE` is for."""
        fitted = a_map({3: "HHHH"})

        assert fitted.quote([3]).tolist() == [[1.0, 0.0, 0.0]]

    def test_the_quotes_come_back_in_the_order_the_margins_were_asked_in(self) -> None:
        fitted = a_map({-1: "AAAA", 1: "HHHH"})

        assert fitted.quote([1, -1, 1]).tolist() == [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]

    def test_an_empty_slate_still_comes_back_as_three_columns(self) -> None:
        """A round a Pundit called nothing in is a shape, not a special case."""
        assert a_map({1: "HHHH"}).quote([]).shape == (0, 3)


class TestTheStrongerClaim:
    def test_a_three_nil_call_is_not_the_same_claim_as_a_two_one(self) -> None:
        """Issue #12's first acceptance criterion, and the whole reason this map is bucketed by
        margin rather than by the Outcome the Scoreline implies. Both calls are Home; only one of
        them is a claim that Home wins nine times in ten."""
        fitted = a_map({1: "HHDA", 3: "HHHD"})

        two_one, three_nil = fitted.quote([1, 3])

        assert two_one.tolist() == [0.5, 0.25, 0.25]
        assert three_nil.tolist() == [0.75, 0.25, 0.0]

    def test_the_same_margin_from_different_scorelines_is_one_bucket(self) -> None:
        """3-1 and 2-0 are both a two-goal claim. The map sees the margin and nothing else, which
        is what makes it the one-feature model ADR 0003 asks for rather than a scoreline model."""
        fitted = a_map({2: "HHHA"})

        assert fitted.quote([2]).tolist() == fitted.quote([2]).tolist()
        assert fitted.buckets[0].margins == (2,)
        assert fitted.buckets[0].calls == 4


class TestThinBucketsMergeTowardZero:
    def test_a_thin_extreme_merges_into_its_neighbour_nearer_zero(self) -> None:
        """Two 3-0 calls cannot carry a rate of their own, so they are read with the 2-0s: six
        calls, five Home, one Away."""
        fitted = a_map({2: "HHHA", 3: "HH"})

        assert fitted.quote([3]).tolist() == fitted.quote([2]).tolist()
        assert fitted.quote([3]).tolist() == [[5 / 6, 0.0, 1 / 6]]

    def test_merging_stops_at_the_first_bucket_that_has_enough(self) -> None:
        """The 4s merge into the 3s and stop there — they do not go on to swallow the 2s."""
        fitted = a_map({2: "AAAA", 3: "HHHH", 4: "HH"})

        assert fitted.quote([4]).tolist() == [[1.0, 0.0, 0.0]]
        assert fitted.quote([2]).tolist() == [[0.0, 0.0, 1.0]]

    def test_a_side_that_never_fills_a_bucket_falls_back_to_the_pooled_rate(self) -> None:
        """Three away calls between them, against a minimum of four. There is no away bucket to
        merge into, so such a call is quoted what this Pundit's calls do on average — which says
        "this record cannot yet tell you what an away call of theirs is worth"."""
        fitted = a_map({-2: "A", -1: "AH", 0: "DDDD", 1: "HHHH"})

        assert fitted.quote([-1]).tolist() == [[5 / 11, 4 / 11, 2 / 11]]

    def test_the_two_sides_merge_independently(self) -> None:
        """A thin away side does not reach across zero and take the home calls with it."""
        fitted = a_map({-2: "AA", -1: "AAAA", 1: "HHHH"})

        assert fitted.quote([-2]).tolist() == [[0.0, 0.0, 1.0]]
        assert fitted.quote([1]).tolist() == [[1.0, 0.0, 0.0]]

    def test_a_thin_draw_call_falls_back_rather_than_picking_a_side(self) -> None:
        """Margin zero has no neighbour nearer zero, and choosing Home or Away for it would be
        the map inventing a lean the calls do not have."""
        fitted = a_map({-1: "AAAA", 0: "DD", 1: "HHHH"})

        assert fitted.quote([0]).tolist() == [[0.4, 0.2, 0.4]]


class TestAMarginNeverCalledBefore:
    def test_a_margin_beyond_anything_called_before_is_read_as_the_widest_one_that_was(
        self,
    ) -> None:
        """A first 5-0 call arrives with no 5-0s behind it. Quoting the pooled rate would read the
        boldest call in the record as an average one; the widest bucket there is says more."""
        fitted = a_map({1: "HHDA", 2: "HHHH"})

        assert fitted.quote([5]).tolist() == fitted.quote([2]).tolist()

    def test_a_gap_between_called_margins_is_read_from_the_side_nearer_zero(self) -> None:
        """Same rule, applied inside the range rather than past its end."""
        fitted = a_map({1: "HHDA", 3: "HHHH"})

        assert fitted.quote([2]).tolist() == fitted.quote([1]).tolist()

    def test_a_margin_on_a_side_nobody_has_called_at_all_is_the_pooled_rate(self) -> None:
        fitted = a_map({0: "DDDD", 1: "HHHH"})

        assert fitted.quote([-3]).tolist() == [[0.5, 0.5, 0.0]]


class TestRefusingToSpeak:
    def test_a_map_needs_the_minimum_before_it_is_fitted_at_all(self) -> None:
        """Below it there is no pooled rate to fall back on either, so a map fitted here would be
        handing back the Outcome that happened as though it were a probability. `covers` is what
        keeps such a Fixture off the walk (:mod:`epl.pundits.calibrated`)."""
        with pytest.raises(MarginMapError, match=r"needs 4 past calls .* and has 3"):
            a_map({1: "HHD"}, minimum=4)

    def test_nothing_to_fit_on_is_refused_in_the_same_terms(self) -> None:
        with pytest.raises(MarginMapError):
            margin.fit([], [], minimum=4)

    def test_a_margin_for_every_outcome_or_neither(self) -> None:
        with pytest.raises(MarginMapError, match="4 margins against 3 Outcomes"):
            margin.fit([1, 1, 1, 1], ["H", "H", "D"], minimum=4)


class TestTheMapAsAReport:
    def test_the_table_names_every_bucket_its_calls_and_its_rates(self) -> None:
        """The map is published as a plain file, so a reader can see what a 3-0 call is worth
        without running anything (issue #12's seventh acceptance criterion)."""
        table = a_map({-1: "AAAA", 0: "DDDD", 2: "HHHA"}).table()

        assert list(table.columns) == list(margin.MAP_COLUMNS)
        assert table["margins"].tolist() == ["-1", "0", "2", margin.POOLED]
        assert table["calls"].tolist() == [4, 4, 4, 12]
        assert table["prob_home"].tolist() == [0.0, 0.0, 0.75, 0.25]

    def test_a_merged_bucket_names_every_margin_in_it(self) -> None:
        table = a_map({2: "HHHA", 3: "HH", 4: "H"}).table()

        assert table["margins"].tolist() == ["2, 3, 4", margin.POOLED]
        assert table["calls"].tolist() == [7, 7]

    def test_the_pooled_fallback_is_the_last_row_and_is_always_there(self) -> None:
        """It is what a margin with no bucket behind it is quoted, and it is this Pundit's own base
        rate — the thing every bucket above it should be read against."""
        table = a_map({-1: "AH", 0: "DDDD", 1: "HHHH"}).table()

        assert table["margins"].tolist() == ["0", "1", margin.POOLED]
        assert table.loc[table["margins"] == margin.POOLED, "calls"].tolist() == [10]
