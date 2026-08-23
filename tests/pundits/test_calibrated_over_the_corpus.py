"""The Calibrated Pundit against the real calls and the real corpus.

This is where issue #12 is discharged. The unit tests say the margin map does what it says on a
dozen hand-worked calls; these say what it is actually worth on nine Seasons of real ones, and pin
every number the docs quote so none of them can drift in silence.

Three findings live here and each is re-derived rather than asserted:

* **The cost of stating certainty is about 0.12 RPS.** That is what being asked for a Scoreline
  instead of a probability charged these two forecasters (ADR 0003).
* **The margin map clears the shared layer's bar comfortably.** `epl.calibration` takes an
  as-stated Pundit from 0.334 to 0.237 and 0.247; reading the Scoreline itself gets to 0.213 and
  0.211, because the shared layer sees a one-hot input with no Scoreline left in it and this one
  sees the margin.
* **The map is monotone in the margin without being made to be.** Nothing in
  :mod:`epl.pundits.margin` enforces it. That it comes out monotone at every step, for both
  Pundits, is the measurement behind the whole design.

Needs a populated ``data/raw/``, which is gitignored, so these skip when it is absent:

    python -m epl.ingest fetch
    python -m epl.pundits fetch
"""

from __future__ import annotations

import pandas as pd
import pytest

from epl.benchmarks import MARKET_LINE, NAIVE_BASELINE
from epl.ingest import load_matches, raw_season_path
from epl.ledger import backtest, schema, scoreboard
from epl.models import ELO
from epl.pundits import dataset, report
from epl.pundits.calibrated import MARGIN_MAP_LAWRENSON, MARGIN_MAP_SUTTON
from epl.pundits.predictor import LAWRENSON, SUTTON

pytestmark = pytest.mark.cache

#: The Seasons the two Pundits worked. Everything below is scored over these rather than the whole
#: Evaluation Window: the Predictions outside them are the same either way, and walking Elo over
#: 21 Seasons to score nine is a minute nobody needs to spend.
PUNDIT_SEASONS = range(2017, 2026)

#: What each Calibrated Pundit scores over its own slate, and how many Fixtures it reached. The
#: count is the Pundit's own minus exactly :data:`epl.pundits.margin.MINIMUM_SAMPLE` — the opening
#: calls of a record, which have no map behind them and are therefore not covered.
CALIBRATED = {
    "lawrenson": {"fixtures": 1856, "rps": 0.212661},
    "sutton": {"fixtures": 1472, "rps": 0.211110},
}

#: The as-stated RPS over that same narrower slate, and the gap. Not the 0.3341 and 0.3343 of
#: issue #11: those are over all 1,896 and 1,512 calls, and these are over the Fixtures both
#: readings reached, which is what makes the subtraction mean anything.
AS_STATED = {"lawrenson": 0.333513, "sutton": 0.334579}
COST_OF_CERTAINTY = {"lawrenson": 0.120852, "sutton": 0.123469}

#: How each map bucketed the margins by the end of the record. Nothing chose these boundaries: the
#: rare margins merged inward because they never reached the minimum, and the common ones stood
#: alone because they did. Lawrenson never called worse than -3 and Sutton called -5 twice.
BUCKETS = {
    "lawrenson": ["-3, -2", "-1", "0", "1", "2", "3, 4", "pooled"],
    "sutton": ["-5, -4, -3, -2", "-1", "0", "1", "2", "3, 4, 5, 6", "pooled"],
}


def _require_football_data_cache() -> None:
    if not raw_season_path(2017, "E0").exists():
        pytest.skip("data/raw/ is not populated")


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    _require_football_data_cache()
    return load_matches()


@pytest.fixture(scope="module")
def calls() -> pd.DataFrame:
    return dataset.load()


@pytest.fixture(scope="module")
def scored(matches: pd.DataFrame) -> pd.DataFrame:
    """Every Predictor a three-way board carries, walked and then calibrated as a whole.

    Calibrated over each Predictor's whole walk before any slate is cut, which is the split
    :func:`epl.ledger.scoreboard.lines` exists for — see its docstring.
    """
    rows = pd.concat(
        [
            backtest.backfill(predictor, matches, seasons=PUNDIT_SEASONS)
            for predictor in (
                LAWRENSON,
                SUTTON,
                MARGIN_MAP_LAWRENSON,
                MARGIN_MAP_SUTTON,
                ELO,
                MARKET_LINE,
                NAIVE_BASELINE,
            )
        ],
        ignore_index=True,
    )
    return scoreboard.calibrated_predictions(rows, matches, seasons=PUNDIT_SEASONS)


def _board(scored: pd.DataFrame, pundit: str) -> pd.DataFrame:
    fair = MARGIN_MAP_LAWRENSON if pundit == "lawrenson" else MARGIN_MAP_SUTTON
    return report.three_way(scored, fair).set_index("predictor")


class TestWhatTheMarginMapIsWorth:
    @pytest.mark.parametrize("pundit", sorted(CALIBRATED))
    def test_the_calibrated_rps_is_what_the_docs_quote(
        self, pundit: str, scored: pd.DataFrame
    ) -> None:
        board = _board(scored, pundit)
        expected = CALIBRATED[pundit]

        assert board.loc[f"margin_map_{pundit}", "rps"] == pytest.approx(
            expected["rps"], abs=5e-5
        )
        assert int(board.loc[f"margin_map_{pundit}", "fixtures"]) == expected["fixtures"]

    @pytest.mark.parametrize("pundit", sorted(CALIBRATED))
    def test_it_clears_the_bar_the_shared_layer_set(
        self, pundit: str, scored: pd.DataFrame
    ) -> None:
        """`epl.calibration` gets an as-stated Pundit to 0.2374 and 0.2473 without seeing a
        Scoreline at all (ADR 0003). Reading the margin does better than that, which is the whole
        justification for a second map existing."""
        generic = {"lawrenson": 0.2374, "sutton": 0.2473}[pundit]

        assert _board(scored, pundit).loc[f"margin_map_{pundit}", "rps"] < generic - 0.02

    @pytest.mark.parametrize("pundit", sorted(CALIBRATED))
    def test_a_calibrated_pundit_beats_the_floor_where_an_as_stated_one_is_below_it(
        self, pundit: str, scored: pd.DataFrame
    ) -> None:
        """The sentence issue #12 exists to make true. As stated, a Pundit scores a tenth of a
        point *worse* than a Predictor that does not know which Clubs are playing; read fairly,
        the same calls beat it."""
        board = _board(scored, pundit)
        floor = board.loc["naive_baseline", "rps"]

        assert board.loc[pundit, "rps"] > floor + 0.09
        assert board.loc[f"margin_map_{pundit}", "rps"] < floor - 0.02

    @pytest.mark.parametrize("pundit", sorted(CALIBRATED))
    def test_it_does_not_beat_the_model_or_the_market(
        self, pundit: str, scored: pd.DataFrame
    ) -> None:
        """Not a requirement, and it is recorded because it might not have held: ADR 0003 is
        explicit that a Calibrated Pundit *may* beat our own models and that this would be a real
        finding. On this corpus it does not — it sits between Elo and the floor."""
        board = _board(scored, pundit)
        fair = board.loc[f"margin_map_{pundit}", "rps"]

        assert board.loc["market_line", "rps"] < board.loc["elo", "rps"] < fair


class TestTheCostOfStatingCertainty:
    @pytest.mark.parametrize("pundit", sorted(COST_OF_CERTAINTY))
    def test_the_gap_between_the_two_readings_is_about_a_tenth_of_a_point(
        self, pundit: str, scored: pd.DataFrame
    ) -> None:
        """ADR 0003's deliverable, in one number: what being asked for a Scoreline instead of a
        probability charged the forecaster."""
        fair = MARGIN_MAP_LAWRENSON if pundit == "lawrenson" else MARGIN_MAP_SUTTON
        (line,) = report.certainty(report.boards(scored, [fair]), [fair]).to_dict("records")

        assert line["as_stated_rps"] == pytest.approx(AS_STATED[pundit], abs=5e-5)
        assert line["cost_of_certainty"] == pytest.approx(COST_OF_CERTAINTY[pundit], abs=5e-5)

    @pytest.mark.parametrize("pundit", sorted(COST_OF_CERTAINTY))
    def test_accuracy_barely_moves_while_rps_collapses(
        self, pundit: str, scored: pd.DataFrame
    ) -> None:
        """The two readings pick almost the same Outcomes — the map only changes the top pick where
        a bucket's mode differs from the call, which is the draw calls. So the 0.12 RPS really is
        the format of the question rather than a different set of opinions."""
        fair = MARGIN_MAP_LAWRENSON if pundit == "lawrenson" else MARGIN_MAP_SUTTON
        (line,) = report.certainty(report.boards(scored, [fair]), [fair]).to_dict("records")

        assert abs(line["calibrated_accuracy"] - line["as_stated_accuracy"]) < 0.01
        assert line["cost_of_certainty"] > 0.11


class TestTheMapItself:
    @pytest.mark.parametrize("pundit", sorted(BUCKETS))
    def test_the_buckets_are_the_ones_the_sample_produced(
        self, pundit: str, matches: pd.DataFrame
    ) -> None:
        """No cap was chosen and no boundary was tuned. The extremes merged inward because they
        never reached :data:`epl.pundits.margin.MINIMUM_SAMPLE`, and the rest stood alone."""
        maps = report.published_maps(matches).set_index("pundit")

        assert maps.loc[pundit, "margins"].tolist() == BUCKETS[pundit]

    @pytest.mark.parametrize("pundit", sorted(BUCKETS))
    def test_a_bigger_predicted_margin_means_a_higher_home_rate_at_every_step(
        self, pundit: str, matches: pd.DataFrame
    ) -> None:
        """Measured, not imposed. :mod:`epl.pundits.margin` does not enforce monotonicity — the
        isotonic layer of ADR 0006 does, because it corrects a scale and must not touch a ranking,
        and here the ranking is the thing being measured. That it comes out monotone anyway, for
        both Pundits, is the finding the whole design rests on."""
        maps = report.published_maps(matches)
        buckets = maps.loc[
            (maps["pundit"] == pundit) & (maps["margins"] != "pooled"), "prob_home"
        ]

        assert buckets.is_monotonic_increasing

    @pytest.mark.parametrize("pundit", sorted(BUCKETS))
    def test_a_three_nil_call_is_worth_far_more_than_a_one_nil(
        self, pundit: str, matches: pd.DataFrame
    ) -> None:
        """Issue #12's first acceptance criterion, measured. The shared layer cannot tell these
        apart at all; here they are 40 points of Home probability apart."""
        maps = report.published_maps(matches)
        mine = maps.loc[maps["pundit"] == pundit].set_index("margins")
        weak = mine.loc["1", "prob_home"]
        # The +3 bucket is named for every margin merged into it, so it is found by its first.
        boldest = next(name for name in mine.index if name.startswith("3"))

        assert mine.loc[boldest, "prob_home"] - weak > 0.3

    @pytest.mark.parametrize("pundit", sorted(BUCKETS))
    def test_no_bucket_rests_on_less_than_the_minimum(
        self, pundit: str, matches: pd.DataFrame
    ) -> None:
        from epl.pundits.margin import MINIMUM_SAMPLE

        maps = report.published_maps(matches)

        assert (maps.loc[maps["pundit"] == pundit, "calls"] >= MINIMUM_SAMPLE).all()


class TestItWalksLikeEveryOtherPredictor:
    @pytest.mark.parametrize("pundit", sorted(CALIBRATED))
    def test_the_walk_audits_clean(self, pundit: str, scored: pd.DataFrame) -> None:
        rows = scored.loc[scored["predictor"] == f"margin_map_{pundit}"]

        assert schema.audit(rows[list(schema.LEDGER_COLUMNS)]) == []

    @pytest.mark.parametrize("pundit", sorted(CALIBRATED))
    def test_every_row_records_the_history_its_map_was_fitted_on(
        self, pundit: str, scored: pd.DataFrame
    ) -> None:
        """Unlike a Pundit, which consumes no history and records ``inputs_seen = 0``, this one
        reads results — so the walk-forward claim is checkable off the stored file months later
        rather than asserted by a test."""
        rows = scored.loc[scored["predictor"] == f"margin_map_{pundit}"]

        assert (rows["inputs_seen"] > 0).all()
        assert (rows["latest_input"] < rows["as_of_instant"]).all()

    @pytest.mark.parametrize("pundit", sorted(CALIBRATED))
    def test_the_shared_layer_now_costs_it_rather_than_buying(
        self, pundit: str, scored: pd.DataFrame
    ) -> None:
        """The confirmation stage 6 asked for, arriving from the other direction. `epl.calibration`
        gained an as-stated Pundit about 0.09 because it was the first genuinely miscalibrated
        input it had seen. Put the margin map in front of it and the gain disappears — it costs
        about 0.001, exactly as it does for Elo and the market. The layer was never broken."""
        board = _board(scored, pundit)
        line = board.loc[f"margin_map_{pundit}"]

        assert line["calibrated_rps"] > line["rps"]
        assert line["calibrated_rps"] - line["rps"] < 0.005

    @pytest.mark.parametrize("pundit", sorted(CALIBRATED))
    def test_it_is_far_better_calibrated_than_the_pundit_it_was_built_from(
        self, pundit: str, scored: pd.DataFrame
    ) -> None:
        """0.02 against 0.33 on the ten-bin error. Still three times the 0.006 the other four
        reach, which is the map's seven buckets showing: it is coarse by construction."""
        board = _board(scored, pundit)

        assert board.loc[pundit, "ece"] > 0.3
        assert board.loc[f"margin_map_{pundit}", "ece"] < 0.03


class TestTheCallsRankedByMiss:
    def test_every_covered_call_is_ranked_and_carries_its_scoreline(
        self, scored: pd.DataFrame, calls: pd.DataFrame
    ) -> None:
        """Spec, user story 34."""
        ranked = report.ranked_calls(scored, calls)

        assert len(ranked) == sum(one["fixtures"] for one in CALIBRATED.values())
        assert ranked["pred_home_goals"].notna().all()
        assert ranked["margin"].notna().all()

    def test_the_best_calls_are_bold_ones_that_came_off(
        self, scored: pd.DataFrame, calls: pd.DataFrame
    ) -> None:
        """A call the map read well is one where a big predicted margin met the Outcome that
        margin usually produces — which is also where stating certainty *paid*, so the cost is
        negative on exactly these rows."""
        ranked = report.ranked_calls(scored, calls).sort_values("miss")
        best = ranked.head(20)

        assert (best["margin"].abs() >= 3).all()
        assert (best["cost_of_certainty"] < 0).all()

    def test_the_worst_calls_are_bold_ones_that_did_not(
        self, scored: pd.DataFrame, calls: pd.DataFrame
    ) -> None:
        ranked = report.ranked_calls(scored, calls).sort_values("miss")
        worst = ranked.tail(20)

        assert (worst["as_stated_rps"] == 1.0).all()
        assert (worst["cost_of_certainty"] > 0).all()
