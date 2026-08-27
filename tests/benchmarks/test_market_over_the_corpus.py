"""The Market Line and the Ceiling Line measured over the real corpus.

Issue #8 states its acceptance in numbers, and numbers over 7,980 Fixtures are the one thing a
unit test cannot check. These re-derive them from the ingested cache rather than trusting
docs/DECISIONS.md, which is what that document asks for:

* 0.19379 normalised, 0.19362 Shin, 0.19359 power — the three within 0.0002 RPS of each other
* mean overround 1.0562 on the market-average pre-match line
* the `BbAv*` to `Avg*` splice invisible at the 2019/20 join
* Seasons 2000/01-2001/02 with no market at all, handled as absent rather than as missing data

They need a populated ``data/raw/``, which is gitignored, so they skip when it is absent:

    python -c "from epl.ingest import fetch_all; fetch_all()"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epl import metrics
from epl.benchmarks import CEILING_LINE, MARKET_LINE, vig
from epl.benchmarks.market import overround_report
from epl.ingest import DIVISIONS, FIRST_SEASON, LAST_SEASON, load_matches, raw_season_path
from epl.ledger import backtest, scoreboard
from epl.windows import EVALUATION_WINDOW, LIVE_SEASON

pytestmark = pytest.mark.cache

#: What issue #8 says the three methods score, to the five places it states them to.
EXPECTED_RPS: dict[str, float] = {
    "normalise": 0.19379,
    "shin": 0.19362,
    "power": 0.19359,
}

#: Premier League Fixtures in the Evaluation Window, every one of them priced.
EXPECTED_FIXTURES = 7980

#: The Season the closing book — and so the Ceiling Line — begins in (ADR 0001).
FIRST_CLOSING_SEASON = 2019


def _require_cache() -> None:
    missing = [
        (season, division)
        for season in range(FIRST_SEASON, LAST_SEASON + 1)
        for division in DIVISIONS
        if not raw_season_path(season, division).exists()
    ]
    if missing:
        pytest.skip(f"raw cache incomplete ({len(missing)} files missing)")


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    _require_cache()
    return load_matches()


@pytest.fixture(scope="module")
def scored(matches: pd.DataFrame) -> pd.DataFrame:
    """Every Evaluation Window Premier League Fixture the Market Line prices, with its Outcome."""
    window = matches.loc[
        matches["season"].isin(list(EVALUATION_WINDOW)) & (matches["division"] == "E0")
    ]
    return window.loc[MARKET_LINE.covers(window)].reset_index(drop=True)


class TestTheMarketLineScoresWhatItShould:
    def test_it_prices_every_fixture_in_the_evaluation_window(
        self, scored: pd.DataFrame
    ) -> None:
        """The Evaluation Window begins in 2005/06 precisely so that this holds: "every scored
        Fixture has a market to be compared against" (CONTEXT.md)."""
        assert len(scored) == EXPECTED_FIXTURES

    @pytest.mark.parametrize("method", sorted(EXPECTED_RPS))
    def test_each_method_scores_what_the_ticket_says(
        self, scored: pd.DataFrame, method: str
    ) -> None:
        book = scored[list(MARKET_LINE.columns)].to_numpy(float)

        card = metrics.score(vig.remove(book, method=method), scored["outcome"].tolist())

        assert card.rps == pytest.approx(EXPECTED_RPS[method], abs=5e-6)

    def test_the_three_methods_differ_by_about_two_ten_thousandths(
        self, scored: pd.DataFrame
    ) -> None:
        """The point of measuring all three: "a reader can see for themselves that the choice
        barely matters for benchmarking" (issue #8). It is worth 0.0002 RPS."""
        book = scored[list(MARKET_LINE.columns)].to_numpy(float)
        outcomes = scored["outcome"].tolist()

        spread = [
            metrics.score(vig.remove(book, method=method), outcomes).rps
            for method in vig.METHODS
        ]

        assert max(spread) - min(spread) == pytest.approx(0.0002, abs=5e-5)

    def test_it_beats_the_naive_baseline_comfortably(self, scored: pd.DataFrame) -> None:
        """0.1936 against 0.2292. Anything that cannot beat the floor has no value; the market
        clears it by 15%, which is the gap a model has to find some of (CONTEXT.md)."""
        book = scored[list(MARKET_LINE.columns)].to_numpy(float)

        card = metrics.score(vig.remove(book), scored["outcome"].tolist())

        assert card.rps < 0.20


class TestTheOverroundIsReported:
    def test_the_mean_margin_is_the_one_on_record(self, scored: pd.DataFrame) -> None:
        """1.0562 — 5.62% taken out. Reported rather than trusted, so that a vig removal which
        quietly stopped removing anything would be visible here first (issue #8)."""
        assert MARKET_LINE.overround(scored).mean() == pytest.approx(1.05616, abs=5e-5)

    def test_every_book_in_the_window_carries_a_margin(self, scored: pd.DataFrame) -> None:
        """A market average below one would be an arbitrage rather than a price."""
        assert (MARKET_LINE.overround(scored) >= 1.0).all()

    def test_the_report_covers_every_priced_season_and_tier(self, matches: pd.DataFrame) -> None:
        """Every Evaluation Window Season, and the Season in progress — which matters more than it
        looks: a live Fixture with no book is a live Fixture the Market Line cannot be sealed on."""
        report = overround_report(matches)
        premier = report.loc[
            (report["predictor"] == "market_line") & (report["division"] == "E0")
        ]
        scored = premier.loc[premier["season"].isin(list(EVALUATION_WINDOW))]

        assert list(premier["season"]) == [*EVALUATION_WINDOW, LIVE_SEASON]
        assert int(scored["fixtures"].sum()) == EXPECTED_FIXTURES

    def test_the_margin_has_narrowed_over_the_window(self, matches: pd.DataFrame) -> None:
        """9.4% in 2005/06 against about 4.1% in the early 2020s. A fact about the market that no
        bug in the removal would reproduce by accident — which is what makes it a sanity check."""
        report = overround_report(matches)
        premier = report.loc[
            (report["predictor"] == "market_line") & (report["division"] == "E0")
        ].set_index("season")

        assert premier.loc[2005, "mean_overround"] == pytest.approx(1.0945, abs=5e-4)
        assert premier.loc[2022, "mean_overround"] == pytest.approx(1.0407, abs=5e-4)


class TestTheSpliceIsInvisible:
    """"`BbAvH/D/A` (2005/06-2018/19) spliced to `AvgH/D/A` (2019/20 onward) with no change of
    definition visible at the join" (issue #8).

    The margin is the quantity that would give a definition change away, because it is a property
    of the book rather than of how the Fixtures turned out — a different quantity spliced in would
    step, where a genuinely continuous one only drifts.
    """

    @pytest.fixture(scope="module")
    def per_season(self, matches: pd.DataFrame) -> pd.Series:
        report = overround_report(matches)
        premier = report.loc[
            (report["predictor"] == "market_line") & (report["division"] == "E0")
        ]
        return premier.set_index("season")["mean_overround"]

    def test_the_series_is_continuous_across_the_join(self, per_season: pd.Series) -> None:
        """2018/19 is the last `BbAv*` Season and 2019/20 the first `Avg*` one."""
        step = abs(per_season.loc[2019] - per_season.loc[2018])

        assert step < 0.002

    def test_the_join_is_a_smaller_step_than_the_series_takes_elsewhere(
        self, per_season: pd.Series
    ) -> None:
        """The stronger claim, and the one that would survive the tolerance above being argued
        with: the change of spelling moves the margin less than an ordinary Season does."""
        steps = per_season.diff().abs().dropna()
        join = abs(per_season.loc[2019] - per_season.loc[2018])

        assert join < steps.max()
        assert join < steps.median()

    def test_no_season_in_the_window_is_missing_the_pre_match_book(
        self, matches: pd.DataFrame
    ) -> None:
        window = matches.loc[
            matches["season"].isin(list(EVALUATION_WINDOW)) & (matches["division"] == "E0")
        ]

        assert MARKET_LINE.covers(window).all()


class TestSeasonsWithNoMarket:
    """"Seasons 2000/01-2001/02 have no odds at all and therefore no market comparison; this is
    handled explicitly rather than as missing data" (issue #8, ADR 0001)."""

    def test_the_market_line_covers_none_of_them(self, matches: pd.DataFrame) -> None:
        early = matches.loc[matches["season"].isin([2000, 2001])]

        assert not MARKET_LINE.covers(early).any()

    def test_walking_them_writes_no_rows_rather_than_empty_ones(
        self, matches: pd.DataFrame
    ) -> None:
        rows = backtest.backfill(MARKET_LINE, matches, seasons=[2000, 2001])

        assert rows.empty

    def test_they_are_absent_from_the_overround_report_rather_than_blank(
        self, matches: pd.DataFrame
    ) -> None:
        """A Season with no market is not a Season whose margin was zero."""
        report = overround_report(matches)

        assert not report["season"].isin([2000, 2001]).any()


class TestTheCeilingLineOverTheCorpus:
    @pytest.fixture(scope="module")
    def closing(self, matches: pd.DataFrame) -> pd.DataFrame:
        window = matches.loc[
            matches["season"].isin(list(EVALUATION_WINDOW)) & (matches["division"] == "E0")
        ]
        return window.loc[CEILING_LINE.covers(window)].reset_index(drop=True)

    def test_it_covers_2019_20_onward_and_nothing_earlier(self, closing: pd.DataFrame) -> None:
        assert set(closing["season"]) == set(range(FIRST_CLOSING_SEASON, LIVE_SEASON))
        assert len(closing) == 2660

    def test_it_beats_the_market_line_on_the_fixtures_they_share(
        self, closing: pd.DataFrame
    ) -> None:
        """The like-for-like comparison, and the only one worth making. A few hours of team news
        is worth about 0.0013 RPS — real, and far smaller than the 0.036 the market takes out of
        the Naive Baseline."""
        outcomes = closing["outcome"].tolist()

        ceiling = metrics.score(
            vig.remove(closing[list(CEILING_LINE.columns)].to_numpy(float)), outcomes
        )
        market = metrics.score(
            vig.remove(closing[list(MARKET_LINE.columns)].to_numpy(float)), outcomes
        )

        assert ceiling.rps < market.rps
        assert market.rps - ceiling.rps == pytest.approx(0.0013, abs=5e-4)

    def test_its_raw_rps_looks_worse_than_the_market_lines_which_is_why_it_needs_a_note(
        self, closing: pd.DataFrame, scored: pd.DataFrame
    ) -> None:
        """The trap the caveat exists for. Scored over its own shorter, harder span the Ceiling
        Line reads 0.1968 against the Market Line's full-window 0.1936, so a bare scoreboard would
        say the closing odds are worse. They are not — they are measured over different Fixtures."""
        ceiling = metrics.score(
            vig.remove(closing[list(CEILING_LINE.columns)].to_numpy(float)),
            closing["outcome"].tolist(),
        )
        market = metrics.score(
            vig.remove(scored[list(MARKET_LINE.columns)].to_numpy(float)),
            scored["outcome"].tolist(),
        )

        assert ceiling.rps > market.rps


class TestBothLandOnTheScoreboard:
    """"Both appear on the scoreboard beside the Naive Baseline" (issue #8), walked end to end
    over the real corpus rather than over a fixture."""

    @pytest.fixture(scope="module")
    def board(self, matches: pd.DataFrame) -> pd.DataFrame:
        rows = pd.concat(
            [
                backtest.backfill(MARKET_LINE, matches),
                backtest.backfill(CEILING_LINE, matches),
            ],
            ignore_index=True,
        )
        return scoreboard.build(rows, matches)

    def test_the_market_line_scores_what_it_should_end_to_end(
        self, board: pd.DataFrame
    ) -> None:
        line = board.set_index("predictor").loc["market_line"]

        assert int(line["fixtures"]) == EXPECTED_FIXTURES
        assert float(line["rps"]) == pytest.approx(EXPECTED_RPS["shin"], abs=5e-6)

    def test_the_ceiling_line_is_scored_over_its_own_span(self, board: pd.DataFrame) -> None:
        line = board.set_index("predictor").loc["ceiling_line"]

        assert int(line["fixtures"]) == 2660

    def test_the_ceiling_lines_caveat_is_on_the_board(self, board: pd.DataFrame) -> None:
        line = board.set_index("predictor").loc["ceiling_line"]

        assert "team news" in str(line["note"])

    def test_every_stored_row_audits_clean(self, matches: pd.DataFrame) -> None:
        """Neither line reads the corpus, so neither has any history to leak — and the ledger
        records that as ``inputs_seen = 0`` rather than as an unchecked claim."""
        rows = backtest.backfill(MARKET_LINE, matches)

        assert scoreboard.schema.audit(rows) == []
        assert (rows["inputs_seen"] == 0).all()
        assert rows["latest_input"].isna().all()

    def test_a_rebuilt_walk_is_identical(self, matches: pd.DataFrame) -> None:
        """Vig removal solves for a parameter by bisection, so "regenerable" has to mean the same
        bytes rather than the same bytes most of the time (ADR 0005)."""
        first = backtest.backfill(MARKET_LINE, matches, seasons=[2024])
        again = backtest.backfill(MARKET_LINE, matches, seasons=[2024])

        assert np.array_equal(
            first[["prob_home", "prob_draw", "prob_away"]].to_numpy(),
            again[["prob_home", "prob_draw", "prob_away"]].to_numpy(),
        )
