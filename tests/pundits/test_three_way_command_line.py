"""The three-way command line, end to end: backfill in, published files out.

The numbers these produce on the real nine Seasons are pinned in
`tests/pundits/test_calibrated_over_the_corpus.py`. What is tested here is the path — that each
command reads the ledger, writes the file it says it writes, and prints the thing a reader needs
in order to read the number correctly. A report nobody can run is not a report.

Everything runs on a fabricated Season of sixty Fixtures with the map's minimum turned down, so it
needs no `data/raw/` and no market odds.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from epl.benchmarks import NAIVE_BASELINE
from epl.ingest import match_table
from epl.ledger import backtest
from epl.paths import processed_dir
from epl.predictors import register
from epl.pundits import __main__ as cli
from epl.pundits import dataset, margin, report
from epl.pundits.calibrated import CalibratedPundit, a_calibrated_pundit
from epl.pundits.predictor import Pundit

#: Small enough that a fabricated Season clears it inside the first month.
MINIMUM = 4

#: How many Fixtures the fabricated Season holds, and how they are called. The margins cycle so
#: that several buckets fill, and the Outcomes lean the way the margins do — which is what gives
#: the map something to find rather than noise.
FIXTURES = 60
MARGINS = (2, 0, -1, 1, 3, -2)


def _calls_and_matches() -> tuple[pd.DataFrame, pd.DataFrame]:
    """One Season of Fixtures, the Pundit's call on each, and what each one did."""
    start = pd.Timestamp("2017-08-12")
    rows = []
    for index in range(FIXTURES):
        margin = MARGINS[index % len(MARGINS)]
        # Home two-thirds of the time where the call leans home, and the mirror where it leans
        # away; a draw every third fixture in the middle bucket.
        outcome = ("H" if index % 3 else "A") if margin > 0 else ("A" if index % 3 else "D")
        # Two Clubs meet once per direction per Season, so every pairing has to be distinct:
        # twenty Clubs, with the gap between them widening every time the list wraps.
        home = index % 20
        rows.append(
            {
                "date": (start + pd.Timedelta(weeks=index // 2)).date(),
                "home_club": f"club_{home:02d}",
                "away_club": f"club_{(home + 1 + index // 20) % 20:02d}",
                "margin": margin,
                "outcome": outcome,
            }
        )

    frame = pd.DataFrame(rows).assign(season=2017, division="E0")
    calls = frame.assign(
        pundit="lawrenson",
        pred_home_goals=frame["margin"].clip(lower=0),
        pred_away_goals=(-frame["margin"]).clip(lower=0),
    )[list(dataset.CALL_COLUMNS)]
    goals = {"H": (1, 0), "D": (1, 1), "A": (0, 1)}
    matches = frame.assign(
        time=pd.NA,
        home_goals=[goals[outcome][0] for outcome in frame["outcome"]],
        away_goals=[goals[outcome][1] for outcome in frame["outcome"]],
    )[
        [
            "season",
            "division",
            "date",
            "time",
            "home_club",
            "away_club",
            "home_goals",
            "away_goals",
            "outcome",
        ]
    ]
    return calls, matches


@pytest.fixture
def unpublished(
    project_root: Path, registry: dict[str, object], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A corpus and a frozen dataset on disk, and nothing in either Prediction store.

    What every one of these commands looks like before `python -m epl.ledger backfill` has run,
    which is the state a reader of a fresh clone is actually in.
    """
    calls, matches = _calls_and_matches()
    processed_dir().mkdir(parents=True, exist_ok=True)
    matches.to_csv(processed_dir() / "matches.csv", index=False)
    monkeypatch.setattr(dataset, "path", lambda: tmp_path / "predictions.csv")
    dataset.write(calls)


@pytest.fixture
def published(
    project_root: Path, registry: dict[str, object], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> CalibratedPundit:
    """A backfilled ledger, a frozen dataset, and the two readings registered over them.

    Registered into the empty registry the ``registry`` fixture installs, so these Predictors
    never reach another test's scoreboard — and so ``report.calibrated_pundits`` finds them the
    same way it finds the real ones, by their being registered.
    """
    calls, matches = _calls_and_matches()

    processed_dir().mkdir(parents=True, exist_ok=True)
    matches.to_csv(processed_dir() / "matches.csv", index=False)
    monkeypatch.setattr(dataset, "path", lambda: tmp_path / "predictions.csv")
    dataset.write(calls)

    # One opponent rather than three: the Market Line needs odds this fabricated Season has none
    # of, and an opponent with no rows would empty the shared slate rather than fail loudly.
    monkeypatch.setattr(report, "OPPONENTS", ("naive_baseline",))

    pundit = register(Pundit("lawrenson", "Mark Lawrenson", note="as stated", calls=calls))
    fair = a_calibrated_pundit(pundit)
    fair.minimum = MINIMUM
    register(fair)

    corpus = match_table()
    for predictor in (NAIVE_BASELINE, pundit, fair):
        backtest.write(backtest.backfill(predictor, corpus, seasons=[2017]))
    return fair


class TestTheThreeWayBoard:
    def test_it_writes_the_board_and_the_cost_of_certainty(
        self, published: CalibratedPundit
    ) -> None:
        """Issue #12's seventh criterion: plain files under `outputs/`, so a frontend can be built
        without any modelling logic leaking into it."""
        assert cli.main(["three-way"]) == 0

        board = pd.read_csv(report.path("three_way"))
        costs = pd.read_csv(report.path("certainty"))
        assert list(board.columns) == list(report.THREE_WAY_COLUMNS)
        assert list(costs.columns) == list(report.CERTAINTY_COLUMNS)

    def test_all_three_readings_are_scored_over_the_same_fixtures(
        self, published: CalibratedPundit
    ) -> None:
        cli.main(["three-way"])
        board = pd.read_csv(report.path("three_way"))

        assert sorted(board["predictor"]) == [
            "lawrenson",
            "margin_map_lawrenson",
            "naive_baseline",
        ]
        assert board["fixtures"].nunique() == 1

    def test_the_gap_is_printed_and_labelled_as_the_cost_of_stating_certainty(
        self, published: CalibratedPundit, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The fifth acceptance criterion. A reader should not have to subtract two rows and guess
        what the difference between them means."""
        cli.main(["three-way"])

        printed = capsys.readouterr().out
        assert "cost of stating certainty" in printed
        assert "cost_of_certainty" in printed

    def test_the_as_stated_reading_really_is_the_worse_one(
        self, published: CalibratedPundit
    ) -> None:
        cli.main(["three-way"])
        board = pd.read_csv(report.path("three_way")).set_index("predictor")
        costs = pd.read_csv(report.path("certainty"))

        assert board.loc["lawrenson", "rps"] > board.loc["margin_map_lawrenson", "rps"]
        assert costs["cost_of_certainty"].iloc[0] > 0

    def test_both_halves_of_the_comparison_are_printed(
        self, published: CalibratedPundit, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Every metric twice, exactly as the main scoreboard does it (ADR 0006)."""
        cli.main(["three-way"])

        printed = capsys.readouterr().out
        assert "pre-calibration" in printed and "post-calibration" in printed

    def test_the_naming_caveat_travels_with_the_number(
        self, published: CalibratedPundit, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """ADR 0003's Consequences section, on the artifact a reader actually receives."""
        cli.main(["three-way"])

        printed = capsys.readouterr().out
        assert "not Mark Lawrenson" in printed
        assert "one-feature model" in printed

    def test_an_empty_ledger_says_so_rather_than_publishing_nothing(
        self, unpublished: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The state a fresh clone is in before `python -m epl.ledger backfill` has run. A file of
        zero rows would read as a measurement rather than as a missing step."""
        assert cli.main(["three-way"]) == 0

        assert "no stored Predictions" in capsys.readouterr().out
        assert not report.path("three_way").exists()


class TestTheRankedCalls:
    def test_it_writes_every_call_and_prints_only_the_ends(
        self, published: CalibratedPundit, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The file holds every call, so a frontend picks its own N rather than inheriting one
        chosen here."""
        assert cli.main(["calls"]) == 0

        ranked = pd.read_csv(report.path("pundit_calls"))
        assert list(ranked.columns) == list(report.CALL_COLUMNS)
        assert len(ranked) == FIXTURES - MINIMUM
        assert capsys.readouterr().out.count("read best") == 1

    def test_the_calls_are_ordered_by_miss(self, published: CalibratedPundit) -> None:
        cli.main(["calls"])

        ranked = pd.read_csv(report.path("pundit_calls"))
        assert ranked["miss"].is_monotonic_increasing

    def test_what_the_miss_means_is_printed_beside_it(
        self, published: CalibratedPundit, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli.main(["calls"])

        assert "miss is the RPS of the fair reading" in capsys.readouterr().out

    def test_the_calls_that_are_not_ranked_at_all_are_counted(
        self, published: CalibratedPundit, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A call with no map behind it has no miss, so a genuinely awful opening call cannot
        appear among the worst however bad it was. Said out loud for the reason `build` names the
        Fixtures nobody called: an unremarked gap is how forty quietly becomes four hundred."""
        cli.main(["calls"])

        printed = capsys.readouterr().out
        assert f"{MINIMUM} of {FIXTURES} calls are not ranked" in printed
        assert "no map had a sample behind yet" in printed

    def test_an_empty_ledger_says_so(
        self, unpublished: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["calls"]) == 0

        assert "no stored Predictions" in capsys.readouterr().out


class TestThePublishedMap:
    def test_it_writes_the_map_itself(self, published: CalibratedPundit) -> None:
        """"What a 3-0 from this Pundit is worth" is the question a reader arrives with, and it
        deserves a file rather than a docstring."""
        assert cli.main(["map"]) == 0

        maps = pd.read_csv(report.path("margin_map"))
        assert list(maps.columns) == list(report.PUBLISHED_MAP_COLUMNS)
        assert margin.POOLED in maps["margins"].tolist()

    def test_every_published_bucket_has_the_minimum_behind_it(
        self, published: CalibratedPundit
    ) -> None:
        cli.main(["map"])

        maps = pd.read_csv(report.path("margin_map"))
        assert (maps["calls"] >= MINIMUM).all()

    def test_a_stronger_call_is_quoted_a_higher_home_rate(
        self, published: CalibratedPundit
    ) -> None:
        cli.main(["map"])

        maps = pd.read_csv(report.path("margin_map")).set_index("margins")
        assert maps.loc["3", "prob_home"] > maps.loc["1", "prob_home"]
