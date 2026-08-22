"""The regenerable store: a Predictor walked over history and written to `outputs/backtest/`.

A Backtest Prediction is reproducible, so the file is a convenience rather than evidence and may be
deleted and rebuilt freely (ADR 0005). Two things have to be true for that to hold: the walk must
be leak-free at every round, and a rebuild must produce the same bytes — otherwise "regenerable"
quietly means "different every time".
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from epl.ledger import backtest, schema


@pytest.fixture
def two_seasons(make_matches: Callable[..., pd.DataFrame]) -> pd.DataFrame:
    """One Season to warm up on and one to be scored, three Fixtures each."""
    return make_matches(
        {"season": 2023, "date": "2023-08-12", "home_club": "arsenal", "outcome": "H"},
        {"season": 2023, "date": "2023-08-19", "home_club": "everton", "outcome": "D"},
        {"season": 2023, "date": "2023-08-26", "home_club": "fulham", "outcome": "A"},
        {"season": 2024, "date": "2024-08-17", "home_club": "arsenal", "outcome": "H"},
        {"season": 2024, "date": "2024-08-24", "home_club": "everton", "outcome": "H"},
        {"season": 2024, "date": "2024-08-31", "home_club": "fulham", "outcome": "A"},
    )


class TestWalkingTheWindow:
    def test_it_predicts_every_fixture_in_the_window(
        self, two_seasons: pd.DataFrame, make_predictor: Callable[..., object]
    ) -> None:
        rows = backtest.backfill(make_predictor(), two_seasons, seasons=[2024])

        assert len(rows) == 3
        assert set(rows["season"]) == {2024}

    def test_every_round_is_predicted_from_its_own_as_of_instant(
        self, two_seasons: pd.DataFrame, make_predictor: Callable[..., object]
    ) -> None:
        """Three Fixtures a week apart are three rounds, each with its own instant and evidence."""
        rows = backtest.backfill(make_predictor(), two_seasons, seasons=[2024])

        assert list(rows["prediction_round"]) == ["2024-08-16", "2024-08-23", "2024-08-30"]
        assert schema.audit(rows) == []

    def test_earlier_seasons_are_evidence_even_though_they_are_not_scored(
        self, two_seasons: pd.DataFrame, make_predictor: Callable[..., object]
    ) -> None:
        """What the Burn-In Window is for: the first scored round already has a warmed-up
        Predictor behind it, without a single Burn-In Fixture being scored (ADR 0008)."""
        rows = backtest.backfill(make_predictor(), two_seasons, seasons=[2024])

        assert rows.iloc[0]["inputs_seen"] == 3
        assert rows.iloc[0]["latest_input"] == pd.Timestamp("2023-08-26")

    def test_a_predictor_never_sees_the_round_it_is_predicting(
        self, two_seasons: pd.DataFrame, make_predictor: Callable[..., object]
    ) -> None:
        rows = backtest.backfill(make_predictor(), two_seasons, seasons=[2024])
        last = rows.iloc[-1]

        assert last["latest_input"] < last["as_of_instant"] < last["kickoff"]

    def test_only_the_named_tiers_are_predicted(
        self, make_matches: Callable[..., pd.DataFrame], make_predictor: Callable[..., object]
    ) -> None:
        matches = make_matches(
            {"date": "2024-08-17", "division": "E0"},
            {"date": "2024-08-17", "division": "E1"},
        )

        rows = backtest.backfill(make_predictor(), matches, seasons=[2024], divisions=("E0",))

        assert set(rows["division"]) == {"E0"}

    def test_a_window_with_no_fixtures_gives_an_empty_ledger(
        self, two_seasons: pd.DataFrame, make_predictor: Callable[..., object]
    ) -> None:
        rows = backtest.backfill(make_predictor(), two_seasons, seasons=[2019])

        assert rows.empty
        assert list(rows.columns) == list(schema.LEDGER_COLUMNS)


class TestTheStore:
    def test_what_is_written_is_what_is_read_back(
        self,
        project_root: Path,
        two_seasons: pd.DataFrame,
        make_predictor: Callable[..., object],
    ) -> None:
        rows = backtest.backfill(make_predictor(), two_seasons, seasons=[2024])
        backtest.write(rows)

        pd.testing.assert_frame_equal(backtest.read(), rows)

    def test_it_writes_one_file_per_predictor_named_after_it(
        self,
        project_root: Path,
        two_seasons: pd.DataFrame,
        make_predictor: Callable[..., object],
    ) -> None:
        rows = pd.concat(
            [
                backtest.backfill(make_predictor("coin"), two_seasons, seasons=[2024]),
                backtest.backfill(make_predictor("dice"), two_seasons, seasons=[2024]),
            ],
            ignore_index=True,
        )

        written = backtest.write(rows)

        assert [path.name for path in written] == ["coin.csv", "dice.csv"]

    def test_reading_one_predictor_reads_only_that_one(
        self,
        project_root: Path,
        two_seasons: pd.DataFrame,
        make_predictor: Callable[..., object],
    ) -> None:
        backtest.write(backtest.backfill(make_predictor("coin"), two_seasons, seasons=[2024]))
        backtest.write(backtest.backfill(make_predictor("dice"), two_seasons, seasons=[2024]))

        assert set(backtest.read("coin")["predictor"]) == {"coin"}

    def test_an_empty_store_reads_as_an_empty_ledger(self, project_root: Path) -> None:
        assert backtest.read().empty
        assert list(backtest.read().columns) == list(schema.LEDGER_COLUMNS)

    def test_regenerating_produces_the_same_bytes(
        self,
        project_root: Path,
        two_seasons: pd.DataFrame,
        make_predictor: Callable[..., object],
    ) -> None:
        """"Deterministic and regenerable" (ADR 0005) means exactly this. A file that came back
        different every run would make a rebuild indistinguishable from a change."""
        first = backtest.write(backtest.backfill(make_predictor(), two_seasons, seasons=[2024]))
        bytes_before = first[0].read_bytes()

        backtest.write(backtest.backfill(make_predictor(), two_seasons, seasons=[2024]))

        assert first[0].read_bytes() == bytes_before

    def test_rows_that_fail_the_audit_are_never_written(
        self,
        project_root: Path,
        two_seasons: pd.DataFrame,
        make_predictor: Callable[..., object],
    ) -> None:
        rows = backtest.backfill(make_predictor(), two_seasons, seasons=[2024])
        rows.loc[0, "latest_input"] = rows.loc[0, "as_of_instant"]

        with pytest.raises(schema.LedgerError, match="future data"):
            backtest.write(rows)

        assert not backtest.path("fixed").exists()
