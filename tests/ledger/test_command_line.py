"""The ledger command line — how a Predictor actually gets scored end to end.

`python -m epl.ledger backfill` then `scoreboard` is the whole path issue #7 asks for: matches in,
Predictions out, scored, published.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from epl.ledger import __main__ as cli
from epl.ledger import backtest, scoreboard
from epl.paths import processed_dir


@pytest.fixture
def corpus(project_root: Path, make_matches: Callable[..., pd.DataFrame]) -> Path:
    """A processed match table on disk, which is what the command line reads."""
    matches = make_matches(
        {"season": 2004, "date": "2004-08-14", "home_club": "arsenal", "outcome": "H"},
        {"season": 2005, "date": "2005-08-13", "home_club": "arsenal", "outcome": "H"},
        {"season": 2005, "date": "2005-08-13", "home_club": "everton", "outcome": "A"},
        {"season": 2005, "date": "2005-08-20", "home_club": "fulham", "outcome": "H"},
    )
    processed_dir().mkdir(parents=True, exist_ok=True)
    path = processed_dir() / "matches.csv"
    matches.to_csv(path, index=False)
    return path


class TestBackfill:
    def test_it_writes_a_backtest_file_for_every_registered_predictor(
        self, corpus: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["backfill"]) == 0

        assert backtest.path("naive_baseline").exists()
        assert "naive_baseline" in capsys.readouterr().out

    def test_it_can_be_narrowed_to_one_predictor(self, corpus: Path) -> None:
        assert cli.main(["backfill", "--predictor", "naive_baseline"]) == 0

        assert set(backtest.read()["predictor"]) == {"naive_baseline"}

    def test_the_predictions_are_leak_free(self, corpus: Path) -> None:
        cli.main(["backfill"])
        rows = backtest.read()

        assert (rows["latest_input"] < rows["as_of_instant"]).all()
        assert (rows["as_of_instant"] < rows["kickoff"]).all()


class TestScoreboard:
    def test_it_scores_what_the_backfill_wrote(
        self, corpus: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli.main(["backfill"])

        assert cli.main(["scoreboard"]) == 0

        board = pd.read_csv(scoreboard.path())
        assert board["predictor"].tolist() == ["naive_baseline"]
        assert board["fixtures"].tolist() == [3]
        assert "rps" in capsys.readouterr().out


class TestScoreboardCompleteness:
    def test_a_registered_predictor_with_no_predictions_is_named_rather_than_dropped(
        self, corpus: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The criterion is that the scoreboard lists each *registered* Predictor. One that has
        never been backfilled has no metrics — epl.metrics refuses to average an empty slate, and
        a NaN on a scoreboard reads as a real number — so it is named instead of printed blank."""
        assert cli.main(["scoreboard"]) == 0

        printed = capsys.readouterr().out
        assert "naive_baseline" in printed
        assert "backfill" in printed


class TestAudit:
    def test_a_clean_ledger_passes(self, corpus: Path) -> None:
        cli.main(["backfill"])

        assert cli.main(["audit"]) == 0

    def test_a_tampered_ledger_fails(
        self, corpus: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The check that has to survive someone editing a stored file by hand, which is the only
        way a leak can enter a store that was audited on the way in."""
        cli.main(["backfill"])
        stored = backtest.path("naive_baseline")
        rows = pd.read_csv(stored)
        rows["latest_input"] = rows["as_of_instant"]
        rows.to_csv(stored, index=False)

        assert cli.main(["audit"]) == 1
        assert "future data" in capsys.readouterr().out
