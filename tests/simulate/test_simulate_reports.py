"""The simulate command line, end to end: a match table in, a projection out.

The numbers a real run produces are pinned in ``test_projection_over_the_corpus.py``. What is
tested here is the path — that each command reads a match table, writes the file it says it writes,
and prints the things a reader needs in order to read the number correctly: both seeds, the chain
that settled the ties, and ADR 0007's sentence about the two fits. A report nobody can run is not a
report, and one that prints a title probability without saying which fit produced it is worse.

Everything runs on a fabricated Season, and the expensive half is replaced: the posterior fit is
minutes and none of these claims are about it. What is *not* replaced is the walk, so every number
printed below came out of the same code a published projection does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epl.simulate import __main__ as cli
from epl.simulate import projection as projection_module
from epl.simulate.posterior import Diagnostics, Posterior

#: A fabricated Season: twenty Clubs, a full double round robin, one round a week.
CLUBS = tuple(f"club_{index:02d}" for index in range(20))
SEASON = 2024


def _season(season: int = SEASON) -> pd.DataFrame:
    """380 Fixtures over 38 weekly rounds, with a result on every one of them."""
    pairs = [(home, away) for home in CLUBS for away in CLUBS if home != away]
    start = pd.Timestamp("2024-08-17")
    rows = []
    for index, (home, away) in enumerate(pairs):
        kickoff = start + pd.Timedelta(weeks=index // 10)
        rows.append(
            {
                "season": season,
                "division": "E0",
                "date": kickoff.date().isoformat(),
                "time": "",
                "home_club": home,
                "away_club": away,
                "home_goals": (index % 4),
                "away_goals": (index % 3),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def matches(tmp_path: Path) -> Path:
    path = tmp_path / "matches.csv"
    _season().to_csv(path, index=False)
    return path


@pytest.fixture
def cheap_posterior(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the minutes-long fit with eight draws of made-up strengths.

    Patched where :mod:`epl.simulate.projection` bound it, which is the only place a projection
    reaches a sampler: if the module ever grew a second route to one, this fixture would stop
    covering it and a test run would take an hour, which is the right kind of failure.
    """
    rng = np.random.default_rng(0)

    def fake_fit(sample, **_: object) -> Posterior:
        attack = rng.normal(0.0, 0.3, (8, sample.club_count))
        return Posterior(
            clubs=sample.clubs,
            attack=attack - attack.mean(axis=1, keepdims=True),
            defence=rng.normal(0.0, 0.3, (8, sample.club_count)),
            home_advantage=rng.normal(0.25, 0.02, 8),
            correction=np.zeros(8),
            diagnostics=Diagnostics(0, 1.0, 1000.0, 1000.0, 2, 2, 4, 20260825),
        )

    monkeypatch.setattr(projection_module, "fit_posterior", fake_fit)


@pytest.mark.usefixtures("cheap_posterior")
class TestProject:
    def test_it_prints_a_table_with_every_club_and_writes_it(
        self, matches: Path, project_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["--matches", str(matches), "project", "--season", str(SEASON),
                         "--simulations", "300"]) == 0

        printed = capsys.readouterr().out
        written = pd.read_csv(project_root / "outputs" / "projection.csv")

        assert len(written) == 20
        assert written["title"].sum() == pytest.approx(1.0)
        assert all(club in printed for club in CLUBS[:3])

    def test_the_written_file_says_what_it_is_of_and_how_to_reproduce_it(
        self, matches: Path, project_root: Path
    ) -> None:
        """"A fixed deterministic seed recorded in the output" — the output, not the terminal."""
        cli.main(["--matches", str(matches), "project", "--season", str(SEASON),
                  "--simulations", "300", "--seed", "4242"])

        written = pd.read_csv(project_root / "outputs" / "projection.csv")

        assert set(written["season"]) == {SEASON}
        assert set(written["seed"]) == {4242}
        assert set(written["sampler_seed"]) == {20260825}
        assert set(written["simulated_seasons"]) == {300}
        assert written["as_of"].nunique() == 1

    def test_it_prints_both_seeds_and_the_chain_that_settled_the_ties(
        self, matches: Path, project_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli.main(["--matches", str(matches), "project", "--season", str(SEASON),
                  "--simulations", "200", "--seed", "4242"])

        printed = capsys.readouterr().out

        assert "4242" in printed
        assert "20260825" in printed
        assert "head-to-head points" in printed
        assert "coin flip" in printed

    def test_it_says_the_two_fits_are_different_fits(
        self, matches: Path, project_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """ADR 0007's sentence, on the one output most likely to be read beside a scoreboard."""
        cli.main(["--matches", str(matches), "project", "--season", str(SEASON),
                  "--simulations", "200"])

        assert cli.DIFFERENT_FITS in capsys.readouterr().out

    def test_a_checkpoint_the_season_does_not_have_is_refused(self, matches: Path) -> None:
        with pytest.raises(SystemExit, match="checkpoints"):
            cli.main(["--matches", str(matches), "project", "--season", str(SEASON),
                      "--checkpoint", "99"])


@pytest.mark.usefixtures("cheap_posterior")
class TestValidate:
    def test_it_writes_a_row_per_club_per_projection_and_says_what_it_found(
        self, matches: Path, project_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["--matches", str(matches), "validate", "--seasons", str(SEASON),
                         str(SEASON), "--checkpoints", "2", "--simulations", "200"]) == 0

        printed = capsys.readouterr().out
        written = pd.read_csv(project_root / "outputs" / "projection_validation.csv")

        assert len(written) == 2 * 20
        assert set(written["checkpoint"]) == {1, 2}
        assert written["title_happened"].sum() == 2
        assert "where the real champion landed" in printed

    def test_the_season_still_being_played_is_dropped_rather_than_refused(
        self, tmp_path: Path, project_root: Path
    ) -> None:
        """A live Season has no final table to validate against, and that is not an error.

        Refusing would turn "validate the Evaluation Window" into a failure for the whole of every
        Season the project is currently forecasting, which is the one it most wants to run.
        """
        both = pd.concat([_season(), _season(SEASON + 1).head(20)], ignore_index=True)
        path = tmp_path / "two.csv"
        both.to_csv(path, index=False)

        cli.main(["--matches", str(path), "validate", "--checkpoints", "1",
                  "--simulations", "200"])

        written = pd.read_csv(project_root / "outputs" / "projection_validation.csv")

        assert set(written["season"]) == {SEASON}

    def test_it_warns_that_the_reliability_points_are_not_independent(
        self, matches: Path, project_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The one caveat a reader needs before believing a ten-bin diagram over 2,520 rows."""
        cli.main(["--matches", str(matches), "validate", "--seasons", str(SEASON), str(SEASON),
                  "--checkpoints", "1", "--simulations", "200"])

        assert "not independent" in capsys.readouterr().out


class TestCheckpoints:
    def test_it_prints_where_a_season_is_projected_from_and_where_it_is_not(
        self, matches: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["--matches", str(matches), "checkpoints", "--season", str(SEASON)]) == 0

        printed = capsys.readouterr().out

        assert "6 checkpoints out of 38 Prediction Rounds" in printed
        assert "32 rounds get no posterior at all" in printed
