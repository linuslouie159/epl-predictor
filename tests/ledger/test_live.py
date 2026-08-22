"""The sealed store: one file per Prediction Round, written before kickoff and never rewritten.

`outputs/live/` is evidence, not output (CLAUDE.md). What makes it evidence is that git history
proves when each Prediction existed, so the tests that matter here are the ones that run git: a
sealed file whose bytes changed after its round's first kickoff is indistinguishable from a
Prediction written with hindsight, and the point of ADR 0005 is that this be *detectable* rather
than merely discouraged.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from epl.ledger import live, schema
from epl.predictors import Evidence

BEFORE_KICKOFF = "2024-08-16T10:00:00+01:00"
AFTER_KICKOFF = "2024-08-20T10:00:00+01:00"


@pytest.fixture
def round_rows(
    make_matches: Callable[..., pd.DataFrame], make_predictor: Callable[..., object]
) -> pd.DataFrame:
    """One Predictor's Predictions for the round anchored at 2024-08-16."""
    fixtures = make_matches(
        {"date": "2024-08-17", "time": "15:00", "home_club": "arsenal", "away_club": "wolves"},
        {"date": "2024-08-17", "time": "17:30", "home_club": "everton", "away_club": "brighton"},
    )
    evidence = Evidence.before(make_matches({"date": "2024-08-14"}), pd.Timestamp("2024-08-16"))
    return schema.predictions_for(make_predictor(), fixtures, evidence)


def git(root: Path, *args: str, when: str | None = None) -> None:
    """Run one git command in ``root``, optionally stamping the commit at a chosen moment."""
    env = {"GIT_COMMITTER_DATE": when, "GIT_AUTHOR_DATE": when} if when else {}
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        env={**os.environ, **env},
    )


@pytest.fixture
def repo(project_root: Path) -> Path:
    """A git repository at the project root, so the seal check has history to read."""
    git(project_root, "init", "-q")
    git(project_root, "config", "user.email", "test@example.com")
    git(project_root, "config", "user.name", "Test")
    return project_root


class TestSealing:
    def test_a_round_is_written_to_one_file_named_after_it(
        self, project_root: Path, round_rows: pd.DataFrame
    ) -> None:
        sealed = live.seal(round_rows, now=pd.Timestamp("2024-08-16 09:00"))

        assert sealed.name == "2024-08-16.csv"
        assert sealed.parent.name == "live"

    def test_a_sealed_round_is_never_rewritten(
        self, project_root: Path, round_rows: pd.DataFrame
    ) -> None:
        """Append-only means new files, never new bytes in an old one (ADR 0005). A correction is
        a superseding row under a new As-Of Instant."""
        live.seal(round_rows, now=pd.Timestamp("2024-08-16 09:00"))

        with pytest.raises(schema.LedgerError, match="already sealed"):
            live.seal(round_rows, now=pd.Timestamp("2024-08-16 10:00"))

    def test_a_round_cannot_be_sealed_once_it_has_started(
        self, project_root: Path, round_rows: pd.DataFrame
    ) -> None:
        """The deadline is the round's first kickoff. After it, a Prediction is a claim about what
        the code would have said, which is exactly what this store refuses to hold."""
        with pytest.raises(schema.LedgerError, match="first kickoff"):
            live.seal(round_rows, now=pd.Timestamp("2024-08-17 16:00"))

    def test_two_rounds_cannot_be_sealed_in_one_file(
        self,
        project_root: Path,
        round_rows: pd.DataFrame,
        make_matches: Callable[..., pd.DataFrame],
        make_predictor: Callable[..., object],
    ) -> None:
        later = schema.predictions_for(
            make_predictor(),
            make_matches({"date": "2024-08-21"}),
            Evidence.before(make_matches({"date": "2024-08-14"}), pd.Timestamp("2024-08-20")),
        )

        with pytest.raises(schema.LedgerError, match="one Prediction Round"):
            live.seal(
                pd.concat([round_rows, later], ignore_index=True),
                now=pd.Timestamp("2024-08-16 09:00"),
            )

    def test_rows_that_fail_the_audit_are_never_sealed(
        self, project_root: Path, round_rows: pd.DataFrame
    ) -> None:
        round_rows.loc[0, "latest_input"] = round_rows.loc[0, "as_of_instant"]

        with pytest.raises(schema.LedgerError, match="future data"):
            live.seal(round_rows, now=pd.Timestamp("2024-08-16 09:00"))

    def test_every_predictor_in_the_round_shares_the_file(
        self,
        project_root: Path,
        round_rows: pd.DataFrame,
        make_matches: Callable[..., pd.DataFrame],
        make_predictor: Callable[..., object],
    ) -> None:
        """One file per round, not per round and Predictor: the round is what gets sealed."""
        other = round_rows.assign(predictor="dice")

        live.seal(
            pd.concat([round_rows, other], ignore_index=True),
            now=pd.Timestamp("2024-08-16 09:00"),
        )

        assert set(live.read()["predictor"]) == {"fixed", "dice"}


class TestReading:
    def test_an_empty_store_reads_as_an_empty_ledger(self, project_root: Path) -> None:
        assert live.read().empty
        assert list(live.read().columns) == list(schema.LEDGER_COLUMNS)

    def test_what_is_sealed_is_what_is_read_back(
        self, project_root: Path, round_rows: pd.DataFrame
    ) -> None:
        live.seal(round_rows, now=pd.Timestamp("2024-08-16 09:00"))

        pd.testing.assert_frame_equal(live.read(), round_rows)


class TestNothingChangesAfterKickoff:
    """Issue #7: "a test fails if any file under outputs/live/ changes after its round's first
    kickoff". Git history is the proof, so the check reads git rather than the files."""

    def test_a_round_committed_before_kickoff_is_clean(
        self, repo: Path, round_rows: pd.DataFrame
    ) -> None:
        live.seal(round_rows, now=pd.Timestamp("2024-08-16 09:00"))
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", "seal 2024-08-16", when=BEFORE_KICKOFF)

        assert live.seal_violations(now=pd.Timestamp("2024-09-01")) == []

    def test_a_round_committed_again_after_kickoff_is_a_violation(
        self, repo: Path, round_rows: pd.DataFrame
    ) -> None:
        sealed = live.seal(round_rows, now=pd.Timestamp("2024-08-16 09:00"))
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", "seal 2024-08-16", when=BEFORE_KICKOFF)

        sealed.write_text(sealed.read_text(encoding="utf-8").replace("0.5", "0.6"), "utf-8")
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", "tidy up", when=AFTER_KICKOFF)

        assert any("after its round's first kickoff" in c for c in live.seal_violations())

    def test_an_uncommitted_edit_after_kickoff_is_a_violation(
        self, repo: Path, round_rows: pd.DataFrame
    ) -> None:
        sealed = live.seal(round_rows, now=pd.Timestamp("2024-08-16 09:00"))
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", "seal 2024-08-16", when=BEFORE_KICKOFF)
        sealed.write_text(sealed.read_text(encoding="utf-8").replace("0.5", "0.6"), "utf-8")

        violations = live.seal_violations(now=pd.Timestamp("2024-09-01"))

        assert any("uncommitted" in complaint for complaint in violations)

    def test_a_round_never_committed_is_a_violation_once_it_has_kicked_off(
        self, repo: Path, round_rows: pd.DataFrame
    ) -> None:
        """An uncommitted file proves nothing about when it was written, and after kickoff there is
        no longer any way to prove it."""
        live.seal(round_rows, now=pd.Timestamp("2024-08-16 09:00"))

        violations = live.seal_violations(now=pd.Timestamp("2024-09-01"))

        assert any("not committed" in complaint for complaint in violations)

    def test_a_round_not_yet_committed_before_kickoff_is_not_a_violation(
        self, repo: Path, round_rows: pd.DataFrame
    ) -> None:
        """Sealing and committing are two steps, and there is still time between them."""
        live.seal(round_rows, now=pd.Timestamp("2024-08-16 09:00"))

        assert live.seal_violations(now=pd.Timestamp("2024-08-16 10:00")) == []

    def test_a_round_deleted_after_the_fact_is_a_violation(
        self, repo: Path, round_rows: pd.DataFrame
    ) -> None:
        """Deleting a sealed round is the most destructive rewrite there is, and the one a check
        that reads the working tree would miss completely: with the file gone there is nothing
        left to notice. Git still has the record, so the check asks git what should be here."""
        sealed = live.seal(round_rows, now=pd.Timestamp("2024-08-16 09:00"))
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", "seal 2024-08-16", when=BEFORE_KICKOFF)

        sealed.unlink()

        assert any("no longer" in complaint for complaint in live.seal_violations())

    def test_a_round_renamed_after_the_fact_is_a_violation(
        self, repo: Path, round_rows: pd.DataFrame
    ) -> None:
        """A rename is a deletion wearing a hat: the round it was sealed under is gone."""
        sealed = live.seal(round_rows, now=pd.Timestamp("2024-08-16 09:00"))
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", "seal 2024-08-16", when=BEFORE_KICKOFF)

        sealed.rename(sealed.with_name("2024-08-20.csv"))

        assert any("2024-08-16.csv" in complaint for complaint in live.seal_violations())

    def test_a_file_that_is_not_a_sealed_round_is_reported(self, repo: Path) -> None:
        (repo / "outputs" / "live").mkdir(parents=True, exist_ok=True)
        (repo / "outputs" / "live" / "notes.csv").write_text("scratch\n", encoding="utf-8")

        assert any("notes.csv" in c for c in live.seal_violations())


def test_the_sealed_store_in_this_repository_has_never_been_rewritten() -> None:
    """The acceptance criterion itself, run against the real `outputs/live/`.

    Vacuous until the live loop starts sealing rounds in stage 8, and load-bearing from the first
    round it seals — which is the point of writing it now rather than then.
    """
    assert live.seal_violations() == []
