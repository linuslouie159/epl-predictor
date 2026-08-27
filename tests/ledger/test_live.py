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


#: The round every test here seals: two Fixtures on Saturday 17 August 2024, anchored to the Friday.
ROUND_FIXTURES: tuple[dict[str, object], ...] = (
    {"date": "2024-08-17", "time": "15:00", "home_club": "arsenal", "away_club": "wolves"},
    {"date": "2024-08-17", "time": "17:30", "home_club": "everton", "away_club": "brighton"},
)


@pytest.fixture
def round_rows(
    make_matches: Callable[..., pd.DataFrame], make_predictor: Callable[..., object]
) -> pd.DataFrame:
    """One Predictor's Predictions for the round anchored at 2024-08-16."""
    fixtures = make_matches(*ROUND_FIXTURES)
    evidence = Evidence.before(make_matches({"date": "2024-08-14"}), pd.Timestamp("2024-08-16"))
    return schema.predictions_for(make_predictor(), fixtures, evidence)


@pytest.fixture
def correction(
    make_matches: Callable[..., pd.DataFrame], make_predictor: Callable[..., object]
) -> Callable[..., pd.DataFrame]:
    """The same round predicted again later — what a bug fix found before kickoff produces."""

    def _make(
        at: str | pd.Timestamp, probabilities: tuple[float, float, float] = (0.2, 0.3, 0.5)
    ) -> pd.DataFrame:
        instant = pd.Timestamp(at)
        return schema.predictions_for(
            make_predictor(probabilities=probabilities),
            make_matches(*ROUND_FIXTURES),
            Evidence.before(make_matches({"date": "2024-08-14"}), instant),
            as_of=instant,
        )

    return _make


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

    def test_a_round_cannot_be_sealed_before_its_own_as_of_instant(
        self, project_root: Path, round_rows: pd.DataFrame
    ) -> None:
        """The other end of the same window. Sealing on the Thursday under Friday's midnight
        instant claims a moment that has not happened, and reads odds that do not exist yet."""
        with pytest.raises(schema.LedgerError, match="has not happened yet"):
            live.seal(round_rows, now=pd.Timestamp("2024-08-15 18:00"))

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


class TestSuperseding:
    """ADR 0005's "adding a superseding row with a new As-Of Instant, never editing history".

    The store refuses to rewrite a sealed file, so a genuine bug found before kickoff has to have
    somewhere else to go. It goes into a new revision of the same round, stamped at the moment the
    correction was actually made — which is later, and is allowed to know more.
    """

    def test_a_correction_is_a_new_file_and_the_original_is_untouched(
        self, project_root: Path, round_rows: pd.DataFrame, correction: Callable[..., pd.DataFrame]
    ) -> None:
        sealed = live.seal(round_rows, now=pd.Timestamp("2024-08-16 09:00"))
        before = sealed.read_bytes()

        at = pd.Timestamp("2024-08-16 14:00")
        revised = live.supersede(correction(at), now=at)

        assert revised.name == "2024-08-16.1.csv"
        assert sealed.read_bytes() == before

    def test_a_round_never_sealed_cannot_be_superseded(
        self, project_root: Path, correction: Callable[..., pd.DataFrame]
    ) -> None:
        with pytest.raises(schema.LedgerError, match="nothing to supersede"):
            live.supersede(correction("2024-08-16 14:00"), now=pd.Timestamp("2024-08-16 14:00"))

    def test_a_correction_must_be_stamped_later_than_what_it_replaces(
        self, project_root: Path, round_rows: pd.DataFrame, correction: Callable[..., pd.DataFrame]
    ) -> None:
        """A replacement claiming the same instant is indistinguishable from the original having
        been rewritten, which is the one thing this store exists to make impossible."""
        live.seal(round_rows, now=pd.Timestamp("2024-08-16 09:00"))

        with pytest.raises(schema.LedgerError, match="new As-Of Instant"):
            live.supersede(correction("2024-08-16"), now=pd.Timestamp("2024-08-16 14:00"))

    def test_a_correction_cannot_be_stamped_before_the_round_it_corrects(
        self, project_root: Path, make_matches: Callable[..., pd.DataFrame],
        make_predictor: Callable[..., object]
    ) -> None:
        """Superseding moves the instant forward. Moving it back would claim to have predicted
        before the market's odds were sampled."""
        with pytest.raises(schema.LedgerError, match="before its own round"):
            schema.predictions_for(
                make_predictor(),
                make_matches(*ROUND_FIXTURES),
                Evidence.before(make_matches({"date": "2024-08-14"}), pd.Timestamp("2024-08-15")),
                as_of=pd.Timestamp("2024-08-15"),
            )

    def test_a_round_that_has_kicked_off_cannot_be_corrected_either(
        self, project_root: Path, round_rows: pd.DataFrame, correction: Callable[..., pd.DataFrame]
    ) -> None:
        """The rule, not a limitation — and the case most likely to be argued with, because a bug
        is usually found at scoring time. This store holds what was forecast *before* kickoff, so a
        row written after it could not be evidence of that whatever it said. The sealed row stands
        and is scored as made; the fix goes into the code, so the next round is right."""
        live.seal(round_rows, now=pd.Timestamp("2024-08-16 09:00"))

        with pytest.raises(schema.LedgerError, match="first kickoff"):
            live.supersede(correction("2024-08-17 16:00"), now=pd.Timestamp("2024-08-17 16:00"))

    def test_both_readings_are_read_back_with_the_correction_last(
        self, project_root: Path, round_rows: pd.DataFrame, correction: Callable[..., pd.DataFrame]
    ) -> None:
        """Name order is not time order once a revision exists — `2024-08-16.1.csv` sorts before
        `2024-08-16.csv` as a string — so the store sorts on what it parses out of the name."""
        live.seal(round_rows, now=pd.Timestamp("2024-08-16 09:00"))
        live.supersede(correction("2024-08-16 14:00"), now=pd.Timestamp("2024-08-16 14:00"))

        stored = live.read()

        assert len(stored) == 4
        assert list(stored["as_of_instant"].unique()) == [
            pd.Timestamp("2024-08-16"),
            pd.Timestamp("2024-08-16 14:00"),
        ]

    def test_a_correction_keeps_the_round_its_fixtures_anchor_to(
        self, project_root: Path, round_rows: pd.DataFrame, correction: Callable[..., pd.DataFrame]
    ) -> None:
        """A round is a property of the Fixture, not of when somebody got around to predicting it,
        so the audit's round check still passes on a row stamped hours later."""
        live.seal(round_rows, now=pd.Timestamp("2024-08-16 09:00"))
        revised = live.supersede(
            correction("2024-08-16 14:00"), now=pd.Timestamp("2024-08-16 14:00")
        )

        assert set(schema.read_csv(revised)["prediction_round"]) == {"2024-08-16"}
        assert schema.audit(live.read()) == []

    def test_two_files_repeating_one_instant_are_a_violation(
        self, repo: Path, round_rows: pd.DataFrame
    ) -> None:
        """A rewrite done by adding a file rather than editing one. `schema.audit` sees inside one
        file; this is the check across the store."""
        live.seal(round_rows, now=pd.Timestamp("2024-08-16 09:00"))
        schema.write_csv(round_rows, live.path("2024-08-16", revision=1))

        assert any("repeat a Fixture at one As-Of Instant" in c for c in live.seal_violations())


class TestCommitting:
    """An uncommitted sealed file proves nothing, so the loop commits what it seals."""

    def test_a_sealed_round_is_committed_and_the_hash_comes_back(
        self, repo: Path, round_rows: pd.DataFrame
    ) -> None:
        sealed = live.seal(round_rows, now=pd.Timestamp("2024-08-16 09:00"))

        committed = live.commit([sealed], message="Seal 2024-08-16")

        assert committed is not None
        logged = subprocess.run(
            ["git", "-C", str(repo), "log", "--format=%H %s", "--", str(sealed)],
            check=True, capture_output=True, text=True,
        ).stdout
        assert logged.startswith(committed)
        assert "Seal 2024-08-16" in logged

    def test_committing_unchanged_bytes_again_is_a_no_op(
        self, repo: Path, round_rows: pd.DataFrame
    ) -> None:
        """The weekly loop is expected to be run more than once in a round; a second run must not
        add an empty commit to the evidence."""
        sealed = live.seal(round_rows, now=pd.Timestamp("2024-08-16 09:00"))
        live.commit([sealed], message="Seal 2024-08-16")

        assert live.commit([sealed], message="Seal 2024-08-16 again") is None

    def test_only_the_named_file_is_staged(self, repo: Path, round_rows: pd.DataFrame) -> None:
        """Sealing a round must never sweep up whatever else was in the working tree."""
        sealed = live.seal(round_rows, now=pd.Timestamp("2024-08-16 09:00"))
        stray = repo / "scratch.txt"
        stray.write_text("not evidence\n", encoding="utf-8")

        live.commit([sealed], message="Seal 2024-08-16")

        assert "scratch.txt" in subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout


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
