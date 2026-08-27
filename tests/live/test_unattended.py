"""The contract a schedule needs from the live loop — issue #19's acceptance criteria, executable.

Everything here is about a run nobody is watching. The loop was already built to be scheduled
(:mod:`epl.live`), so most of these are checks that a property the loop *has* cannot be lost; three
are new, and each is new because a schedule asks a question a person at a keyboard never does.

**Which clock.** A person runs the loop on the Friday afternoon they are thinking about. A cron
entry fires at a wall-clock time on a machine whose zone is nobody's decision — a Pi in Singapore,
a container that defaults to UTC. Kickoffs come off Football-Data in UK local time, so the moment a
round's window is judged against has to be UK local too, whatever the machine thinks.

**Which exit code.** A person reads the message. A schedule reads the exit code, and reads it twice
a week for months — so "there was nothing to seal", which is the ordinary case most of the week and
the *only* case until upcoming Fixtures have a source, must not be spelt the same way as "the
rolling file changed shape". One of those is a Tuesday; the other needs somebody.

**Whether it is proven.** A person can see the commit. A schedule on a machine that is not the one
holding the repository has to push, and a sealed round that was written but not pushed is the one
outcome that must be loud (ADR 0005).
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from epl.ingest.fixtures import fixtures_dir
from epl.ingest.football_data import IngestError
from epl.ledger import live as store
from epl.live import __main__ as cli
from epl.live import upcoming
from epl.paths import processed_dir
from epl.windows import LIVE_SEASON

#: Inside the window of the round anchored to Friday 2026-08-28, whose Fixtures are the Saturday.
INSIDE_THE_WINDOW = pd.Timestamp("2026-08-28 14:00:00")

#: Two Premier League Fixtures on the Saturday, in Football-Data's own shape.
ROLLING_CSV = "\r\n".join(
    [
        "Div,Date,Time,HomeTeam,AwayTeam,AvgH,AvgD,AvgA",
        "E0,29/08/2026,15:00,Arsenal,Everton,1.55,4.20,5.50",
        "E0,29/08/2026,17:30,Man City,Chelsea,1.85,3.70,4.10",
        "",
    ]
)

#: What every fetch of the real file has actually held: no row in a tier this project predicts.
#: Measured on three fetches, 21 and 27 August 2026 — docs/DECISIONS.md, "Measured at stage 13".
NO_PREMIER_LEAGUE_CSV = "\r\n".join(
    [
        "Div,Date,Time,HomeTeam,AwayTeam,AvgH,AvgD,AvgA",
        "EC,27/08/2026,19:45,Boreham Wood,Boston Utd,1.54,4.05,4.88",
        "",
    ]
)


@pytest.fixture
def stopped_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "clock", lambda: INSIDE_THE_WINDOW)


@pytest.fixture
def corpus(project_root: Path, make_matches: Callable[..., pd.DataFrame]) -> Path:
    """A match table with the live Season under way and nowhere near finished."""
    matches = make_matches(
        {"season": LIVE_SEASON, "date": "2026-08-21", "home_club": "arsenal",
         "away_club": "coventry", "outcome": "H"},
        {"season": LIVE_SEASON, "date": "2026-08-22", "home_club": "everton",
         "away_club": "chelsea", "outcome": "A"},
        {"season": LIVE_SEASON, "date": "2026-08-24", "home_club": "man_city",
         "away_club": "coventry", "outcome": "H"},
    )
    processed_dir().mkdir(parents=True, exist_ok=True)
    path = processed_dir() / "matches.csv"
    matches.to_csv(path, index=False)
    return path


def _cache(body: str) -> Path:
    """Leave a rolling fixtures file where ``--cached`` will find it."""
    fixtures_dir().mkdir(parents=True, exist_ok=True)
    path = fixtures_dir() / "fixtures_20260828T120000Z.csv"
    path.write_bytes(body.encode("utf-8"))
    return path


@pytest.fixture
def rolling(project_root: Path) -> Path:
    return _cache(ROLLING_CSV)


@pytest.fixture
def nothing_we_predict(project_root: Path) -> Path:
    return _cache(NO_PREMIER_LEAGUE_CSV)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(project_root: Path) -> Path:
    """A git repository at the project root, so a sealed round can be committed."""
    for command in (
        ("init", "-q", "-b", "main"),
        ("config", "user.email", "test@example.com"),
        ("config", "user.name", "Test"),
    ):
        _git(project_root, *command)
    return project_root


@pytest.fixture
def remote(repo: Path, tmp_path: Path) -> Path:
    """A bare repository for ``origin``, so a push has somewhere real to land."""
    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    _git(repo, "remote", "add", "origin", str(bare))
    return bare


def _commits(root: Path, ref: str = "HEAD") -> int:
    """How many commits ``ref`` has. A bare repository's HEAD points at whatever branch git
    defaults to, which is not the one anything was pushed to, so a remote is counted by name."""
    return len(_git(root, "log", "--oneline", ref).splitlines())


class TestWhichClock:
    """Criterion 7's other half. The flag it forbids is not the only way to seal a round at the
    wrong moment: a machine in the wrong zone does it silently, and without anybody choosing to.

    **What these can and cannot catch, stated because it is easy to over-read them.** On a machine
    already in Europe/London, `pd.Timestamp.now()` and :func:`epl.ledger.live.uk_now` return the
    same value, so no test written here can tell the fixed code from the broken code there. What
    the first test below does catch is a reversion on any machine that is *not* in the UK — which
    is the build machine (UTC+8) and every runner that defaults to UTC. That is the population the
    bug was found in, and it is the population a schedule runs in.

    The expectation is therefore derived from **UTC**, not from `Europe/London` local time. Asking
    pandas for the London wall clock and comparing it against a function that asks pandas for the
    London wall clock would pass whatever either did.
    """

    def test_the_clock_is_the_uk_reading_of_the_current_utc_instant(self) -> None:
        """Kickoffs are recorded in UK local time (:data:`epl.ledger.live.LOCAL_ZONE`), and the
        window is judged by comparing against them. On the machine this was written on, eight hours
        ahead, `pd.Timestamp.now()` said 2026-08-28 00:17 where the UK said 2026-08-27 17:17 — a
        different *day*, and the wrong side of a Friday round's midnight As-Of Instant.

        ``pd.Timestamp.now(tz="UTC")`` names the same instant whatever the machine's zone is, so
        adding London's offset to it is an expectation this test does not share any code path with.
        """
        utc = pd.Timestamp.now(tz="UTC")
        offset = ZoneInfo(store.LOCAL_ZONE).utcoffset(utc.to_pydatetime())
        assert offset is not None
        expected = utc.tz_localize(None) + offset

        assert abs(store.uk_now() - expected) < pd.Timedelta(seconds=5)

    def test_it_is_naive_because_everything_it_is_compared_against_is(self) -> None:
        """A tz-aware moment cannot be compared with a naive kickoff at all — pandas raises. The
        conversion has to land back on a naive UK wall-clock reading."""
        assert cli.clock().tz is None

    def test_the_command_line_has_not_grown_a_clock_of_its_own(self) -> None:
        """Two clocks an hour apart would let :func:`epl.ledger.live.seal` refuse a round
        :mod:`epl.live.upcoming` had just offered, which is the failure `window` is written once to
        prevent. This is what fails if `clock` is ever put back to reading the machine's zone —
        on any machine outside the UK, which is where it will be run."""
        assert abs(cli.clock() - store.uk_now()) < pd.Timedelta(seconds=5)


class TestNothingToSeal:
    """Criterion 4. A schedule that fails twice a week forever is a schedule nobody reads."""

    def test_a_file_with_no_premier_league_row_is_a_quiet_success(
        self, corpus: Path, nothing_we_predict: Path, repo: Path, stopped_clock: None,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The only case there has ever been. Three fetches of the real file, on 21 and twice on 27
        August 2026, held no E0 row at all, so until upcoming Fixtures have a confirmed source this
        is what every single fire of the schedule will do."""
        assert cli.main(["seal", "--cached"]) == 0

        assert "no row in a tier this project predicts" in capsys.readouterr().out
        assert store.sealed_rounds() == []

    def test_a_clock_outside_every_window_is_a_quiet_success(
        self, corpus: Path, rolling: Path, repo: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The ordinary case the rest of the week: the file holds a round, and its window has
        already shut. Nothing is wrong, and nothing is to be done."""
        monkeypatch.setattr(cli, "clock", lambda: pd.Timestamp("2026-08-29 20:00:00"))

        assert cli.main(["seal", "--cached"]) == 0

    def test_a_season_the_corpus_cannot_place_is_still_a_loud_failure(
        self, project_root: Path, rolling: Path, repo: Path, stopped_clock: None,
        make_matches: Callable[..., pd.DataFrame], capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The distinction the exit code exists to draw. A stale
        :data:`epl.windows.LIVE_SEASON` is not a quiet Tuesday — it is the loop about to stamp
        Fixtures with the wrong Season, and somebody has to look at it."""
        processed_dir().mkdir(parents=True, exist_ok=True)
        make_matches({"season": LIVE_SEASON - 1}).to_csv(
            processed_dir() / "matches.csv", index=False
        )

        assert cli.main(["seal", "--cached"]) == 1

        assert "not under way" in capsys.readouterr().out

    def test_a_rolling_file_that_changed_shape_is_a_loud_failure(
        self, corpus: Path, project_root: Path, repo: Path, stopped_clock: None,
    ) -> None:
        """Upstream dropping a column is the other thing that must never read as a quiet Tuesday.

        It is loud in the loudest way available: :class:`epl.ingest.football_data.IngestError` is
        not a :class:`~epl.live.upcoming.LiveError` and is deliberately not caught here. The shape
        of the rolling file is :mod:`epl.ingest`'s to complain about, and a schedule reading cron's
        mail gets the missing column named. Widening the handler would put the one failure that
        means "upstream changed" behind the same exit code as "it is Wednesday".
        """
        _cache("Div,Date,HomeTeam\r\nE0,29/08/2026,Arsenal\r\n")

        with pytest.raises(IngestError, match=r"missing required columns \['AwayTeam'\]"):
            cli.main(["seal", "--cached"])

    def test_upcoming_says_the_same_thing_and_still_writes_nothing(
        self, corpus: Path, nothing_we_predict: Path, stopped_clock: None,
    ) -> None:
        """`upcoming` is what a person runs to ask the question. It agrees with `seal` about
        whether the answer is a problem."""
        assert cli.main(["upcoming", "--cached"]) == 0

        assert store.sealed_rounds() == []


class TestFiringTwiceInsideOneRound:
    """Criterion 3. The schedule fires at 16:00 and again at 18:30, on purpose: a retry inside the
    window is what makes a transient upstream failure survivable, and it is only free if the second
    fire is genuinely inert."""

    def test_the_second_fire_adds_no_commit(
        self, corpus: Path, rolling: Path, repo: Path, stopped_clock: None,
    ) -> None:
        cli.main(["seal", "--cached"])
        after_first = _commits(repo)

        assert cli.main(["seal", "--cached"]) == 0

        assert _commits(repo) == after_first

    def test_the_second_fire_leaves_the_bytes_alone(
        self, corpus: Path, rolling: Path, repo: Path, stopped_clock: None,
    ) -> None:
        cli.main(["seal", "--cached"])
        before = store.path("2026-08-28").read_bytes()

        cli.main(["seal", "--cached"])

        assert store.path("2026-08-28").read_bytes() == before


class TestPushing:
    """Criterion 5, on a machine that is not the one holding the repository.

    A commit in a checkout on a Pi is real history, and it is history nobody else can see. The
    round is evidence once it has left the machine that made it, so a push that fails has to be as
    loud as a commit that fails — the file on disk looks identical either way.
    """

    def test_a_sealed_round_is_pushed(
        self, corpus: Path, rolling: Path, repo: Path, remote: Path, stopped_clock: None,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert cli.main(["seal", "--cached", "--push"]) == 0

        assert "pushed to origin/main" in capsys.readouterr().out
        assert _commits(remote, "main") == 1

    def test_a_push_that_fails_is_loud(
        self, corpus: Path, rolling: Path, repo: Path, stopped_clock: None,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """No remote configured at all, which is what a missing deploy key looks like from here."""
        assert cli.main(["seal", "--cached", "--push"]) == 1

        printed = capsys.readouterr().out
        assert "NOT PUSHED" in printed
        assert store.path("2026-08-28").exists(), "the round is still sealed and committed"

    def test_the_retry_fire_pushes_a_round_the_first_fire_could_not(
        self, corpus: Path, rolling: Path, repo: Path, tmp_path: Path, stopped_clock: None,
    ) -> None:
        """The 18:30 fire earning its place. The 16:00 fire sealed and committed but could not
        reach the remote; the round is already sealed, so the second fire has nothing to write —
        and pushing is the whole of what is left to do."""
        assert cli.main(["seal", "--cached", "--push"]) == 1
        bare = tmp_path / "origin.git"
        subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
        _git(repo, "remote", "add", "origin", str(bare))

        assert cli.main(["seal", "--cached", "--push"]) == 0

        assert _commits(bare, "main") == 1

    def test_pushing_without_committing_is_refused_rather_than_resolved(
        self, corpus: Path, rolling: Path, repo: Path, stopped_clock: None,
    ) -> None:
        """The one combination that could report success over an unproven seal: the round is
        written and deliberately not committed, and the push then advances a branch that does not
        contain it — `pushed to origin/main`, exit 0, and the round sitting in the working tree."""
        with pytest.raises(SystemExit) as refused:
            cli.main(["seal", "--cached", "--no-commit", "--push"])

        assert refused.value.code == 2

    def test_pushing_is_opt_in(
        self, corpus: Path, rolling: Path, repo: Path, stopped_clock: None,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A push is an outward-facing act on somebody's repository, so the schedule asks for it
        and a person at a keyboard does not get it by surprise."""
        assert cli.main(["seal", "--cached"]) == 0

        assert "PUSHED" not in capsys.readouterr().out


class TestTheClockIsNotAnOption:
    """Criterion 7, stated as a test rather than as a comment.

    :func:`epl.live.__main__.clock` is a function so that a test can stop it and an operator
    cannot. An operator who could name the moment could seal a round after its own kickoff and have
    the file say otherwise, which is the one claim this store exists to keep true.
    """

    def test_seal_takes_no_now_flag(
        self, corpus: Path, rolling: Path, repo: Path, stopped_clock: None,
    ) -> None:
        with pytest.raises(SystemExit) as refused:
            cli.main(["seal", "--cached", "--now", "2026-08-28 14:00:00"])

        assert refused.value.code == 2

    def test_no_command_takes_one(self) -> None:
        """Checked across the parser rather than on `seal` alone: the flag would be just as
        dangerous on a command that goes on to write."""
        for command in ("upcoming", "seal", "score"):
            with pytest.raises(SystemExit):
                cli.main([command, "--now", "2026-08-28 14:00:00"])


class TestNothingToSealIsItsOwnRefusal:
    """The seam the exit codes rest on. `seal` does not read messages to decide what happened.

    Which of the two silences a given refusal is stays :mod:`epl.live.upcoming`'s business — it is
    the module that already tells them apart in its complaint — and the two cases raising it are
    checked there, beside the rest of the window's behaviour.
    """

    def test_it_is_a_live_error_so_nothing_that_catches_those_stops_catching_it(self) -> None:
        assert issubclass(upcoming.NothingToSeal, upcoming.LiveError)
