"""The push half: what to say about the fire that has just finished, if anything.

Short-lived and called by `deploy/run_live.sh` after every scheduled run, once the loop's own exit
code has been recorded. It reads the log and the two artefact directories, decides whether there is
news, and sends it. Then it exits 0 — always, whatever happened — because **a notify failure must
never break the run that triggered it** (issue #20's last criterion). Monitoring that can take down
the thing it monitors is worse than no monitoring, and the loop's exit code is a contract several
other things read (`epl.live.__main__._seal`).

**What it says, and how it knows.** Not by parsing the loop's prose. A fire either produced
something or it did not, and that is a fact on disk: a `seal` that sealed leaves a file under
`outputs/live/` newer than the fire, and a `score` that scored leaves `outputs/live_scoreboard.csv`
newer than the fire. Reading artefacts rather than sentences means a reworded message cannot make
this go quiet, which is the one failure a notifier must not have.

**Most fires say nothing, and that is correct.** Two of the three crontab lines are `seal --push`
and the second is a retry designed to find the round already sealed. A bot that announced every
fire would be a bot nobody reads by November — the same argument that made `NothingToSeal` exit 0
in the first place (issue #19).

**It speaks about the fire that happened; it cannot speak about the ones that did not.** So open
risk 7 is reported here — it is a claim about today's two fires and their bytes — and open risk 6 is
not, because "no fire at all" is by definition not something a process the fire started can notice.
That one belongs to :func:`epl.bot.serve.run`, which outlives the fires and reports each gap once;
both show up in `/health` on demand. Splitting them this way is also what stops a fortnight-long gap
being re-announced three times a day until it scrolls out of the lookback.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

import epl.ledger as ledger
from epl.bot import answers, api, fires, watch
from epl.bot.settings import BotError, Settings
from epl.paths import live_dir, outputs_dir

#: What a `score` fire leaves behind when it scored something. `epl.live.__main__._score` writes it
#: only when there is a board to write, so the file being newer than the fire *is* the question
#: "did anything get scored this time?" — read as an artefact rather than off a log line, so that
#: rewording the loop cannot silence the bot. A `seal`'s equivalent is any file under
#: :func:`epl.paths.live_dir`, which is a directory rather than a name and so is not a constant.
SCORED_BOARD = "live_scoreboard.csv"


def run(
    subcommand: str,
    *,
    settings: Settings | None = None,
    telegram: api.Telegram | None = None,
    found: Sequence[fires.Fire] | None = None,
) -> str | None:
    """Send whatever this fire is worth saying, and return the message that was sent.

    ``None`` when there was nothing to say, which is most fires. Never raises: every failure path
    prints and returns, because this is called from inside a scheduled run whose exit code means
    something.
    """
    try:
        message = compose(subcommand, found=found)
    except Exception as broke:
        print(f"[epl.bot] could not compose a message: {type(broke).__name__}: {broke}")
        return None
    if message is None:
        return None

    try:
        resolved = settings or Settings.from_environment()
        bot = telegram or api.Telegram(resolved)
        reached = bot.broadcast(message)
    except BotError as unset:
        print(f"[epl.bot] not sending: {unset}")
        return None
    except Exception as unreachable:
        print(f"[epl.bot] send failed: {type(unreachable).__name__}: {unreachable}")
        return None

    print(f"[epl.bot] notified {reached} chat(s)")
    return message


def compose(subcommand: str, *, found: Sequence[fires.Fire] | None = None) -> str | None:
    """What this fire is worth saying, or ``None``.

    Pure apart from reading the log and two directories' modification times, which is what lets the
    whole decision be tested without a token.
    """
    ledger.register_all()
    log = fires.read() if found is None else tuple(found)
    fire = fires.latest(log, subcommand=subcommand)
    if fire is None:
        return None

    if fire.failed:
        return answers.failure(fire)
    if subcommand == "seal":
        return _about_a_seal(fire, log)
    if subcommand == "score":
        return _about_a_score(fire)
    return None


def _about_a_seal(fire: fires.Fire, log: Sequence[fires.Fire]) -> str | None:
    """A round was sealed, or a round was not and somebody should perhaps know why.

    The retry is the common case and says nothing: it finds the round already sealed, writes no
    file, and the only thing it has left to do is push (`deploy/crontab`, "WHY TWICE").
    """
    if _written_since(_sealed_files(), fire.started):
        return answers.sealed_announcement()
    if fire.verdict in fires.SILENCES:
        concern = watch.stale_upstream(log)
        return None if concern is None else answers.quiet(fire, concern)
    return None


def _about_a_score(fire: fires.Fire) -> str | None:
    """A sealed round has been scored, which happens only once its Fixtures have been played.

    `epl.live.__main__._score` writes the board only when there is one to write, so its file being
    newer than the fire is exactly the question "did anything get scored this time?".
    """
    board = outputs_dir() / SCORED_BOARD
    if not _written_since([board], fire.started):
        return None
    return answers.scored_announcement()


def _sealed_files() -> list[Path]:
    directory = live_dir()
    return sorted(directory.glob("*.csv")) if directory.exists() else []


def _written_since(paths: Sequence[Path], moment: pd.Timestamp) -> bool:
    """Whether any of these files was written at or after ``moment``.

    A fire's stamp carries the machine's offset (:mod:`epl.bot.fires`) and a file's modification
    time is an absolute instant, so the comparison is made in UTC rather than in either wall clock.
    """
    threshold = pd.Timestamp(moment).tz_convert("UTC")
    for path in paths:
        try:
            written = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
        except OSError:
            continue
        if written >= threshold:
            return True
    return False


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m epl.bot notify <subcommand>`. Always exits 0 — see the module docstring."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    subcommand = arguments[0] if arguments else "seal"
    run(subcommand)
    return 0


__all__ = ["SCORED_BOARD", "compose", "main", "run"]
