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
from epl.ledger import readings
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

    A `prematch` fire can be worth *several* messages — three matches kick off at three o'clock on a
    Saturday and each gets its own card, because a card is about one match and nothing else. They
    are sent as separate messages rather than one long one for the same reason: a notification
    reading "Chelsea v Brighton" is useful, and one reading "3 matches" is a thing to open later.
    """
    try:
        messages = compose_all(subcommand, found=found)
    except Exception as broke:
        print(f"[epl.bot] could not compose a message: {type(broke).__name__}: {broke}")
        return None
    if not messages:
        return None

    try:
        resolved = settings or Settings.from_environment()
        bot = telegram or api.Telegram(resolved)
        reached = sum(bot.broadcast(message) for message in messages)
    except BotError as unset:
        print(f"[epl.bot] not sending: {unset}")
        return None
    except Exception as unreachable:
        print(f"[epl.bot] send failed: {type(unreachable).__name__}: {unreachable}")
        return None

    print(f"[epl.bot] notified {reached} chat(s) with {len(messages)} message(s)")
    return "\n\n".join(messages)


def compose_all(subcommand: str, *, found: Sequence[fires.Fire] | None = None) -> list[str]:
    """Every message this fire is worth sending, in order. Usually none, sometimes one.

    The list exists for `prematch` alone; every other fire has at most one thing to say, and
    :func:`compose` is the way to ask for it.
    """
    ledger.register_all()
    log = (
        fires.read(subcommand=subcommand) if found is None else tuple(found)
    )
    fire = fires.latest(log, subcommand=subcommand)
    if fire is None:
        return []
    if fire.failed:
        return [answers.failure(fire)]
    if subcommand == "prematch":
        return _about_a_prematch(fire)
    message = _about_one_fire(subcommand, fire, log)
    return [] if message is None else [message]


def compose(subcommand: str, *, found: Sequence[fires.Fire] | None = None) -> str | None:
    """What this fire is worth saying, or ``None``.

    Pure apart from reading the log and two directories' modification times, which is what lets the
    whole decision be tested without a token. A `prematch` fire with several cards comes back
    joined, which is right for a caller asking what this fire had to say and is not how it is
    sent: :func:`run` sends the cards one at a time.
    """
    messages = compose_all(subcommand, found=found)
    return "\n\n".join(messages) if messages else None


def _about_one_fire(
    subcommand: str, fire: fires.Fire, log: Sequence[fires.Fire]
) -> str | None:
    if subcommand == "seal":
        return _about_a_seal(fire, log)
    if subcommand == "score":
        return _about_a_score(fire)
    return None


def _about_a_prematch(fire: fires.Fire) -> list[str]:
    """One card per Fixture this fire read, and nothing at all when it read none.

    Nothing at all is the overwhelmingly common case — forty fires a matchday against ten Fixtures a
    round — which is why this is decided from the artefact rather than the exit code: a Reading
    written since the fire started is the whole question, and both silences exit 0 on purpose
    (:data:`epl.live.prematch.NOTHING_DUE`).

    The Readings are read back from the store rather than parsed out of the log, for the same reason
    a `seal` announcement is: a reworded line must not be able to silence the bot, and the numbers
    in the message have to be the numbers that were written down.
    """
    fresh = _readings_since(fire.started)
    if fresh.empty:
        return []
    cards = [
        answers.prematch_card(one, now=fires.uk_now())
        for _, one in fresh.groupby(["home_club", "away_club"], sort=False)
    ]
    # A card comes back empty when the Predictor the messages are built around said nothing about
    # that Fixture, which is not impossible and is not this module's business to explain. Dropped
    # rather than sent: Telegram refuses an empty message outright, so passing one on would turn a
    # Fixture nobody could forecast into a send failure in the log.
    return [card for card in cards if card.strip()]


def _readings_since(moment: pd.Timestamp) -> pd.DataFrame:
    """Pre-Match Readings written at or after this fire started.

    Cut by the file's modification time rather than by the rows' As-Of Instant, because a day's file
    is appended to: the rows an earlier fire wrote are still in it, and announcing them again would
    send a second card for a match that already had one. Same artefact-not-prose rule as everywhere
    else here, and the same comparison in UTC as :func:`_written_since`.
    """
    day = readings.path(pd.Timestamp(moment).tz_convert(fires.LOCAL_ZONE).tz_localize(None))
    if not _written_since([day], moment):
        return readings.read_day(pd.Timestamp(day.stem)).iloc[0:0]

    held = readings.read_day(pd.Timestamp(day.stem))
    if held.empty:
        return held
    threshold = pd.Timestamp(moment).tz_convert(fires.LOCAL_ZONE).tz_localize(None)
    return held.loc[held["as_of_instant"] >= threshold].reset_index(drop=True)


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


__all__ = ["SCORED_BOARD", "compose", "compose_all", "main", "run"]
