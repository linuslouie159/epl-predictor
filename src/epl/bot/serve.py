"""The pull half: a long-polling bot that answers questions and can change nothing.

Started by `python -m epl.bot serve` and kept running by compose (`deploy/docker-compose.yml`,
service `bot`). Unlike the live loop's containers, which exist for the length of one command and
whose exit code is their whole interface, this one is long-lived — so it carries
``restart: unless-stopped`` and comes back after a reboot, which is issue #20's ninth criterion.

**Every handler is read-only, and there is no argument that can make one otherwise.** The dispatch
table below is the bot's entire vocabulary and none of it is a verb. That is ADR 0005 enforced at
the door: a Prediction in `outputs/live/` is evidence because the loop wrote it before kickoff under
a moment nobody could choose, and a chat message is the easiest imaginable way to write one that was
not. `tests/bot/test_the_bot_is_read_only.py` checks it structurally, by walking imports.

**The allowlist refuses silently.** An unlisted id gets nothing — not a refusal, not an error, not
a "you are not authorised". A reply confirms both that the bot exists and that the id reached it,
which is the one thing an unlisted id should not learn; a bot username is guessable, and this one
can read a machine's log.

**Two guards against a second poller, catching different things.** Telegram hands each update to
whichever poller asked first, so a forgotten instance does not fail visibly: it steals half the
replies, and the half that arrives makes the bot look like it is working. :func:`single_instance`
takes an OS lock and catches a second process on this machine *before* it has polled at all;
:class:`epl.bot.api.Conflict` catches one on a machine no lock can see, and is fatal rather than
retried — retrying would leave both alive and both wrong.

**It also watches the schedule while it waits.** Long polling is idle time, so every
:data:`CHECK_SECONDS` the loop asks :mod:`epl.bot.watch` whether anything has gone quiet, and says
so unprompted. That is the only way open risk 6 is ever reported without somebody thinking to ask
— and it is retrospective by nature, because a Pi that is off runs no bot either.
"""

from __future__ import annotations

import contextlib
import sys
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import epl.ledger as ledger
from epl.bot import answers, api, fires, watch
from epl.bot.settings import BotError, Settings
from epl.paths import project_root

#: Where the single-instance lock lives: beside the loop's own lock and its log, in the one
#: directory this project keeps machine-local runtime in. The directory comes from
#: :data:`epl.bot.fires.RUNTIME_DIR` rather than being spelt again, because a lock written somewhere
#: other than where the schedule looks is a lock that guards nothing.
LOCK_RELATIVE_PATH = (*fires.RUNTIME_DIR, "telegram_bot.lock")

#: How often the loop looks at the schedule's health unprompted. Six hours: often enough that a
#: missed Friday is heard about the same evening, rare enough that a quiet week is quiet.
CHECK_SECONDS = 6 * 60 * 60

#: How long to wait after a transport failure before polling again. Telegram being briefly
#: unreachable is a Tuesday; a tight retry loop against it is a way to get rate-limited.
BACKOFF_SECONDS = 30


class AlreadyRunning(BotError):
    """Another instance holds the lock on this machine."""


@dataclass(frozen=True)
class Command:
    """One slash command: its name, the "/" menu's summary, and what it answers.

    ``answer`` takes whatever followed the command and the moment, so a handler that needs neither
    ignores both — and there is one signature rather than a special case per command.
    """

    name: str
    summary: str
    answer: Callable[[str, pd.Timestamp], str]


#: The bot's whole vocabulary. This tuple is both the dispatch table and the "/" menu registered
#: with Telegram, so the two cannot drift — a command that answers and is not offered is invisible,
#: and one that is offered and does not answer is a bug reported by a phone.
#:
#: **Ordered for a reader, not alphabetically.** The two commands somebody opens the app to use are
#: first, the retrospective pair next, and the machine's own health last — because this tuple is
#: also the "/" menu Telegram renders, and a menu whose first entry is a diagnostic is a menu
#: written for the person who built the bot.
COMMANDS: tuple[Command, ...] = (
    Command(
        "next",
        "The next match and its odds",
        lambda argument, now: answers.next_match(now),
    ),
    Command(
        "week",
        "Every match in the current round",
        lambda argument, now: answers.round_digest(argument or None),
    ),
    Command(
        "disagree",
        "Model against the bookmakers",
        lambda argument, now: answers.disagreements(argument or None),
    ),
    Command(
        "club",
        "One club, e.g. /club arsenal",
        lambda argument, now: answers.for_club(argument, now),
    ),
    Command(
        "results",
        "How the last round turned out",
        lambda argument, now: answers.last_results(),
    ),
    Command(
        "record",
        "How the forecasts have scored",
        lambda argument, now: answers.live_record(),
    ),
    Command(
        "board",
        "The Evaluation Window scoreboard",
        lambda argument, now: answers.evaluation_board(),
    ),
    Command(
        "explain",
        "What the numbers mean",
        lambda argument, now: answers.explain(),
    ),
    Command(
        "health",
        "Is the schedule still running?",
        # Every log, not just the loop's: `prematch` writes to its own file (`epl.bot.fires.LOGS`)
        # and a reader asking whether the schedule is alive means all of it.
        lambda argument, now: answers.health(fires.read_every(), now=now),
    ),
    Command(
        "help",
        "This menu",
        lambda argument, now: answers.help_text(COMMANDS),
    ),
)


#: Spellings that mean one of the commands above. Deliberately not entries in :data:`COMMANDS`,
#: because that tuple is also the "/" menu and a menu that lists three names for one answer is a
#: menu that makes the bot look bigger than it is.
#:
#: `/start` is there because it is the *first* thing anybody sends a bot: the button Telegram shows
#: on an unopened conversation sends exactly that, and a first reply of "No such command" is a poor
#: account of a bot that in fact works. Measured on the first real deployment (issue #20), where
#: pressing Start was also the step that let the push half deliver at all.
#:
#: `/round` and `/live` are the names `/week` and `/record` replaced. They are kept as aliases
#: rather than deleted for the reason `/start` is here at all: they are in one person's muscle
#: memory and in this repository's own documentation, and a bot that answers "No such command" to
#: the name it used last week is a bot that looks broken rather than tidied.
ALIASES: dict[str, str] = {
    "start": "help",
    "round": "week",
    "live": "record",
    "commands": "help",
}


def dispatch(text: str, *, now: pd.Timestamp) -> str | None:
    """The answer to one message, or ``None`` when it was not a command at all.

    ``None`` rather than a complaint, because a bot that replied to every sentence in a group chat
    would be unusable there. An unknown *command* does get an answer: somebody typed a slash and
    meant something by it.
    """
    if not text.startswith("/"):
        return None
    head, _, argument = text[1:].partition(" ")
    # Telegram addresses a command to one bot in a group as `/help@epl_predictor_bot`. A bot that
    # matched only the bare form would look broken in the one place several people can see it.
    name = head.split("@", 1)[0].strip().lower()
    name = ALIASES.get(name, name)
    for command in COMMANDS:
        if command.name == name:
            return command.answer(argument.strip(), now)
    return f"No such command: /{name}. Try /help."


def handle(
    update: dict, settings: Settings, telegram: api.Telegram, *, now: pd.Timestamp
) -> None:
    """Answer one Telegram update, if the sender is allowed and said something to answer.

    A handler that raises is reported to the sender rather than propagated: the loop must survive
    its own answers, or the monitoring goes down before the thing it monitors.
    """
    body = update.get("message") or {}
    sender = (body.get("from") or {}).get("id")
    if sender is None or not settings.allows(int(sender)):
        print(f"[epl.bot] refused user id={sender}")
        return

    text = str(body.get("text") or "")
    # The chat, falling back to the sender: a private conversation numbers the chat after the user,
    # and a group message that got this far is from an allow-listed sender either way.
    chat = int((body.get("chat") or {}).get("id") or sender)
    try:
        answer = dispatch(text, now=now)
    except Exception as broke:
        answer = f"That failed: {type(broke).__name__}: {broke}"
    if answer is not None:
        telegram.send(chat, answer)


def poll_once(
    telegram: api.Telegram,
    settings: Settings,
    *,
    offset: int | None,
    now: pd.Timestamp,
) -> int | None:
    """One long poll, every update in it handled, and the offset to ask with next.

    Telegram redelivers anything unacknowledged, so an offset that failed to advance is a bot
    answering the same message for ever — which is why it moves past an update that *raised* as
    surely as past one that answered.
    """
    batch = telegram.updates(offset)
    for update in batch:
        handle(update, settings, telegram, now=now)
        offset = int(update["update_id"]) + 1
    return offset


def run(
    settings: Settings,
    *,
    telegram: api.Telegram | None = None,
    forever: bool = True,
) -> int:
    """Poll until stopped. Returns the process exit code.

    ``forever=False`` runs a single poll, which is what a test and a smoke check want. The lock is
    taken by :func:`main` rather than here, so a caller that already holds it — or a test that does
    not need it — is not fighting itself for it.
    """
    ledger.register_all()
    bot = telegram or api.Telegram(settings)

    username = bot.whoami()
    print(f"[epl.bot] {username or 'unknown bot'}: {settings.redacted()}")
    bot.set_commands([(command.name, command.summary) for command in COMMANDS])

    offset: int | None = None
    announced: set[tuple[int, str]] = set()
    # `None` means the health check is due now rather than in six hours. A bot that has just
    # started is a bot whose machine may have just come back, which is when open risk 6 is most
    # likely to be true and least likely to be looked for.
    checked: float | None = None
    while True:
        try:
            offset = poll_once(bot, settings, offset=offset, now=fires.uk_now())
        except api.Conflict as clash:
            print(f"[epl.bot] STOPPING: {clash}")
            return 1
        except api.TransportError as unreachable:
            print(f"[epl.bot] {unreachable}; retrying in {BACKOFF_SECONDS}s")
            if not forever:
                return 1
            time.sleep(BACKOFF_SECONDS)
            continue

        if checked is None or time.monotonic() - checked >= CHECK_SECONDS:
            checked = time.monotonic()
            _report(bot, announced)
        if not forever:
            return 0


def _report(bot: api.Telegram, announced: set[tuple[int, str]]) -> None:
    """Say anything the schedule has gone quiet about, and say each thing once.

    ``announced`` is in memory only and is lost on a restart, so a restarted bot repeats itself.
    That is the right way round: a restart is the moment open risk 6 is *most* likely to be true,
    since the commonest reason this process stopped is the machine it runs on having stopped.

    **The loop's log only, where `/health` reads every log, and the asymmetry is deliberate.**
    Both concerns are claims about `seal` fires — which anchor days went by unfired, and whether
    a round's two fires read the same stale bytes — and :func:`epl.bot.watch.absent` measures
    its window from the *first* fire it is given. Handing it `prematch` fires would move that
    window's start to
    whenever the half-hourly schedule was installed, for no gain: they are filtered out immediately
    afterwards. `/health` is a different question, asked by a person, and means all of it.
    """
    for concern in watch.concerns(fires.read(), now=fires.uk_now()):
        key = (concern.risk, concern.headline)
        if key not in announced:
            announced.add(key)
            bot.broadcast(concern.message())


@contextlib.contextmanager
def single_instance(path: Path | None = None) -> Generator[Path]:
    """Hold an exclusive OS lock for the life of the bot, or refuse to start.

    Two pollers on one token do not conflict loudly: Telegram gives each update to whoever asked
    first, so the second instance eats half the replies and the first goes on looking healthy. This
    catches that before a single poll is made — which is the difference between a clear refusal and
    an afternoon spent wondering why a code change "did not take effect".

    It sees only this machine. The one on another host is caught by :class:`epl.bot.api.Conflict`.
    """
    lock = path or project_root().joinpath(*LOCK_RELATIVE_PATH)
    lock.parent.mkdir(parents=True, exist_ok=True)
    handle_ = lock.open("w")
    try:
        _take(handle_, lock)
        yield lock
    finally:
        handle_.close()


def _take(handle_: object, lock: Path) -> None:
    """The platform's own exclusive, non-blocking file lock.

    Both spellings, because this is developed on Windows and deployed on a Pi, and a lock that
    worked in only one of those places would be absent exactly where the bot runs unattended.
    """
    try:
        import msvcrt

        msvcrt.locking(handle_.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
        return
    except ImportError:
        pass
    except OSError as held:
        raise AlreadyRunning(_held(lock)) from held

    import fcntl

    try:
        fcntl.flock(handle_.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[attr-defined]
    except OSError as held:
        raise AlreadyRunning(_held(lock)) from held


def _held(lock: Path) -> str:
    return (
        f"another epl.bot instance holds {lock}. Only one may poll this token: two split the "
        "updates between them silently, so the second would steal half the replies rather than "
        "fail. Stop the other one first."
    )


def main(settings: Settings | None = None) -> int:
    """Take the lock, then poll. What `python -m epl.bot serve` calls."""
    try:
        resolved = settings or Settings.from_environment()
    except BotError as refused:
        print(refused, file=sys.stderr)
        return 1
    try:
        with single_instance():
            return run(resolved)
    except AlreadyRunning as clash:
        print(clash, file=sys.stderr)
        return 1


__all__ = [
    "ALIASES",
    "BACKOFF_SECONDS",
    "CHECK_SECONDS",
    "COMMANDS",
    "LOCK_RELATIVE_PATH",
    "AlreadyRunning",
    "Command",
    "dispatch",
    "handle",
    "main",
    "poll_once",
    "run",
    "single_instance",
]
