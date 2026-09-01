"""Command line entry point for the Telegram bot.

    python -m epl.bot serve            poll for commands until stopped — the long-lived half
    python -m epl.bot notify seal      say what the fire that just ran did, if anything
    python -m epl.bot check            print what the bot would say about the schedule's health
    python -m epl.bot answer /week     print one command's answer, as plain text

``serve`` is what `deploy/docker-compose.yml`'s `bot` service runs, and it is the only long-lived
process this project has. ``notify`` is what `deploy/run_live.sh` calls after every scheduled fire
and **always exits 0**, because a notify failure must never break the run that triggered it.

``check`` sends nothing. It is the answer to "is this thing wired up?" on a machine where sending
a real message to prove it would be the worst possible way to find out — and it needs neither a
token nor a network, so it is also what to run before either exists.

``answer`` sends nothing either, and exists because these messages are laid out to a character
budget: a table that wraps on a phone has lost the alignment it was built for, and finding that out
by sending a hundred messages to a real chat is a poor way to iterate. It goes through
:func:`epl.bot.serve.dispatch`, so what it prints is literally what the command would reply — a
preview that composed its own answer could look right while the command was broken.
"""

from __future__ import annotations

import argparse
import sys

import epl.ledger as ledger
from epl.bot import fires, notify, render, serve
from epl.bot.settings import Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m epl.bot", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("serve", help="poll Telegram for commands until stopped")

    pushing = sub.add_parser("notify", help="say what the fire that just ran did")
    pushing.add_argument(
        "subcommand",
        nargs="?",
        default="seal",
        help="which of the loop's commands just ran (default: seal)",
    )

    sub.add_parser("check", help="print the health answer without sending anything")

    previewing = sub.add_parser("answer", help="print one command's answer without sending it")
    previewing.add_argument(
        "text",
        nargs="+",
        help="the command as a Telegram user would type it, e.g. /week or /club arsenal",
    )

    args = parser.parse_args(argv)

    if args.command == "serve":
        return serve.main()
    if args.command == "notify":
        return notify.main([args.subcommand])
    if args.command == "check":
        return _check()
    if args.command == "answer":
        return _answer(" ".join(args.text))
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


def _answer(text: str) -> int:
    """Print what one command would reply, with the markup stripped for a terminal.

    The leading slash is optional, and that is not mere convenience. This project is developed on
    Windows and Git Bash rewrites a bare ``/next`` into ``C:/Program Files/Git/next`` before Python
    sees it — a shell doing something reasonable to something that is not a path. Accepting the name
    without the slash sidesteps it, and accepting it *with* the slash keeps the preview spelt the
    way a Telegram user would type it.
    """
    ledger.register_all()
    wanted = text.strip()
    command = wanted if wanted.startswith("/") else f"/{wanted.rsplit('/', 1)[-1]}"
    reply = serve.dispatch(command, now=fires.uk_now())
    if reply is None:  # pragma: no cover - a leading slash is guaranteed above
        print(f"not a command: {text!r}", file=sys.stderr)
        return 1
    print(render.strip_tags(reply))
    return 0


def _check() -> int:
    """Print the health answer and whether the settings are there. Sends nothing, ever.

    Two questions in one, because they fail in ways that look alike from a Pi: a bot that is not
    configured and a schedule that is not firing both produce silence.

    The answer comes through :func:`epl.bot.serve.dispatch` rather than by calling
    :func:`epl.bot.answers.health` directly, so what this prints is literally what `/health` would
    reply. A check that composed its own answer could pass while the command it stands in for was
    broken, which is the one thing a check must not do.
    """
    ledger.register_all()
    print(serve.dispatch("/health", now=fires.uk_now()))
    print()
    try:
        settings = Settings.from_environment()
    except Exception as unset:
        print(f"Not configured: {unset}")
        return 1
    print(f"Configured: {settings.redacted()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
