"""Command line entry point for the Telegram bot.

    python -m epl.bot serve            poll for commands until stopped — the long-lived half
    python -m epl.bot notify seal      say what the fire that just ran did, if anything
    python -m epl.bot check            print what the bot would say about the schedule's health

``serve`` is what `deploy/docker-compose.yml`'s `bot` service runs, and it is the only long-lived
process this project has. ``notify`` is what `deploy/run_live.sh` calls after every scheduled fire
and **always exits 0**, because a notify failure must never break the run that triggered it.

``check`` sends nothing. It is the answer to "is this thing wired up?" on a machine where sending
a real message to prove it would be the worst possible way to find out — and it needs neither a
token nor a network, so it is also what to run before either exists.
"""

from __future__ import annotations

import argparse
import sys

import epl.ledger as ledger
from epl.bot import fires, notify, serve
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

    args = parser.parse_args(argv)

    if args.command == "serve":
        return serve.main()
    if args.command == "notify":
        return notify.main([args.subcommand])
    if args.command == "check":
        return _check()
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


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
