"""How the bot is wired into the Pi, checked in the files rather than on the Pi — issue #20.

Three of its acceptance criteria are satisfied by configuration rather than by code, and each of
them fails silently if the configuration is wrong.

**The token lives outside the repository.** `epl.bot.settings` reads the environment and nothing
else, so the only way a token gets committed is `deploy/.env` ceasing to be ignored. That is one
line of `.gitignore` away at all times.

**The bot survives a reboot.** `restart: unless-stopped` on a compose service is the whole of that
promise, and a service without it looks identical until the Pi is power-cycled — which is exactly
the event open risk 6 is about, so the bot would be absent at the only moment it was needed.

**A notify failure never breaks the run that triggered it.** Guarded twice: `epl.bot.notify.main`
returns 0 whatever happens, and `deploy/run_live.sh` exits with the *loop's* code. Both are checked,
because the second is a shell script somebody will one day edit without reading the Python.

The same reasoning as `test_the_schedule_is_runnable.py`: the mistakes deployment makes are not
caught by unit tests and are found on the Pi, late, by their absence.
"""

from __future__ import annotations

import re
import subprocess

import pytest

from epl.paths import project_root

COMPOSE = "deploy/docker-compose.yml"
RUN_LIVE = "deploy/run_live.sh"


def _read(relative: str) -> str:
    return (project_root() / relative).read_text(encoding="utf-8")


class TestTheTokenIsNotInTheRepository:
    def test_the_env_file_is_ignored(self) -> None:
        """`deploy/.env` is where SETUP.md tells the operator to put the token."""
        result = subprocess.run(
            ["git", "check-ignore", "deploy/.env"],
            cwd=project_root(),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            "deploy/.env is not gitignored, so the place the operator is told to put the bot "
            "token is a place git would commit it"
        )

    def test_the_example_is_not_ignored_and_holds_no_token(self) -> None:
        """The example is committed on purpose; it must therefore hold only variable names."""
        example = _read("deploy/.env.example")

        assert "EPL_TELEGRAM_TOKEN" in example
        assert re.search(r"\b\d{8,}:[A-Za-z0-9_-]{30,}", example) is None

    def test_no_tracked_file_holds_something_shaped_like_a_bot_token(self) -> None:
        """A Bot API token has a recognisable shape — digits, a colon, 35 URL-safe characters.

        Cheap to check and worth checking: a leaked token is one message away from somebody else
        reading this machine's log, and a token pasted into a docstring "just to test it" is the
        realistic way that happens.
        """
        tracked = subprocess.run(
            ["git", "grep", "-nIE", r"[0-9]{8,10}:[A-Za-z0-9_-]{35}", "--", ".", ":!*.csv"],
            cwd=project_root(),
            capture_output=True,
            text=True,
            check=False,
        )
        assert tracked.stdout == "", tracked.stdout


class TestTheBotIsAServiceThatComesBack:
    def test_it_restarts_unless_stopped(self) -> None:
        compose = _read(COMPOSE)
        bot = compose[compose.index("\n  bot:"):]

        assert "restart: unless-stopped" in bot

    def test_the_loop_s_own_services_do_not_restart(self) -> None:
        """The other half of the same decision. A long-running container that had crashed would
        look exactly like one with nothing to seal, and the loop has nothing to seal most weeks."""
        compose = _read(COMPOSE)
        live = compose[compose.index("\n  live:"):compose.index("\n  notify:")]

        assert "restart:" not in live

    def test_the_bot_never_mounts_the_deploy_key(self) -> None:
        """It cannot push and has no business pushing, so it does not hold the key that would.

        Least privilege, and specifically: the bot is the only long-lived process here and the only
        one that takes input from outside the machine.
        """
        compose = _read(COMPOSE)
        after_live = compose[compose.index("\n  notify:"):]

        assert ".ssh" not in after_live

    def test_its_output_is_not_buffered_away(self) -> None:
        """Found on the first real start, on the Pi (issue #20).

        Python block-buffers stdout when it is not a terminal, and in a container it never is. A
        one-shot service flushes when it exits; a long-lived one does not reach the flush, so the
        bot ran, polled, and logged nothing at all. That is worse than a crash — a crashed container
        is visibly not running, and this one was `Up 43 seconds` and mute.

        It also silently removes the diagnostic SETUP.md sends an operator to: when the bot answers
        nobody, the console is supposed to say `refused user id=N`.
        """
        assert 'PYTHONUNBUFFERED: "1"' in _read(COMPOSE)

    def test_both_halves_of_the_bot_run_the_bot_and_not_the_loop(self) -> None:
        """A service that inherited `python -m epl.live` would seal on a `docker compose up`."""
        compose = _read(COMPOSE)

        assert compose.count('entrypoint: ["python", "-m", "epl.bot"]') == 2


class TestTheNotifierCannotBreakTheRun:
    """The notifier is a shell function with two callers now, and these read the call sites.

    They used to read source order against the single inline `docker compose ... run --rm notify`
    line: exit code captured above it, `===== END` written above it. That was a fair proxy while
    there was one caller, and it stopped being one when the lock-timeout path needed to notify too
    (`deploy/run_live.sh`, and the collision that made it necessary is documented in
    `test_the_schedule_does_not_collide.py`). Source order and execution order are the same thing
    only until somebody writes a function, so these now say what they always meant.
    """

    #: Every place the script actually calls the notifier — the ordinary end of a run, and a loop
    #: fire that could not take the lock. Not the definition, which sits above both by necessity.
    CALL = "\nnotify_about_the_fire\n", "\n    notify_about_the_fire\n"

    def test_the_schedule_exits_with_the_loop_s_code_and_not_the_notifier_s(self) -> None:
        script = _read(RUN_LIVE)

        assert script.rstrip().endswith('exit "$run_exit"')
        # Captured from the loop's own container, on the line after it, so nothing run later —
        # the notifier included — can be what cron learns.
        assert "run --rm live " in script.split("run_exit=$?")[0].rsplit("\n", 2)[-2]

    def test_the_notifier_call_cannot_fail_the_script(self) -> None:
        """One `|| true`, inside the function, so it covers every caller rather than the first."""
        script = _read(RUN_LIVE)
        body = script[script.index("notify_about_the_fire() {"):]

        assert "|| true" in body.split("\n}")[0]

    @pytest.mark.parametrize("call", CALL)
    def test_it_is_called_after_the_end_line_the_bot_reads(self, call: str) -> None:
        """The notifier reads the exit code off the `===== END` line, so that line has to be
        written before it runs — at *every* call site. Calling it earlier would report the fire as
        unfinished, and the lock-timeout path exists precisely to be reported."""
        script = _read(RUN_LIVE)
        before = script[: script.index(call)]

        assert "===== END  %s" in before, (
            f"the notifier is called at {call.strip()!r} without an END line written first"
        )

    @pytest.mark.parametrize("subcommand", ["seal", "score"])
    def test_the_notifier_is_told_which_command_ran(self, subcommand: str) -> None:
        """`seal` and `score` produce different artefacts and different messages, and the bot picks
        the last fire *of that subcommand* — so a notifier told nothing would report the wrong one
        on a day both ran."""
        assert 'notify "$subcommand"' in _read(RUN_LIVE)
        assert subcommand in _read("deploy/crontab")
