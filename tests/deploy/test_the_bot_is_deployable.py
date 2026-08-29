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

    def test_both_halves_of_the_bot_run_the_bot_and_not_the_loop(self) -> None:
        """A service that inherited `python -m epl.live` would seal on a `docker compose up`."""
        compose = _read(COMPOSE)

        assert compose.count('entrypoint: ["python", "-m", "epl.bot"]') == 2


class TestTheNotifierCannotBreakTheRun:
    def test_the_schedule_exits_with_the_loop_s_code_and_not_the_notifier_s(self) -> None:
        script = _read(RUN_LIVE)

        assert script.rstrip().endswith('exit "$run_exit"')
        assert script.index("run_exit=$?") < script.index("run --rm notify")

    def test_the_notifier_call_cannot_fail_the_script(self) -> None:
        script = _read(RUN_LIVE)
        call = script[script.index("run --rm notify"):]

        assert "|| true" in call.split("\n\n")[0]

    def test_it_is_called_after_the_end_line_the_bot_reads(self) -> None:
        """The notifier reads the exit code off the `===== END` line, so that line has to exist
        before it runs. Calling it earlier would report every fire as unfinished."""
        script = _read(RUN_LIVE)

        assert script.index("===== END") < script.index("run --rm notify")

    @pytest.mark.parametrize("subcommand", ["seal", "score"])
    def test_the_notifier_is_told_which_command_ran(self, subcommand: str) -> None:
        """`seal` and `score` produce different artefacts and different messages, and the bot picks
        the last fire *of that subcommand* — so a notifier told nothing would report the wrong one
        on a day both ran."""
        assert 'notify "$subcommand"' in _read(RUN_LIVE)
        assert subcommand in _read("deploy/crontab")
