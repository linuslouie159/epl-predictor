"""The shape `deploy/run_live.sh` writes, built here so several tests can be written against it.

Not a test module and not a fixture: it is a tiny renderer of somebody else's format, and every test
that reads a log needs one. Keeping it beside the tests rather than in `src/epl/bot` is deliberate —
nothing in the package writes this format, it only reads it, and a writer in the package would be a
second definition of a shape whose only real definition is a shell script.
"""

from __future__ import annotations

#: One real block, copied from docs/DECISIONS.md, "The schedule fired, unattended, and pushed".
#: The retry slot, in the Pi's converted zone: `13:30 -0400` is 18:30 in London.
THE_FIRE_THAT_PROVED_THE_SCHEDULE = """
===== RUN 2026-08-28 13:30:01 -0400  (seal --push) =====
rolling file: fixtures_20260828T173004Z.csv
2026-08-28: 10 Fixtures, as-of 2026-08-28T00:00:00, first kickoff 2026-08-28T20:00:00
2026-08-28 is already sealed — nothing to do
  a genuine correction goes in with --supersede (ADR 0005)
pushed to origin/main
===== END  2026-08-28 13:30:08 -0400  (exit 0) =====
"""


def block(
    stamp: str,
    command: str,
    *body: str,
    exit_code: int | None = 0,
    ended: str | None = None,
) -> str:
    """One `===== RUN =====` block in the exact shape `deploy/run_live.sh` prints it.

    The leading newline, the two spaces after `END` and the trailing space a subcommand with no
    arguments leaves behind are all the shell's, and all reproduced — they are the parts of the
    format most likely to be got wrong by a parser written from a description of it.
    """
    lines = [f"\n===== RUN {stamp}  ({command}) =====", *body]
    if exit_code is not None:
        lines.append(f"===== END  {ended or stamp}  (exit {exit_code}) =====")
    return "\n".join(lines) + "\n"
