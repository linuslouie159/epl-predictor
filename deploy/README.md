# Running the live loop unattended

Issue #19. The loop itself is built and documented in `src/epl/live/`; this directory is only the
part that makes it run without anybody there — an image, a compose file, a wrapper cron calls, and
the crontab itself.

**Read this first: there is nothing to seal yet.** `fixtures.csv` has never been seen carrying a
Premier League row — three fetches, 21 and 27 August 2026, the last of them the day before a round
— so every fire of this schedule will currently report "no row in a tier this project predicts"
and exit 0. That is the designed behaviour and not a broken install (issue #19, criterion 4). The
schedule is installed *now* so that the day upstream starts listing Premier League Fixtures, the
loop is already watching. See docs/DECISIONS.md, "The live loop, and the input it is still waiting
for", and open risk 2.

## What runs where

| Piece | Where it lives |
|---|---|
| The conda environment (ADR 0009) | inside the image, built from `environment.yml` |
| The code, the corpus, `outputs/live/` | a normal `git clone` on the Pi, bind-mounted at `/repo` |
| The schedule | the Pi's own crontab (`deploy/crontab`) |
| The logs | `deploy/logs/live_loop.log` on the Pi, gitignored |

The image is the *environment*, not the project. `git pull` on the Pi updates the code with no
rebuild, and the sealed rounds land in a checkout you can inspect and push by hand.

## Sharing the Pi with the paper-trading project

This box also runs `strat testing vector bt`, and the two are deliberately deployed differently.

That project installs as a venv of prebuilt aarch64 wheels — `requirements-live.txt` says so in as
many words — precisely so that nothing on the Pi needs a toolchain. This project cannot be
installed that way. ADR 0009 chose conda-forge for the prebuilt scientific stack, and
`environment.yml` pins the BLAS provider because the wrong one aborts the interpreter on every
LAPACK call. Two package managers arguing over one machine's system libraries is what the image
prevents, and it is the only reason there is one.

So: **that project stays on bare metal, with its venv and its systemd unit; this one stays in a
container.** Neither installs anything the other can see. What they share is the Pi's crontab and
a log convention, and both are on purpose:

- `deploy/run_live.sh` writes the same `===== RUN … =====` / `===== END … (exit N) =====` blocks
  that project's `live/run_daily.sh` writes, so one habit reads both logs.
- The cron slots do not overlap. This loop fires Tuesday and Friday at 06:00, 16:00 and 18:30 UK;
  the trading loop fires on weekday mornings before the US open. If that changes, note that this
  one holds a lock (`deploy/logs/live_loop.lock`) against *itself* only — the two projects touch
  no common file.

There is no Telegram wiring here. That is a separate piece of work with its own bot, and this
directory publishes nothing shaped for it.

## Setting it up

**[SETUP.md](./SETUP.md)** — the runbook, from a Pi with nothing on it to an installed
schedule. Docker, a deploy key, the image, the raw cache, a smoke test, and the crontab.

It is a separate file rather than a section here because the two are read at different
moments and by different people. That one is followed once, at the Pi, with a terminal open.
This one is read afterwards, when something has gone wrong or when somebody is deciding
whether to change how any of it works. Keeping the steps in one place also means there is
only one copy of them to keep true.

## What each exit code means

The exit code is the entire interface to cron, so it is worth knowing exactly what it claims.

| Exit | What happened | Does anybody need to look? |
|---|---|---|
| 0 | The round was sealed, committed and pushed | No |
| 0 | The round was already sealed; nothing written, nothing committed, pushed again | No |
| 0 | No Prediction Round is inside its window — most of the week | No |
| 0 | The rolling file held no Premier League Fixture — every fire so far | No, and see open risk 2 |
| 1 | Sealed but **not committed**, or committed and **not pushed** | **Yes** — it is not evidence yet |
| 1 | `LIVE_SEASON` has gone stale, in either direction | **Yes** |
| non-zero, with a traceback | Upstream changed the rolling file's shape, or the network failed | **Yes**, unless the 18:30 retry clears it |

That last row is the one place the schedule can cry wolf: a transient fetch failure at 16:00 mails
you a traceback for something the 18:30 fire will fix by itself. It is left loud on purpose. A
failed fetch and a *sustained* outage are the same event until the retry has run, and the version
of this that stayed quiet on the first fire would also stay quiet on the week upstream went down
for good. One avoidable email per outage is the cheaper side of that trade.

The two silences at the top of the "no" column are told apart by
`epl.live.upcoming.NothingToSeal`, not by reading messages. That class exists so that a schedule
firing twice a week for a season does not teach its owner to ignore a red job — which is the state
in which the row that matters goes unread.

### The failure you will actually hit: a diverged branch

This repository is worked on from a desktop and pushed from a Pi, so sooner or later the Pi's `main`
is behind `origin/main` and the push is rejected as a non-fast-forward. The loop reports `NOT
PUSHED` and exits 1, which is correct — the round is sealed and committed on the Pi and is not
evidence anywhere else yet.

**Do not fix it by force-pushing, and do not fix it by rebasing the sealed commit.** `outputs/live/`
is append-only, and a rewritten commit is a sealed round whose recorded time is now a different
time — which is precisely the claim the store exists to keep true (ADR 0005). Merge instead:

```bash
cd ~/epl-predictor
git pull --no-rebase       # a merge commit is fine; a rewritten history is not
git push
```

The next scheduled fire then finds the round already sealed and pushes it as its ordinary business.
If the divergence is in `outputs/live/` itself — two machines sealed the same round — stop and read
`python -m epl.ledger audit` before doing anything, because that is a real violation and not a git
inconvenience.

**The loop deliberately does not `git pull` itself, so this stays red until you merge.** Self-healing
was available and was declined, for two reasons. A pull changes the code the loop is about to run,
unreviewed, at 16:00 on a Friday — so the round would be sealed by whatever was last pushed rather
than by what was last tested. And an unattended merge that hits a conflict leaves a checkout with
conflict markers in it and `outputs/live/` inside that checkout; the one directory in this
repository that is evidence is the last one that should be repaired by a script running alone at
night. A red job asking a person to merge is the cheaper failure.

## Things not to do

- **Do not add a `--now` flag** so a missed round can be sealed late. `epl.live.__main__.clock` is
  a function and not an option on purpose, and `tests/live/test_unattended.py` fails if that
  changes. A round whose window has shut is gone; that is what makes the ones in the store worth
  anything (ADR 0005).
- **Do not schedule `python -m epl.ledger backfill`.** It would regenerate Predictions this loop
  had sealed, which is ADR 0005's exact failure, and it would move the Evaluation Window's numbers
  on a timer.
- **Do not rewrite a file under `outputs/live/`.** A correction is a new revision at a new As-Of
  Instant: `seal --supersede`, and only while the round is still open.
- **Do not set the container's `TZ` to fix a scheduling problem.** Nothing depends on it any more —
  `epl.ledger.live.uk_now` converts explicitly — so changing it will look like it helped and will
  not have.
