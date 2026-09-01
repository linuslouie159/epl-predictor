# Running the live loop unattended

Issue #19. The loop itself is built and documented in `src/epl/live/`; this directory is only the
part that makes it run without anybody there — an image, a compose file, a wrapper cron calls, and
the crontab itself.

**Read this first: this schedule seals real Prediction Rounds.** It was written when it could not —
`fixtures.csv` had never been seen carrying a Premier League row across three fetches — and that
changed on 28 August 2026, when the fourth fetch found a full round and the loop sealed it. So a
fire that reports "no row in a tier this project predicts" and exits 0 is still designed behaviour
(issue #19, criterion 4), but it is **no longer the only outcome and no longer the expected one**.
Two silences now look identical from the log and both cost a round: the Pi being off through a
window (open risk 6), and every fetch inside a window landing on a copy upstream has not regenerated
(open risk 7). See docs/DECISIONS.md, "Bringing it up on the Pi, and the first Sealed Prediction
Round".

## What runs where

| Piece | Where it lives |
|---|---|
| The conda environment (ADR 0009) | inside the image, built from `environment.yml` |
| The code, the corpus, `outputs/live/` | a normal `git clone` on the Pi, bind-mounted at `/repo` |
| The schedule | the Pi's own crontab (`deploy/crontab`) |
| The logs | `deploy/logs/live_loop.log` on the Pi, gitignored |
| The Telegram bot | the same image, service `bot`, `up -d` and restarting (issue #20) |
| The bot's token and allowlist | `deploy/.env` on the Pi, gitignored |

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
| 0 | The rolling file held no Premier League Fixture — a stale upstream copy, or a genuinely empty week | No, but see open risk 7 |
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

## The Telegram bot (issue #20)

Two processes out of one image, and they are opposite shapes on purpose.

```bash
# the long-lived half: polls Telegram, answers /round /live /board /health
docker compose -f deploy/docker-compose.yml up -d bot

# the short-lived half: deploy/run_live.sh calls this itself after every fire
docker compose -f deploy/docker-compose.yml run --rm notify notify seal

# neither of the above, and no token needed: what would the bot say right now?
docker compose -f deploy/docker-compose.yml run --rm notify check
```

Put `EPL_TELEGRAM_TOKEN` and `EPL_TELEGRAM_ALLOWED_IDS` in `deploy/.env` (see `.env.example`). With
neither set, nothing changes: `notify` prints "not sending" and exits 0, and `serve` refuses to
start rather than running an open bot.

Four things worth knowing before touching it:

- **It cannot write to either Prediction store**, and that is checked structurally rather than
  promised — `tests/bot/test_the_bot_is_read_only.py` walks every import and every call in
  `src/epl/bot/`. A chat app must not be a second door into `outputs/live/` (ADR 0005).
- **It never sees the deploy key.** The ssh mount is on the `live` service alone. The bot is the
  only long-lived process here and the only one taking input from outside the machine.
- **It restarts and the loop's containers do not.** A bot that is not running hears nothing, so it
  carries `restart: unless-stopped`; a *loop* container that restarted after a crash would look
  exactly like one with nothing to seal, which is most weeks.
- **Only one may poll a token.** A second `up -d bot` fails on the lock file; a second poller on
  another machine is caught by Telegram's 409 and stops. Both matter, because two pollers do not
  conflict visibly — they split the updates, and the half that arrives looks like it is working.

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
- **Do not give the bot a command that writes.** Not `/seal`, not `/score`, not "just a `/refresh`".
  Everything the loop does is timed against a moment nobody chose, and a message from a phone is the
  shortest route in the whole system to a Sealed Prediction that was not.
- **Do not let a notify failure fail the run.** `deploy/run_live.sh` exits with the *loop's* code
  and `python -m epl.bot notify` exits 0 whatever happens. Monitoring that can take down the thing
  it monitors is worse than none.
- **Do not point `prematch` at `outputs/live/`.** It writes Pre-Match Readings to
  `outputs/prematch/`, a separate committed store that is never scored. A Reading is stamped an hour
  before kickoff, *after* its round was sealed, so it has seen results the sealed forecast had not;
  in the sealed store a later instant means a superseding bug fix and the scoreboard keeps the
  latest one, so admitting Readings there would improve the live track record every week for a
  reason nobody could see. See `outputs/prematch/README.md`.
- **`prematch` is optional and the rest of the schedule does not depend on it.** Drop its crontab
  line and the loop still seals, scores and pushes and the bot still answers every command; what is
  lost is the message an hour before each kickoff.
