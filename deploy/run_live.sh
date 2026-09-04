#!/usr/bin/env bash
# What cron actually calls (issue #19). One command of the live loop, logged, in a container.
#
#   deploy/run_live.sh seal --push
#   deploy/run_live.sh score
#   deploy/run_live.sh upcoming
#
# This is a wrapper and holds one policy, named below. Every decision that could seal a round at
# the wrong moment lives in the code: which moment it is (epl.ledger.live.uk_now), whether a round
# is inside its window (epl.ledger.live.window), and what the exit code means
# (epl.live.__main__._seal). A shell script that decided any of those would be a second place to
# check, and the sealed store's whole value is that there is one.
#
# THE ONE POLICY IS WHO WINS THE LOCK, and it is here because it can be nowhere else. The lock
# exists to stop a second container starting at all, so the process that loses it never reaches
# Python — a priority expressed in `epl.live` would be a priority expressed inside the thing that
# was not allowed to run. See the flock section below for what it decides and what it cost.
#
# The log format is the other project's on this Pi, on purpose. The paper-trading loop already
# writes `===== RUN ... =====` / `===== END ... (exit N) =====` blocks into live/logs/ and already
# has a reader for them. Two loops on one machine writing two log conventions is how a box becomes
# unreadable, and matching costs nothing here.

set -u

subcommand="${1:-upcoming}"
shift || true

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$script_dir")"
log_dir="$script_dir/logs"
lock="$log_dir/live_loop.lock"

# WHY prematch HAS ITS OWN LOG. `live_loop.log` is a file a person opens when something has gone
# wrong, and it holds three fires a week. `prematch` fires around forty times on a matchday and
# almost all of them write one line saying nothing kicks off within the hour. Mixing them would
# bury the blocks that matter under a thousand that do not. It is the same block format and the
# same parser either way (`epl.bot.fires.LOGS` names this file), so `/health` still sees them.
#
# One of two things in this script that vary by subcommand, and this one is a filename rather than
# a decision. The other is the lock below, which is a decision and says so. Every choice that could
# seal or read something at the wrong *moment* still lives in the code.
if [ "$subcommand" = "prematch" ]; then
    log="$log_dir/prematch.log"
else
    log="$log_dir/live_loop.log"
fi

mkdir -p "$log_dir"
cd "$root" || exit 1

# The Telegram notifier (issue #20), and three things about where it is called from.
#
# AFTER an END line, always, because it reads this log to find out what just happened and the exit
# code is on that line. Its own output lands outside any `===== RUN` block and is ignored on the
# next read, which is what the parser drops the stand-down and flock lines for.
#
# NOT allowed to change what cron learns. Every caller captures its own exit code first; the
# `|| true` is here so that a notifier which cannot start — no token yet, no network, an image
# without the bot in it — leaves the loop's own exit code alone. Monitoring that can take down the
# thing it monitors is worse than none, which is also why `python -m epl.bot notify` exits 0
# whatever happens inside it. Two guards, because this one is a shell script that somebody will one
# day edit without reading the Python.
#
# A FUNCTION because there are two callers now: the ordinary end of a run, and a `seal` or `score`
# that could not take the lock. The second is the whole reason the first stopped being enough — it
# used to `exit 0` several lines above this, so the one fire nobody could afford to miss was also
# the one fire the notifier was never told about.
notify_about_the_fire() {
    docker compose -f "$script_dir/docker-compose.yml" run --rm notify notify "$subcommand" \
        >> "$log" 2>&1 || true
}

# A missing `flock` is checked for separately, and the reason is worth stating: `flock -n` exits 1
# when the lock is held, and a shell exits 127 when the command is not there — so `if ! flock`
# treats "not installed" as "someone else is running", and the loop then stands down quietly and
# for ever. That is the one failure this schedule must not be able to have: silent, and shaped
# exactly like success. Caught while testing this script under a shell that has no flock.
if ! command -v flock > /dev/null 2>&1; then
    printf '%s  flock is not installed; refusing to run without a lock\n' \
        "$(date '+%Y-%m-%d %H:%M:%S %z')" | tee -a "$log" >&2
    exit 1
fi

# THE LOCK, AND WHICH SIDE OF IT YIELDS.
#
# Two containers committing into one bind-mounted checkout at once is a corrupt index rather than a
# race anyone would enjoy diagnosing, so all four crontab lines take one lock. `prematch --push`
# commits too, which is why it shares this one rather than getting its own.
#
# What `flock -n` alone does NOT express is that the four fires are not equally important, and the
# gap cost a seal. Stage 18 put `prematch` on the hour and the half hour, which is exactly when the
# two `seal --push` fires land; `flock -n` has no priority, so cron started both in the same second
# and whichever won took it. Measured on the Pi, 2026-09-01 11:00:01: the seal stood down for a
# `prematch` fire that had nothing to do and was finished three seconds later. The round survived
# only because the 18:30 retry happened to win its own race, and a round that loses both is lost for
# good — `supersede` refuses a round at or after its first kickoff (ADR 0005).
#
# So the priority is stated here rather than left to whichever process cron happened to start:
#
#   prematch  yields at once (-n). It costs at most one of the two chances a Fixture gets at a
#             card, the run already in flight is doing the more important work, and the sealed
#             round is untouched. This is the case deploy/crontab has always described.
#   the loop  waits (-w). A `prematch` fire holds this for about four seconds with nothing due and
#             under a minute with a Fixture due, against a sealing window that runs from 16:00 UK
#             to a first kickoff no earlier than 19:45. Ten minutes is generous either way.
#
# A wait that expires is NOT a stand-down. A seal that could not run is the failure this schedule
# can least afford, so it is written as a real block below and exits 1, rather than adding a line
# nothing reads — `epl.bot.fires.parse` drops lines outside a `===== RUN` block on purpose, which
# is why the 1 Sep stand-down reached neither cron nor the bot nor the log's own reader.
LOCK_WAIT=600

if [ "$subcommand" = "prematch" ]; then
    lock_wait=(-n)
else
    lock_wait=(-w "$LOCK_WAIT")
fi

exec 9>"$lock"
if ! flock "${lock_wait[@]}" 9; then
    stamp="$(date '+%Y-%m-%d %H:%M:%S %z')"
    if [ "$subcommand" = "prematch" ]; then
        printf '%s  live loop already running (%s); standing down\n' "$stamp" "$subcommand" \
            >> "$log"
        exit 0
    fi
    # The loop's own format, so this is a fire with a non-zero exit rather than a new thing for
    # somebody to teach the parser about: `epl.bot.fires.Fire.failed` reports it, `epl.bot.notify`
    # announces it, and cron mails the exit code.
    printf '\n===== RUN %s  (%s %s) =====\n' "$stamp" "$subcommand" "$*" >> "$log"
    printf 'could not take the lock in %ss; %s did not run\n' "$LOCK_WAIT" "$subcommand" >> "$log"
    printf '===== END  %s  (exit 1) =====\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" >> "$log"
    notify_about_the_fire
    exit 1
fi

stamp="$(date '+%Y-%m-%d %H:%M:%S %z')"
printf '\n===== RUN %s  (%s %s) =====\n' "$stamp" "$subcommand" "$*" >> "$log"

# `run --rm` rather than `up`: the container exists for the length of one command, and its exit
# code is the whole interface. Not wrapped in `set -e` — a non-zero exit is the thing being
# recorded, not a reason to abort before recording it.
docker compose -f "$script_dir/docker-compose.yml" run --rm live "$subcommand" "$@" >> "$log" 2>&1
run_exit=$?

stamp_end="$(date '+%Y-%m-%d %H:%M:%S %z')"
printf '===== END  %s  (exit %d) =====\n' "$stamp_end" "$run_exit" >> "$log"

# SILENT most of the time, and that is correct. Two of the three loop crontab lines are
# `seal --push` and the second is a retry designed to find the round already sealed; a bot that
# announced every fire is a bot nobody reads by November. See the function for why it is one.
notify_about_the_fire

# Cron mails a job that exits non-zero and says nothing about one that does not, which is exactly
# the split the loop's own exit codes were written to (issue #19, criteria 4 and 5): a week with
# nothing to seal is silent, and a round that was sealed but could not be pushed is not.
exit "$run_exit"
