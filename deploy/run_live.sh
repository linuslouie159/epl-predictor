#!/usr/bin/env bash
# What cron actually calls (issue #19). One command of the live loop, logged, in a container.
#
#   deploy/run_live.sh seal --push
#   deploy/run_live.sh score
#   deploy/run_live.sh upcoming
#
# This is a wrapper and holds no policy. Every decision that could seal a round at the wrong
# moment lives in the code: which moment it is (epl.ledger.live.uk_now), whether a round is inside
# its window (epl.ledger.live.window), and what the exit code means (epl.live.__main__._seal). A
# shell script that decided any of those would be a second place to check, and the sealed store's
# whole value is that there is one.
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
log="$log_dir/live_loop.log"
lock="$log_dir/live_loop.lock"

mkdir -p "$log_dir"

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

# The 16:00 and 18:30 fires are far enough apart never to overlap, and a slow first run plus a
# hand-run `score` are not. Two containers committing into one checkout at once is a corrupt index
# rather than a race anyone would enjoy diagnosing, so a second run stands down — and stands down
# quietly, because the run already in flight is doing the work.
exec 9>"$lock"
if ! flock -n 9; then
    printf '%s  live loop already running (%s); standing down\n' \
        "$(date '+%Y-%m-%d %H:%M:%S %z')" "$subcommand" >> "$log"
    exit 0
fi

cd "$root" || exit 1

stamp="$(date '+%Y-%m-%d %H:%M:%S %z')"
printf '\n===== RUN %s  (%s %s) =====\n' "$stamp" "$subcommand" "$*" >> "$log"

# `run --rm` rather than `up`: the container exists for the length of one command, and its exit
# code is the whole interface. Not wrapped in `set -e` — a non-zero exit is the thing being
# recorded, not a reason to abort before recording it.
docker compose -f "$script_dir/docker-compose.yml" run --rm live "$subcommand" "$@" >> "$log" 2>&1
run_exit=$?

stamp_end="$(date '+%Y-%m-%d %H:%M:%S %z')"
printf '===== END  %s  (exit %d) =====\n' "$stamp_end" "$run_exit" >> "$log"

# Cron mails a job that exits non-zero and says nothing about one that does not, which is exactly
# the split the loop's own exit codes were written to (issue #19, criteria 4 and 5): a week with
# nothing to seal is silent, and a round that was sealed but could not be pushed is not.
exit "$run_exit"
