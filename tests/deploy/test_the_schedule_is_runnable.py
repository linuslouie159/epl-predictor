"""The schedule's own executability, checked in the index rather than on disk.

Issue #21. `deploy/run_live.sh` is what all three crontab lines invoke, by absolute path and with
no interpreter in front of it. A clone that lands it without the executable bit does not fail
loudly at install time — cron fires, the shell answers **exit 126**, and the log records a
`===== RUN =====` block wrapped around one line of `Permission denied`. That is a schedule which
looks installed and seals nothing, which is the failure mode this project can least afford to have
twice a week.

**Why the git index and not the filesystem.** This repository is developed on Windows, where
`core.filemode` is `false` and every checkout reports `rwxr-xr-x` whatever the index holds. So a
test that asked the working tree would pass on the machine where the mistake is made and fail only
on the Pi, which is the wrong way round. `git ls-files -s` reports the mode actually recorded in
the tree, and that is the thing a fresh clone gets.

Found by running `deploy/run_live.sh upcoming` on the Pi for the first time, at issue #21. It had
been committed `100644` since #19 and no test looked.
"""

from __future__ import annotations

import subprocess

import pytest

from epl.paths import project_root

#: Mode git records for a file it will check out executable.
EXECUTABLE = "100755"

#: Everything cron invokes directly. `deploy/crontab` names `run_live.sh` and nothing else today;
#: this is a tuple so that a second entry point is a one-line change rather than a new test.
INVOKED_BY_CRON = ("deploy/run_live.sh",)


def _recorded_mode(path: str) -> str:
    """The mode `path` carries in the git index, as a fresh clone would receive it."""
    result = subprocess.run(
        ["git", "ls-files", "-s", "--", path],
        cwd=project_root(),
        capture_output=True,
        text=True,
        check=True,
    )
    if not result.stdout.strip():
        raise AssertionError(f"{path} is not tracked by git, so no clone would receive it at all")
    return result.stdout.split()[0]


@pytest.mark.parametrize("path", INVOKED_BY_CRON)
def test_what_cron_invokes_is_committed_executable(path: str) -> None:
    assert _recorded_mode(path) == EXECUTABLE, (
        f"{path} is committed non-executable, so a fresh clone cannot run it and every "
        f"scheduled fire would exit 126. Fix with: git update-index --chmod=+x {path}"
    )
