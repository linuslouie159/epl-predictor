# Setting up the live loop on a Raspberry Pi

A runbook, start to finish, from a Pi with nothing on it to an installed schedule. Follow it in
order — several steps exist only to make a later one work, and they are marked where that is true.

`deploy/README.md` is the companion to this file and answers *why*: why a container at all, why
these times, what each exit code means, and how this coexists with the paper-trading project on the
same machine. This file answers *how*, and tries not to repeat it.

**Two things to know before you start.**

**Nothing here has ever been run.** The image was written on a machine with no running Docker, so
every command below is checked statically — `docker compose config` parses, the `sed` in the
Dockerfile is guarded by a `grep` that fails the build loudly if its assumption breaks, and the cron
wrapper was tested end to end against a stubbed `docker`. The conda solve on aarch64 is the one step
with no evidence behind it. Expect step 3 to be where trouble is, if there is any.

**The loop will seal nothing, and that is correct.** `fixtures.csv` has never been observed carrying
a Premier League row — three fetches, 21 and twice on 27 Aug 2026 — so every fire will report "no
row in a tier this project predicts" and exit 0. You are installing a watcher, not starting a track
record. See `deploy/README.md` and docs/DECISIONS.md open risk 2.

Budget an hour, most of it waiting for step 3 and step 5.

---

## 0. Check the Pi can host this

```bash
uname -m                              # must be aarch64
cat /proc/device-tree/model           # expect Raspberry Pi 5
free -h                               # RAM; see the note below
df -h /                               # need ~8 GB free
command -v flock && echo "flock ok"   # the cron wrapper refuses to run without it
```

**`aarch64` is not optional.** A 32-bit Raspberry Pi OS reports `armv7l`, and conda-forge does not
ship this scientific stack for it. If that is what you have, stop — the fix is reinstalling 64-bit
Pi OS, not working around this.

**RAM.** A conda solve is memory-hungry and a 4 GB Pi can be tight. If step 3 dies with the build
process killed, that is the OOM killer rather than a broken Dockerfile; add swap and retry:

```bash
sudo dphys-swapfile swapoff
sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup && sudo dphys-swapfile swapon
```

**Docker.** If `docker --version` and `docker compose version` both answer, skip ahead.

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker "$USER"
```

Then **log out and back in** — group membership is read at login, and without it every `docker`
command needs `sudo`, which would leave the loop's files owned by root. Confirm with `docker run
--rm hello-world` as your own user, no `sudo`.

---

## 1. An SSH key GitHub will accept

Do this before cloning. The clone must be over SSH, because the loop pushes unattended and cannot be
handed a password.

```bash
ssh-keygen -t ed25519 -C "epl-loop@$(hostname)" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
```

Add that public key to the repository as a **deploy key with write access**: GitHub → the repo →
Settings → Deploy keys → Add deploy key → tick *Allow write access*. A deploy key scoped to this one
repository is the right choice over a personal key; it is the only thing the Pi needs to reach.

Then, and this step is load-bearing:

```bash
ssh -T git@github.com
```

**Do not skip it.** The container mounts `~/.ssh` **read-only**, so it cannot write `known_hosts`
and therefore cannot accept GitHub's host key itself. Running this by hand is what puts the entry
there. Skip it and the first push fails host-key verification, and the loop reports `NOT PUSHED` for
what reads like a credentials problem and is not.

You want the reply `Hi <you>/epl-predictor! You've successfully authenticated, but GitHub does not
provide shell access.` — "successfully authenticated" is the part that matters.

---

## 2. Clone, and give the clone an identity

```bash
git clone git@github.com:linuslouie159/epl-predictor.git ~/epl-predictor
cd ~/epl-predictor
git config user.name  "Your Name"
git config user.email "you@example.com"
```

Set on the clone, not globally, on purpose: a commit in the sealed store proves *when* a Prediction
was made and should say who made it — the Pi's owner, rather than a container image.

Check the remote is SSH and that a push works before anything depends on it:

```bash
git remote -v          # must say git@github.com:..., not https://
git push               # "Everything up-to-date", not a password prompt
```

---

## 3. Build the image

Before the ingest below, because the ingest runs inside it.

```bash
cd ~/epl-predictor
docker build -f deploy/Dockerfile -t epl-predictor:live .
```

This is the slow step — the conda solve dominates it, and on a Pi that is a long wait. If you have a
desktop with Docker, cross-building there and loading the result is usually the better trade:

```bash
# on the desktop, in a clone of this repo
docker buildx build --platform linux/arm64 -f deploy/Dockerfile -t epl-predictor:live --load .
docker save epl-predictor:live | ssh pi@raspberrypi 'docker load'
```

Verify:

```bash
docker run --rm epl-predictor:live --help      # the loop's own help text
```

**If it fails**, in rough order of likelihood: the OOM killer (see step 0); a conda-forge solve
conflict, which would be the first real evidence that `environment.yml` needs an aarch64 exception;
or the `grep` guard failing, which means the pip section of `environment.yml` moved and the
Dockerfile's comment explains what to do. It is a floating `:latest` base — if a rebuild ever
behaves differently from a previous one, pin it to a digest before suspecting anything else.

---

## 4. Tell compose who owns the clone

Before the ingest, or the cache it writes ends up owned by the wrong user and the loop cannot
refresh it later.

```bash
id -u && id -g
```

If those are both `1000` — the first account on Raspberry Pi OS — there is nothing to do. Otherwise:

```bash
cp deploy/.env.example deploy/.env
# edit PUID and PGID to match
```

---

## 5. Seed the raw cache

`data/raw/` and `data/processed/` are gitignored, so they do not arrive with the clone, and `score`
rebuilds the match table from the whole cache. 108 files, fetched once:

```bash
docker compose -f deploy/docker-compose.yml run --rm --entrypoint python live -m epl.ingest fetch
docker compose -f deploy/docker-compose.yml run --rm --entrypoint python live -m epl.ingest build
```

Check it landed:

```bash
find data/raw/football-data -name '*.csv' -not -path '*fixtures*' | wc -l   # 108
wc -l data/processed/matches.csv                                            # ~52,760
```

The match count is approximate on purpose and will only ever go up: 52,755 lines on 27 Aug 2026,
and the Live Season is ingested, so it grows every week the schedule runs `score`. What matters is
that it is in the tens of thousands rather than empty.

Please do not re-run `fetch` casually. It is polite to a free file host that asks nothing in return,
and it is the reason this runs on a Pi rather than on GitHub Actions at all.

---

## 6. Smoke-test, writing nothing

```bash
deploy/run_live.sh upcoming
echo "exit: $?"
tail -n 30 deploy/logs/live_loop.log
```

`upcoming` is the whole of `seal`'s work up to the point where `seal` would write. Exit 0 and a log
block like this is success:

```
===== RUN 2026-08-27 17:16:57 +0100  (upcoming ) =====
rolling file: fixtures_20260827T142131Z.csv
0 upcoming Premier League Fixtures in 2026/27
the rolling file held no Fixture in a tier this project predicts, so there is no Prediction Round to seal
===== END  2026-08-27 17:16:57 +0100  (exit 0) =====
```

**"0 upcoming Premier League Fixtures" is the expected answer**, not a failure. If it ever says
something else, that is the day this project has been waiting for — see `deploy/README.md`.

Sanity-check the clock while you are here, because a wrong one is silent:

```bash
docker compose -f deploy/docker-compose.yml run --rm --entrypoint python live \
  -c "from epl.ledger import live; print('loop clock (UK):', live.uk_now())"
date                                   # the Pi's own idea of now
```

The first line must be UK wall-clock time. It does not have to match the second, and on a Pi set to
another zone it should not.

---

## 7. Prove the push path before anything depends on it

Worth its own step. Nothing will seal for the foreseeable future, so `--push` will not exercise
itself, and you would otherwise discover a broken deploy key on the one day it finally mattered.

```bash
docker compose -f deploy/docker-compose.yml run --rm --entrypoint git live push --dry-run
```

That runs git *inside the container*, with the same mounted key, the same `HOME`, and the same uid
the schedule will use. `Everything up-to-date` means the whole chain works: key readable, host key
accepted, remote reachable, write access granted.

If it fails here, fix it now:

| Message | Cause |
|---|---|
| `Host key verification failed` | step 1's `ssh -T` was skipped; the read-only mount cannot fix this itself |
| `Permission denied (publickey)` | key not added to GitHub, or added without write access |
| `dubious ownership` | `GIT_CONFIG_*` in compose is not reaching git — check `deploy/.env` uid |
| `could not read Username` | the remote is HTTPS; re-point it at SSH (step 2) |

---

## 8. Install the schedule

```bash
cd ~/epl-predictor
sed "s|__ROOT__|$PWD|g" deploy/crontab > /tmp/epl.cron
crontab -l > /tmp/current 2>/dev/null; cat /tmp/epl.cron >> /tmp/current; crontab /tmp/current
crontab -l | tail -6
```

Confirm the paths in the output are absolute and real, and that `CRON_TZ=Europe/London` survived.

**If your cron does not honour `CRON_TZ`** — Debian's does; some do not — the consequence is mild
and worth knowing. The loop would fire at the wrong hour and exit 0 having sealed nothing, because
the window is judged in UK time by the code (`epl.ledger.live.uk_now`) whatever cron believes. A
misconfigured schedule here seals nothing; it cannot seal something wrongly. Check with:

```bash
grep -r CRON_TZ /usr/share/doc/cron/ 2>/dev/null | head -2
```

If it is unsupported, either set the whole Pi to `Europe/London` (`sudo raspi-config`, Localisation)
or convert the three times into the Pi's own zone by hand.

---

## 9. Watch the first real fire

The next scheduled slot is 06:00, 16:00 or 18:30 UK on a Tuesday or Friday. After it passes:

```bash
tail -n 40 ~/epl-predictor/deploy/logs/live_loop.log
```

You want a `===== RUN` block with `(exit 0)`. Cron mails you only when a job exits non-zero, which is
the whole design: a week with nothing to seal is silent, and a round that could not be pushed is not.

**A log with no new `===== RUN` block for a week is what an off Pi looks like from here**, and it is
the mitigation for open risk 6 — a round whose window passes while the Pi is off is lost and cannot
be sealed later.

---

## Backing it out

```bash
crontab -l | grep -v 'deploy/run_live.sh' | crontab -    # stop the schedule
docker image rm epl-predictor:live                        # reclaim the space
```

Leave the clone alone if anything was ever sealed. `outputs/live/` is append-only evidence, and
deleting a round that was committed is itself a violation the audit reports (ADR 0005).

---

## The one thing that will go wrong later

Once the Pi is pushing, your desktop clone and the Pi are two writers on `main`, and sooner or later
they diverge. The loop goes red with `NOT PUSHED` and stays red until a human merges — deliberately,
and it does not `git pull` itself.

**Merge, never rebase or force**, because a rewritten commit is a sealed round whose recorded time is
now a different time. `deploy/README.md`, "The failure you will actually hit", has the commands and
the reasoning, and is where to read it when it happens rather than now.
