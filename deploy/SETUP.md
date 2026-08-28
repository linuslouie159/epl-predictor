# Setting up the live loop on a Raspberry Pi

A runbook, start to finish, from a Pi with nothing on it to an installed schedule. Follow it in
order — several steps exist only to make a later one work, and they are marked where that is true.

`deploy/README.md` is the companion to this file and answers *why*: why a container at all, why
these times, what each exit code means, and how this coexists with the paper-trading project on the
same machine. This file answers *how*, and tries not to repeat it.

**Three things to know before you start.**

**This has now been run on a Pi, and the numbers below are that run.** On 28 Aug 2026 the whole of
this file was followed on a Raspberry Pi 5 (8 GB, NVMe, Raspberry Pi OS, system zone
`America/New_York`) at issue #21. Where a step was wrong it has been corrected here rather than
annotated. The one thing it does not cover is a 4 GB Pi or an SD card — see step 0.

**The aarch64 build is fast, and the old advice to cross-build was wrong.** This is the correction
most worth reading before you start, because it inverts what step 3 used to say:

| | amd64 desktop, 27 Aug | **Pi 5, 28 Aug** |
|---|---|---|
| `conda env create` | 1450 s | **73 s** |
| whole build | ~27 min | **271 s** |

`environment.yml` needed **no aarch64 exception** — conda-forge solved it in about 7 seconds for
`linux-aarch64` and every pin, OpenBLAS included, resolved as written (ADR 0009 holds on this
platform). The desktop figure was almost certainly Docker Desktop's VM filesystem rather than
anything about x86-64; a Pi 5 on NVMe simply does this work quickly. **Build natively. Do not
cross-build**, and budget five minutes rather than an hour.

The image is **5.76 GB on disk, 1.3 GB of content**, which is the cost of full `environment.yml`
parity. Budget the disk, not the time.

**The loop may well seal something, and this changed on the day it was first deployed.** Every
earlier draft of this file said the loop would seal nothing, because `fixtures.csv` had never been
observed carrying a Premier League row. On 28 Aug 2026 at 15:12 UTC — the fourth fetch ever taken,
and the first from the Pi — it carried **10**, the whole of that evening's round with market-average
odds attached, and the round was sealable there and then. Treat step 6 as live: if it names Fixtures,
the next `seal` will write into `outputs/live/`, which is append-only evidence that cannot be
rewritten. See `deploy/README.md` and docs/DECISIONS.md, "Bringing it up on the Pi, and the
first Sealed Prediction Round".

Budget half an hour, most of it waiting for step 5.

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

**RAM.** A conda solve is memory-hungry and a 4 GB Pi can be tight. Measured on 28 Aug 2026: an
8 GB Pi 5 with 2 GB of swap already configured built the image without coming near the limit, and
the solve was over in 73 seconds. A 4 GB board is still the untested case. If step 3 dies with the
build process killed, that is the OOM killer rather than a broken Dockerfile; add swap and retry:

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

**If step 1's key is not on GitHub yet, this clone cannot happen** — and adding a deploy key needs a
human at a browser, which is not always the same minute as the rest of this. That is the one place
it is worth getting ahead: the repository is public, so clone over HTTPS and re-point the remote,
then finish step 1 whenever the key lands. The clone is byte-identical either way.

```bash
git clone https://github.com/linuslouie159/epl-predictor.git ~/epl-predictor
cd ~/epl-predictor
git remote set-url origin git@github.com:linuslouie159/epl-predictor.git
```

Nothing before step 7 needs the key, so the build and the ingest can run meanwhile. **Do not leave
it at that**, though: an HTTPS remote is exactly the `could not read Username for 'https://github.com'`
row in step 7's table, and the loop would seal rounds it could never push.

---

## 3. Build the image

Before the ingest below, because the ingest runs inside it.

```bash
cd ~/epl-predictor
docker build -f deploy/Dockerfile -t epl-predictor:live .
```

**Measured on a Pi 5 with NVMe, 28 Aug 2026: 271 seconds total**, of which `conda env create` was
73 s and the image export 174 s. The image comes out at 5.76 GB on disk, 1.3 GB of content.

This step used to carry advice to cross-build on a desktop and `docker save | ssh 'docker load'` the
result, on the strength of a 27-minute amd64 build. **That advice was wrong and has been removed.**
The Pi was six times faster than the desktop it was meant to borrow, so cross-building would trade a
four-minute native build for a multi-gigabyte image transfer plus QEMU. Build here.

Verify:

```bash
docker run --rm epl-predictor:live --help      # the loop's own help text
```

**If it fails**, in rough order of likelihood: the OOM killer (see step 0); or the `grep` guard
failing, which means the pip section of `environment.yml` moved and the Dockerfile's comment
explains what to do. It is a floating `:latest` base — if a rebuild ever behaves differently from a
previous one, pin it to a digest before suspecting anything else.

A conda-forge solve conflict is no longer on that list. It was the named unknown until 28 Aug 2026,
and `linux-aarch64` resolved `environment.yml` as written in about 7 seconds. If it starts failing,
that is upstream drift on a floating base, not this platform.

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

`upcoming` is the whole of `seal`'s work up to the point where `seal` would write. Exit 0 is success,
and there are two shapes it comes in. This is the one seen on the Pi on 28 Aug 2026, and the stamps
are in the *host's* zone — `-0400` here, because this Pi's system zone is `America/New_York`:

```
===== RUN 2026-08-28 11:12:00 -0400  (upcoming ) =====
rolling file: fixtures_20260828T151202Z.csv
10 upcoming Premier League Fixtures in 2026/27
prediction_round as_of_instant  season  fixtures       first_kickoff        last_kickoff   status
      2026-08-28    2026-08-28    2026        10 2026-08-28 20:00:00 2026-08-31 20:00:00 sealable
  2026-08-28 20:00 crystal_palace v man_city
  ...
sealable now: 2026-08-28
===== END  2026-08-28 11:12:03 -0400  (exit 0) =====
```

The other shape names no Fixtures and says "the rolling file held no Fixture in a tier this project
predicts". **Both are exit 0 and neither is a failure.** Which one you get depends on whether
upstream has regenerated `fixtures.csv` since the round was listed, and for the first three fetches
ever taken it was always the empty one — see docs/DECISIONS.md, open risk 7.

**If it says `sealable now:`, stop and read step 7 before running `seal`.** The next `seal` will
write a real Prediction into `outputs/live/`, which is append-only: a round can be superseded at a
new As-Of Instant but never rewritten, and it cannot be sealed at all once its first kickoff has
passed. That is the moment this schedule exists for, and it is not one to walk into by accident.

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

Worth its own step, and it earned that on the first Pi: **this is where the only genuine bug in the
whole deployment turned up**, with everything else already green. Do not skip it because step 6
worked.

```bash
docker compose -f deploy/docker-compose.yml run --rm --entrypoint git live push --dry-run
```

That runs git *inside the container*, with the same mounted key, the same `HOME`, and the same uid
the schedule will use. `Everything up-to-date` — or a `a1b2c3d..e4f5g6h  main -> main` line if the Pi
is ahead — means the whole chain works: key readable, host key accepted, remote reachable, write
access granted.

**The bug, because it will not look like a bug in this file.** ssh resolves `~` from the *passwd
database* for the running uid, not from `$HOME`. Setting `HOME=/home/epl` and mounting the key there
is therefore not enough: this base image has a real `ubuntu` user at uid 1000, so ssh read
`/home/ubuntu/.ssh`, found nothing, and reported `Host key verification failed` — while the correct
`known_hosts` sat mounted and readable a directory away. `docker-compose.yml` now sets
`GIT_SSH_COMMAND` naming the key and `known_hosts` outright, which is what makes this independent of
whatever the floating base image's passwd file happens to contain.

If it fails here, fix it now:

| Message | Cause |
|---|---|
| `Host key verification failed` | **Check the mount path before blaming step 1.** ssh resolves `~` from the passwd database, not `$HOME`, so it reads the uid-1000 user's real home — which in this base image is `/home/ubuntu`, not the `/home/epl` the key is mounted at. `GIT_SSH_COMMAND` in docker-compose.yml names both files outright to settle it. If that is present and correct, *then* it is step 1's `ssh -T` having been skipped; the read-only mount cannot fix that itself |
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

### Then prove `CRON_TZ` actually works, because on the first Pi it did not

This file used to say "Debian's does; some do not" and move on. **Raspberry Pi OS's cron
(3.0pl1-197) ignores it**, measured 28 Aug 2026. Do not take it on trust — grepping the docs proves
nothing either. Fire a probe:

```bash
# a throwaway entry 3 minutes from now, written in LONDON time
LON=$(TZ=Europe/London date -d "+3 minutes" "+%M %H")
echo "should fire at $(date -d '+3 minutes' '+%H:%M %Z') if CRON_TZ is honoured"
rm -f /tmp/cron_tz_probe
crontab -l > /tmp/cur
printf "%s * * * date -u > /tmp/cron_tz_probe\n" "$LON" >> /tmp/cur
crontab /tmp/cur
sleep 240; cat /tmp/cron_tz_probe    # a file means honoured; "No such file" means ignored
```

Remove the probe line afterwards either way.

**If it is ignored, do not set the whole Pi to `Europe/London`.** That used to be the first
suggestion here and it is the wrong one on any Pi with a second tenant: this one also runs a
paper-trading loop whose crontab is written around the US market open, and moving the system zone
would silently reschedule somebody else's job. Convert these three times instead, and **delete the
`CRON_TZ` line while you do** — leaving it behind next to already-converted times is how a future
cron that starts honouring it shifts them a second time.

The conversion on the first Pi, whose zone is `America/New_York`:

| `deploy/crontab` says (UK) | installed as (Pi local) | command |
|---|---|---|
| 06:00 | **01:00** | `score` |
| 16:00 | **11:00** | `seal --push` |
| 18:30 | **13:30** | `seal --push` |

**Why an hour of DST drift does not matter here, and five hours does.** The UK and the US change
clocks on different dates, so for a few weeks a year these land an hour off. The window runs from
midnight to the round's first kickoff and the odds are sampled in the afternoon, so an hour either
way changes nothing. Five hours does: uncorrected, `18:30` fires at 23:30 London, **past a 20:00
kickoff**, and that round can never be sealed — `supersede` refuses a round after its first kickoff
on purpose. The old note here claimed a misconfigured schedule "seals nothing; it cannot seal
something wrongly", and the second half of that is still true. The first half is not a consolation:
a loop that silently seals nothing looks exactly like a week with nothing to seal.

---

## 9. Watch the first real fire

The next scheduled slot is 06:00, 16:00 or 18:30 UK on a Tuesday or Friday — **in whatever those
became on this Pi**, if you applied step 8's conversion. Check with `crontab -l` rather than from
memory. After it passes:

```bash
tail -n 40 ~/epl-predictor/deploy/logs/live_loop.log
```

You want a `===== RUN` block with `(exit 0)`. Cron mails you only when a job exits non-zero, which is
the whole design: a week with nothing to seal is silent, and a round that could not be pushed is not.

**A log with no new `===== RUN` block for a week is what an off Pi looks like from here**, and it is
the mitigation for open risk 6 — a round whose window passes while the Pi is off is lost and cannot
be sealed later.

**And a block that *is* there, says exit 0, and sealed nothing is the harder case.** It means either
a genuinely empty week or a rolling file upstream had not regenerated when both fires read it (open
risk 7), and the log cannot tell you which. On a Tuesday or Friday of a Premier League round, the
second is worth ruling out by hand:

```bash
cd ~/epl-predictor && deploy/run_live.sh upcoming     # writes nothing; safe at any hour
```

Reading the log is the mitigation for one silence and not the other, which is most of the argument
for issue #20's bot.

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
