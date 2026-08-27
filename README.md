# EPL Predictor

Forecasts English Premier League match Outcomes as calibrated probabilities, projects the final table
by simulation, and scores itself honestly against the betting market and against public pundits on
identical metrics.

Vocabulary is defined in [CONTEXT.md](./CONTEXT.md). Decisions and their reasoning are in
[docs/adr/](./docs/adr/). Read both before changing anything — several choices here look wrong until
you know why they were made.

## What "working" means

Measured on the Evaluation Window (2005/06–2025/26, 7,980 Fixtures):

| Predictor | RPS | Meaning |
|---|---|---|
| Naive Baseline | 0.2294 | the floor — beat this or the model has no value |
| Elo | 0.1994 | the first real model, built |
| **Target** | **≤ 0.1986** | market + 0.005 — this is success |
| **Dixon-Coles** | **0.1975** | the goals model, built — **target met** |
| Market Line | 0.1936 | the opponent |
| Ceiling Line | 0.1968* | reference only; knows team news we don't |
| Margin Map (Lawrenson's calls) | 0.2127‡ | the same calls read fairly |
| Margin Map (Sutton's calls) | 0.2111‡ | the same, 2022/23 onward |
| Lawrenson, as-stated | 0.3341† | a published Scoreline taken literally |
| Sutton, as-stated | 0.3343† | the same, 2022/23 onward |

\* over its own 2,660 Fixtures from 2019/20, not the 7,980 above — the two numbers are not
comparable. On the Fixtures they share, the Market Line scores 0.1981 and the Ceiling Line beats it
by 0.0013 RPS, which is what a few hours of team news is worth.

† over the 1,896 and 1,512 Fixtures each Pundit called, and **not a verdict on either of them**. A
Scoreline read as `[1, 0, 0]` is a claim of certainty nobody made, which is what makes this number
worse than the floor while the same calls pick the right Outcome 51% and 49% of the time — seven
points above the floor and four behind the market. See [Pundits](#pundits).

‡ **a one-feature model, not a person** — the same calls with each Scoreline read as what a call of
that predicted goal margin has historically produced ([ADR 0003](./docs/adr/0003-calibrated-pundit-predictor.md)).
Over 1,856 and 1,472 Fixtures, the 40 opening calls of each record having no map behind them yet.
The gap to the as-stated row above — **0.1209 and 0.1235 RPS** — is the cost of stating certainty.

The system does **not** need to beat the market. It needs to be leak-free, well calibrated, and
within a stated distance of the market while beating the Naive Baseline and the Pundits.

Elo alone takes **0.030 of the 0.036 RPS** the market takes out of the floor — 84% of the available
edge from ratings and nothing else. Reading the goals rather than the Outcomes takes **0.032, or
89%**, and clears the target Elo missed by 0.0008. A model that *beat* the market here would be
evidence of a leak, not of a good model.

Interestingly, that 0.0019 does not show up on accuracy: Dixon-Coles picks the winner slightly
*less* often than Elo, 53.6% against 53.8%. The two agree about who wins; the goals model is better
calibrated about how sure it should be, which is what RPS measures and accuracy does not.

Every number above is pre-calibration, which is the better of the two columns the scoreboard
publishes and the one a Predictor earned.

## The one rule

No future data, ever. A Prediction may only use data timestamped strictly before its As-Of Instant,
which is the most recent Tuesday or Friday before kickoff — the same instant the Market Line is
sampled and the same one Pundits publish at. All three see the same information by construction.

Hyperparameters are tuned only on the Burn-In Window (2000/01–2004/05) and frozen. Scoring starts at
2005/06.

## Data

| Source | Use | Coverage |
|---|---|---|
| Football-Data.co.uk `E0`–`E3` | results, match stats, odds | 2000/01– , four tiers |
| Football-Data.co.uk `fixtures.csv` | upcoming Fixtures + Market Line | rolling; **no E0 row seen yet** |
| MyFootballFacts | Pundit backfill (Lawrenson, Sutton) | 2017/18–2025/26, 3,408 rows, frozen |
| MyFootballFacts | Pundit calls for the Season in progress | ~a round behind the football |
| ~~BBC Sport~~ | ~~live Pundit predictions~~ | **not used — see below** |

**The BBC is the origin of every Pundit call and is not a source this project reads.** Issue #16
tested it: it is reachable and its articles are machine-readable, and `bbc.co.uk/robots.txt` forbids
scraping, dataset creation and text-and-data mining, and disallows `ClaudeBot`, `Claude-Web` and
`anthropic-ai` outright. MyFootballFacts permits exactly that and is therefore the only Pundit source
here — but it transcribes a matchday *after* it has been played, so **a Pundit cannot be part of a
Sealed Prediction**. Measured: nought of the next round's ten calls two days before kickoff. See
docs/DECISIONS.md, "The BBC spike".

Odds column availability is era-dependent and this matters: no odds at all before 2002/03,
market-average pre-match (`BbAv*`) from 2005/06 spliced to `Avg*` from 2019/20, market-average closing
(`AvgC*`) only from 2019/20. There is no xG anywhere in Football-Data.

`data/raw/` holds byte-identical cached downloads and is never edited.

## Build order

1. **Ingest** — Football-Data `E0`–`E3` + `fixtures.csv`; Club/Alias resolution
2. **Metrics** — RPS, Brier, log loss, accuracy, calibration. Unit-tested against hand-worked examples.
   Prediction Rounds are derived here too, since every later stage is scoped by them
3. **Elo** — pyramid-wide ratings → ordered logit → shared calibration layer
4. **Benchmarks** — Market Line (vig removed) and Naive Baseline
5. **Pundits** — backfill, grading, Calibrated Pundit, three-way scoreboard
6. **Dixon-Coles** — Scoreline probabilities, MLE per Prediction Round
7. **Season Projection** — Bayesian posterior → Monte Carlo → title / top-4 / relegation
8. **Live loop** — seal each Prediction Round before kickoff, and score it after
9. **Frontend** — reads `outputs/` only; no modelling logic

Stages 5 and 7 were moved: the Season Projection needs goal difference (24 of 26 Seasons had points
ties), so it cannot be built on Elo alone; the Pundit tracker needs no model at all, so it comes early.

The list above is the shape of the system, not the order the tickets unblock in. The **Prediction
ledger and the Naive Baseline (#7) come before Elo**, not after the benchmarks: one Predictor has to
go through the whole pipeline and out the other side scored before a real model is worth building on
top of it. The tracker's `Blocked by` fields are authoritative where the two disagree.

## Deferred to v2

XGBoost / ML layer, the Golden Boot player model, and the API-Football client. Each gets a written
stub explaining what it would do and what it needs. API-Football was deferred because `fixtures.csv`
already carries upcoming Fixtures with the Market Line, free and unauthenticated — a premise stage 13
measured and could not confirm. See [The live loop](#the-input-is-the-part-that-is-not-proven).

The stubs are in **`src/epl/v2/`** — prose and named constants, no implementation. They are Python
modules rather than a docs page so that "nothing in the pipeline imports one" is a thing a test can
check, and `tests/v2/test_stubs_are_unreachable.py` checks it: deleting the directory would break no
import and move no number. Each carries its entry price as a `WHAT_IT_NEEDS` tuple, because the
sentence a stub loses first is the one saying what it would take to pick it up.

`api_football.py` is the live one. It holds the two fetches that failed to find a Premier League row
(`FETCHES_MEASURED`), the count that decides the question (`PREMIER_LEAGUE_ROWS_SEEN`, currently 0),
and the conditions that would revive the client.

## Layout

```
data/raw/               cached downloads, byte-identical, never edited
data/processed/         cleaned tables
src/epl/windows.py      Season identity, Burn-In and Evaluation Windows
src/epl/rounds.py       the As-Of Instant rule and Prediction Rounds
src/epl/predictors.py   the Predictor contract, the Evidence a Predictor sees, the registry
src/epl/calibration.py  the shared isotonic layer every Predictor's output passes through
src/epl/ingest/         football-data fetch + clean; the raw cache write rule (cache.py)
src/epl/clubs/          canonical Club table + Alias resolution
src/epl/metrics/        RPS, Brier, log loss, calibration measured
src/epl/models/         elo.py, ordered_logit.py, burn_in.py
src/epl/models/likelihood.py    the Dixon-Coles likelihood both fits share (ADR 0007)
src/epl/models/dixon_coles.py   the maximum-likelihood fit and the Predictor over it
src/epl/benchmarks/     market line + ceiling line (vig.py), naive baseline
outputs/overround.csv   the margin in each book per Season; regenerable, gitignored
src/epl/pundits/        myfootballfacts.py, dataset.py, grading.py, predictor.py
src/epl/pundits/margin.py         the margin map; calibrated.py the Predictor over it
src/epl/pundits/report.py         the three-way board, the certainty gap, the calls by miss
src/epl/pundits/live.py           the Season in progress, and why it cannot be sealed from
src/epl/pundits/predictions.csv   the frozen backfill: 3,408 calls, committed with the code
outputs/three_way.csv   the board over each Pundit's shared Fixtures; regenerable, gitignored
outputs/certainty.csv   the two readings and the gap between them; regenerable, gitignored
outputs/pundit_calls.csv  every call ranked by miss; regenerable, gitignored
outputs/margin_map.csv  what a call of each margin is worth; regenerable, gitignored
outputs/sequential.csv  every Fixture predicted per round and per kickoff; regenerable, gitignored
src/epl/simulate/       Bayesian fit + Monte Carlo Season Projection
src/epl/simulate/posterior.py     the same likelihood sampled instead of maximised (ADR 0007)
src/epl/simulate/checkpoints.py   the handful of rounds a posterior is allowed to run at
src/epl/simulate/table.py         the final league table and the chain that breaks a tie in it
src/epl/simulate/projection.py    the 10,000-Season walk over the draws, and the two seeds
src/epl/simulate/validation.py    where the real champion landed, across completed Seasons
outputs/projection.csv  one Season Projection: title, Europe, relegation; regenerable, gitignored
outputs/projection_validation.csv  every projection against what happened; regenerable, gitignored
src/epl/ledger/         Prediction stores, the row audit, the scoreboard
src/epl/ledger/live.py  the sealed store: the window, the revisions, and the seal audit
outputs/backtest/       regenerable, gitignored — one file per Predictor
outputs/live/           SEALED, committed, append-only — one file per Prediction Round
outputs/scoreboard.csv  every Predictor over the Evaluation Window; regenerable, gitignored
outputs/reliability.csv 10-bin diagrams per Predictor, raw and calibrated; regenerable, gitignored
src/epl/live/           the live loop: the upcoming round, the seal, the retrospective score
src/epl/live/upcoming.py   the rolling fixtures file -> the one round that can be sealed now
src/epl/live/seal.py       every registered Predictor over that round, sealed and committed
outputs/live_scoreboard.csv  the Season in progress, scored on its own board; gitignored
src/epl/v2/             the deferred features, written down rather than built — no code runs here
deploy/                 the schedule: the image, the compose file, the cron wrapper, the crontab
deploy/logs/            what an unattended run left behind; machine-local, gitignored
```

Three modules are called `live` and they are different things: `epl.ledger.live` is the sealed
*store*, `epl.pundits.live` is stage 12's spike into the Pundit archive, and `epl.live` is the loop
that uses both.

`calibration.py` sits beside the contract it wraps rather than inside `models/`, because it is not a
model: it takes Predictions and returns Predictions, and it treats the Market Line and a Pundit
exactly as it treats Elo.

## Environment

Miniforge + conda-forge; `environment.yml` is the source of truth. PyMC with `nutpie` for the
Bayesian fits, pytest for tests. See [ADR 0009](./docs/adr/0009-conda-forge-toolchain.md) for why not
pip and venv.

```
conda env create -f environment.yml
conda activate epl-predictor
```

Two pins in there are load-bearing rather than preferences, and both break the build outright rather
than drifting: `arviz <1`, without which `import pymc` fails, and **`libblas=*=*openblas`**, without
which conda selects MKL 2026 and every LAPACK call aborts the interpreter — taking `numpy.linalg`,
`scipy.linalg`, L-BFGS-B and PyMC with it. See docs/DECISIONS.md, "Added at stage 9".

## Running the ingest

```
python -m epl.ingest fetch              # fill data/raw/ — 108 files, 27 Seasons x 4 tiers, ~16 MB
python -m epl.ingest fetch --refresh    # re-download, including the growing current Season
python -m epl.ingest build              # write matches.csv (52,672 closed) + odds_availability.csv
python -m epl.ingest fixtures           # fetch the rolling forward-Fixture file + Market Line
python -m epl.ingest clubs              # audit: Club spellings the Alias table does not know
python -m epl.clubs.build               # rebuild clubs.csv, aliases.csv, teamname_replacements.json
```

The 27th Season is the one being played. It is ingested from stage 13 because the live loop cannot
predict it otherwise and cannot score what it sealed — and it is in **neither Window**: never fitted
on, never backfilled, scored on its own board. `matches.csv` therefore grows every Saturday, which is
why every fixed count in the corpus tests is over the 26 closed Seasons and the Season in progress is
checked for being partial instead.

`clubs` exits non-zero on an unknown spelling. That is deliberate — a Club whose name fails to map
would have its rating history split in two at the point the spelling changed, and pyramid-wide Elo
would carry that split across a promotion without anything looking wrong.

`--refresh` does not overwrite. When upstream's bytes differ from the cached ones, the cached copy
moves to `superseded/` first, because the current Season's file backfills results and odds into rows
already published — and losing what the cache held at seal time is exactly the failure
[ADR 0005](./docs/adr/0005-split-prediction-ledger.md) exists to prevent.

Narrowing `--seasons` or `--divisions` prints a warning: a subset table is fine for a quick look and
must not be used to score a Predictor ([ADR 0004](./docs/adr/0004-rate-the-whole-pyramid.md)).

`odds_availability.csv` records, per Season and tier, which odds columns the source actually carried
— Bet365 from 2002/03, `BbAv*` from 2005/06, `Avg*` and `AvgC*` from 2019/20. It exists because once
odds are a float column full of nulls, a Season with no market is indistinguishable from one the
market priced at nothing. Benchmark code reads this rather than re-deriving the era boundaries.

Everything that reaches upstream takes an injectable `fetcher`, so tests run against local fixtures
with no network access and without patching anything:

```python
from epl.ingest import fetch_season, mapping_fetcher, directory_fetcher
fetch_season(2025, "E0", fetcher=directory_fetcher("tests/data"))
```

## Prediction Rounds

A Fixture's As-Of Instant is midnight at the start of the most recent Tuesday or Friday before its
kickoff, and Fixtures sharing one form a Prediction Round. Over 2000/01–2025/26 that is **1,189
rounds**, 45.7 per Season, 8.31 Fixtures each.

```python
from epl.ingest import load_matches
from epl.rounds import assign_rounds, prediction_rounds

matches = load_matches(divisions=("E0",))
assign_rounds(matches)      # every Fixture gains as_of_instant + prediction_round
prediction_rounds(matches)  # one row per round, with the first kickoff it must be sealed before
```

`assign_rounds` raises rather than returning a frame in which any Fixture kicks off at or before
the instant it is predicted from. The design recorded 1,332 rounds; the anchor rule it states as
code produces 1,189, and [docs/DECISIONS.md](./docs/DECISIONS.md) records why the rule was kept and
the count corrected.

## Scoring

```python
from epl import metrics

metrics.rps([0.5, 0.3, 0.2], "H")          # 0.145 — the primary metric
metrics.score(predictions, outcomes)        # RPS, Brier, log loss, accuracy, in one Scorecard
metrics.reliability(predictions, outcomes)  # the 10-bin reliability diagram
```

No function in `epl.metrics` takes a Predictor, and the package imports nothing that can produce a
Prediction — asserted structurally, so the three-way scoreboard cannot become apples-to-oranges.

## Tests

```
pytest                  # unit tests, plus corpus integrity checks when data/raw/ is populated
pytest --run-network    # also hits football-data.co.uk and the MyFootballFacts index
```

Tests marked `cache` re-derive the measured facts in [docs/DECISIONS.md](./docs/DECISIONS.md) from
the ingested corpus rather than trusting them, and skip when `data/raw/` is absent — including the
nine cached Pundit pages, which `python -m epl.pundits fetch` puts there. Tests marked `network`
check that upstream still serves the shapes the live loop assumes.

Every fixed count in a `cache` test is over the **26 closed Seasons**. The corpus also holds the
Season in progress, which grows every Saturday, so a count including it would be a test that failed
weekly and told nobody anything; that Season is checked for being present and partial instead.

`tests/live/test_live_loop_over_the_corpus.py` is the one that cannot be done with two invented
Clubs: it registers all nine Predictors against the real corpus and pins which four can speak to a
Fixture nobody has played. Its Fixtures are hand-built, and the file says so — Football-Data's rolling
file has never been seen carrying a Premier League row, so a real upcoming round is exactly what
cannot be obtained. Nothing it asserts depends on who is playing whom.

## Predictions, the ledger and the scoreboard

A **Predictor** is anything with a name and a `predict`. It is handed one Prediction Round's
Fixtures and the **Evidence** visible at that round's As-Of Instant, and returns one probability
distribution per Fixture:

```python
from epl.predictors import Evidence, register

class MyModel:
    name = "my_model"

    def predict(self, fixtures, evidence):
        seen = evidence.matches(divisions=("E0",))   # everything already played, and nothing else
        ...                                          # -> an (n, 3) array over (Home, Draw, Away)

MY_MODEL = register(MyModel())
```

Evidence is the corpus already cut at the instant, rather than the instant with an invitation to go
and read — a Predictor cannot reach a row it should not have. It also records what it handed over,
so every stored Prediction carries `inputs_seen` and `latest_input`, and the audit can re-check the
project's one rule off the file months later.

`fixtures` is guarded the same way and for the same reason. A Fixture "carries no result until it is
played", so `predict` sees only `schema.VISIBLE_FIXTURE_COLUMNS` — who is playing, when, and the
Market Line the Fixture was priced at. The corpus is a table of *played* matches, and handing over
the rest of the row would deliver the answer sheet in the same call as the question.

```
python -m epl.ledger backfill      # walk every registered Predictor over the Evaluation Window
python -m epl.ledger scoreboard    # score both stores twice; write outputs/scoreboard.csv
python -m epl.ledger reliability   # the 10-bin diagrams, raw and calibrated
python -m epl.ledger audit         # re-check every stored row, and the seal on outputs/live/
```

Both stores share one row schema, so scoring never knows which it is reading. `audit` is the one to
run in anger: rows are checked on the way *into* both stores, so it only ever fails on a file that
changed after it was written — including any file under `outputs/live/` whose git history shows a
commit at or after its round's first kickoff.

The Naive Baseline is the floor, and it is fitted walk-forward: at each round it counts only the
Outcomes its Evidence holds. It scores **0.22938 RPS** over the Evaluation Window's 7,980 Fixtures.
The published 0.2292 is the whole-window figure, computed from rates that already know how the
window turned out; the 0.0002 difference is the leak being refused, not a bug.

## The live loop

Everything above looks backwards. The live loop produces the other kind of record: a **Sealed
Prediction**, written before its Fixture kicked off and committed, so that git history — not a
timestamp inside a file, which anyone could type — is the proof of when it existed.

```
python -m epl.live upcoming  # what the rolling fixtures file holds, and what could be sealed
python -m epl.live seal      # predict the upcoming round, write it to outputs/live/, commit it
python -m epl.live seal --push       # and push it, which is what makes it evidence off this machine
python -m epl.live score     # ingest results, then score what has been sealed
deploy/run_live.sh seal --push       # the same, in the container cron fires (deploy/README.md)
```

**A round may be sealed only inside its own window**: at or after its As-Of Instant, and strictly
before its first kickoff. Both ends are refusals rather than warnings. Sealing early claims a moment
that has not happened; sealing late is a claim about what the code would have said, which is the one
thing this store must never hold.

**Five of the nine registered Predictors say nothing, and nothing in the loop knows which five.**
Two Pundits have published no call the frozen dataset holds, two Calibrated Pundits have no call to
read, and the Ceiling Line has no closing odds because the match has not closed. Each answers
`covers` with nothing, exactly as it does for a Season it never covered, and the run *records* which
Predictors were silent rather than being told in advance. Elo, Dixon-Coles, the Market Line and the
Naive Baseline are what get sealed.

**A sealed round is corrected by superseding it, never by rewriting it.** `seal --supersede` writes
a new revision of the round's file — `2026-08-28.csv`, then `2026-08-28.1.csv` — at a new As-Of
Instant, and every superseding row must be stamped strictly later than the row it replaces. Such a
Prediction may genuinely know more than the one it replaces, because it was genuinely made later;
stamping it at the round's original midnight to keep the comparison tidy is the fiction
[ADR 0005](./docs/adr/0005-split-prediction-ledger.md) exists to prevent. Running the loop twice in
one round is a no-op, which is what lets it be put on a schedule.

**The Live Season is scored on its own board.** It is ingested — a Predictor cannot see it
otherwise, and a sealed Prediction cannot be joined to a result — but it is never backfilled and
never folded into the Evaluation Window. `outputs/scoreboard.csv` therefore says what it said last
week; `outputs/live_scoreboard.csv` is where the live record accumulates. `score` refreshes the Live
Season from upstream and rebuilds the match table itself, so scoring a played round is one command
and not three.

**The loop runs on a schedule, on a Raspberry Pi, in a container** — `deploy/`, issue #19. The Pi's
crontab fires `seal --push` at 16:00 and again at 18:30 UK on Tuesdays and Fridays, and `score` at
06:00 on the same days. 16:00 is after Football-Data samples the afternoon's odds and before a 19:45
kickoff; 18:30 is a retry, and it is free because sealing is idempotent inside a round.

**The exit code is the whole interface, and its one important line is that nothing to seal is a
success.** No round is inside its window most of the week, and until upcoming Fixtures have a source
it is *every* fire — so a loop that went red twice a week would teach its owner to ignore it.
`epl.live.upcoming.NothingToSeal` separates the two silences that need nobody from the refusals that
do; a round sealed but not committed, or committed but not pushed, exits 1 and says so, because the
file on disk looks identical either way.

**The push is deliberate and is recorded as such.** A commit proves *when* to anyone who can reach
the machine holding it, and on a Pi in a cupboard that is nobody — so an unattended loop that does
not push has not finished sealing anything. It is opt-in (`--push`), over a deploy key. Actions was
weighed and rejected: it would re-fetch the whole raw cache twice a week forever, because
`score` rebuilds the match table from all 108 files and the cache is gitignored. The price accepted
in exchange is that **a round whose window passes while the Pi is off is lost** — open risk 6.
See docs/DECISIONS.md, "The schedule, and where it runs".

### The input is the part that is not proven

The loop is built and tested end to end. What it reads is not. Football-Data's rolling
`fixtures.csv` is the only confirmed source of upcoming Premier League Fixtures with a Market Line,
and **on every fetch so far it has held none**:

| Fetched | Upstream `Last-Modified` | Rows | E0 rows |
|---|---|---|---|
| 2026-08-21 06:33 UTC — 2026/27 round 1 kicked off that evening | not recorded | 3 | **0** |
| 2026-08-27 06:12 UTC — the day before round 2 | Tue 25 Aug 09:59 GMT | 5 | **0** |
| 2026-08-27 14:21 UTC — same day, eight hours later | Tue 25 Aug 09:59 GMT | 5 | **0** |

Every time, the file held only Fixtures dated on or before the day it was generated — one League One
tie and two Spanish on the first, one National League tie and four Spanish on the other two — and the
second was already two days stale. So the file is regenerated irregularly and its forward horizon at
generation time is a couple of days, which is shorter than a Prediction Round's own window.

The third fetch came back **byte-identical** to the second, eight hours later on the eve of a round:
same md5, same `Last-Modified`. That rules out the hopeful reading of the first two — a fetch timed
later in the day would not have caught a fresher batch, because upstream had not written one in two
and a half days.

The first fetch's `E2` row is worth noticing: this is not a file that omits English football. It has
carried an English tier this project ingests. It has never been seen carrying the one tier this
project predicts.

This is issue #17's open risk 2, and it is **documented rather than closed**. Three fetches are not
proof that a Premier League row never appears; they are proof that one has not been seen yet, and
that nothing in this project should assume one will. `python -m epl.live upcoming` is what answers the
question on any given day, and it answers it without writing anything.

Issue #19's schedule now asks it twice a week, which changes who is watching rather than what the
answer is. A fire that finds no Premier League row is a quiet success by design, so the evidence
accumulates in `deploy/logs/live_loop.log` — and **nothing will announce the day the answer
changes.** The loop will simply start sealing rounds.

The consequence reaches [Deferred to v2](#deferred-to-v2): the reason given for not needing an
API-Football client is that "`fixtures.csv` already carries upcoming Fixtures with the Market Line".
That premise is exactly what is in doubt, and issue #18's stub records the measurement rather than
the assumption.

## The market benchmarks

The **Market Line** is the opponent: the market-average *pre-match* book with the vig removed. It
scores **0.19362 RPS** over the Evaluation Window's 7,980 Fixtures — 0.036 better than the floor,
which is the gap a model has to find some of.

The **Ceiling Line** is the same arithmetic on the *closing* book, from 2019/20. It is a reference
upper bound and never the headline opponent, and it carries a note onto the scoreboard saying so,
because its raw RPS looks worse than the Market Line's purely by being measured over a different,
harder span ([ADR 0001](./docs/adr/0001-pre-match-odds-as-market-benchmark.md)).

Vig removal offers three methods behind one interface, with **Shin as the default**:

```
python -m epl.benchmarks methods       # one book under all three
python -m epl.benchmarks overround     # the margin per Season and tier; writes outputs/overround.csv
```

| Method | RPS | |
|---|---|---|
| normalisation | 0.19379 | divides the book out proportionally |
| Shin | 0.19362 | **default** — corrects favourite-longshot bias from a stated mechanism |
| power | 0.19359 | corrects it harder, from a free exponent |

A spread of 0.0002, so the choice is near-immaterial for benchmarking — which is the point of
being able to run all three rather than being told. The overround report is the removal's receipt:
the margin falls from 9.4% in 2005/06 to about 4.1% in the early 2020s, and a removal that had
quietly stopped removing anything would show up there before it showed up on the scoreboard.

Seasons 2000/01–2001/02 carry no odds at all. Each line declares which Fixtures it `covers`, so
those Seasons produce no rows rather than invented ones — no market comparison, not a market
comparison of zero.

## Elo

One rating pool across all four tiers, folded forward in kickoff order, mapped onto three
probabilities by an ordered logit. It scores **0.19943 RPS** over the Evaluation Window.

```python
from epl.models import ELO, Ratings, Settings

ELO.predict(fixtures, evidence)        # (n, 3) over (Home, Draw, Away)
ELO.ratings_at(evidence).rating("leeds")
```

Nothing in the model knows what a division is. A Club that is promoted is a Club whose next
opponents happen to be better, which is the whole mechanism — Elo is zero-sum, so ratings stay
comparable across tiers connected only by promotion and relegation (ADR 0004). Measured: the three
Clubs promoted for 2005/06 arrive on 1679.8, 1623.9 and 1565.6, each from over 200 matches, and by
the first scored Prediction Round the thinnest Premier League rating rests on 190.

The draw band is never coded, only fitted. Across ten Supremacy deciles Elo's predicted draw rate
falls **30.2% → 14.5%**, monotonically, and the rate that actually happened falls 27.6% → 13.8%.

```
python -m epl.models fit       # re-derive the frozen hyperparameters on the Burn-In Window
python -m epl.models draws     # the draw rate against Supremacy, predicted and observed
python -m epl.models ratings   # the pool at a Season's first Prediction Round
```

`epl.models.burn_in` is the only place a hyperparameter may be fitted, and it cuts the corpus to
2000/01–2004/05 before it walks a single match — so handing it the whole 26 Seasons is
indistinguishable from handing it the five (ADR 0008). 2000/01 warms the ratings and is not fitted
on. What it found — K 28.5, home advantage 80 rating points, a logit scale of 186.9 and a draw band
of ±0.6232 — is frozen as literals in `models/elo.py`, and `python -m epl.models fit` prints the
fit beside them so the two cannot drift apart in silence.

## Dixon-Coles

The goals model, and the first Predictor to clear the target. Each Club gets an attack and a
defence, the home Club is expected to score `exp(attack_home − defence_away + home_advantage)` and
the away Club `exp(attack_away − defence_home)`, and the probability of every Scoreline up to 15–15
collapses onto the same three Outcomes everything else on the board is scored on. It scores
**0.19752 RPS**.

```python
from epl.models import DIXON_COLES

DIXON_COLES.predict(fixtures, evidence)      # (n, 3) over (Home, Draw, Away)
DIXON_COLES.scorelines(fixtures, evidence)   # (n, 16, 16) over exact Scorelines
DIXON_COLES.strengths_at(evidence).table()   # one attack and one defence per Club
```

The likelihood lives in `models/likelihood.py`, apart from the fit, because **the Bayesian
posterior at #14 has to share it** — ADR 0007 fits this model two ways precisely so the expensive
tool goes only where it does real work, and one shared likelihood is what stops the two from
becoming two models. Nothing in that module knows what an optimiser or a Predictor is.

It rates all four tiers, like Elo and for a different reason. Elo is zero-sum, so a rating survives
a promotion by construction; here nothing in the likelihood knows a division exists, and the tiers
are joined **only** by the Clubs that changed tier inside the decay horizon. That turns out to be
enough: mean attack falls monotonically E0 → E3 (+0.45, +0.11, −0.09, −0.31) with nothing ordering
them, and fitting the Premier League alone costs 0.002 RPS on the Burn-In Window.

One hyperparameter, fitted in the Burn-In Window and frozen: a **time-decay half-life of 322.5
days**. Anything from 270 to 480 scores within 0.0001 RPS of it, so it is a well-determined region
rather than a well-determined number; what the data does exclude is the short end, where a 60-day
half-life costs 0.007 RPS.

Dixon-Coles' famous low-score correction — the τ that lifts 0-0 and 1-1 and lowers 1-0 and 0-1 —
**has all but vanished from this corpus**. The 1997 paper fitted about −0.13 on four Seasons of one
division; here it wanders around zero and changes sign, and pinning it at zero costs 0.00011 RPS.
It is kept because it belongs to the shared likelihood rather than to this fit, and because 0.00011
in the right direction is a measurement.

```
python -m epl.models fit          # both fits re-derived on the Burn-In Window, beside the literals
python -m epl.models strengths    # the attack and defence table at a Season's first round
python -m epl.models sequential   # ADR 0002's diagnostic — see below
```

### What the weekly batch gives up

ADR 0002 predicts in weekly Prediction Rounds on purpose, withholding Saturday's results when it
calls Monday night's game, so that the model, the market and the Pundits all see the same
information. It also promises to measure what that costs, and `python -m epl.models sequential`
is where: every Fixture is predicted twice by the same Predictor, once from its round's As-Of
Instant and once from an Evidence cut at its own kickoff.

| Predictor | Over | Fixtures | Batch RPS | Sequential RPS | Withheld |
|---|---|---|---|---|---|
| Elo | whole window | 7,980 | 0.19943 | 0.19942 | **+0.00001** |
| Elo | 2019/20 on | 2,660 | 0.20525 | 0.20523 | **+0.00002** |
| Dixon-Coles | whole window | 7,980 | 0.19752 | 0.19749 | **+0.00003** |
| Dixon-Coles | 2019/20 on | 2,660 | 0.20343 | 0.20338 | **+0.00006** |

**It costs essentially nothing** — two to three orders of magnitude below the 0.0019 that reading
the goals is worth, and below the resolution anything here is reported at. The comparability
ADR 0002 bought turns out to have been very nearly free.

Neither number is a score, and only the first is on the scoreboard. The second is what a model that
broke the comparison would get, and it is bounded rather than exact in two directions that cancel:
before 2019/20 no kickoff time is recorded, so a Fixture cannot see its own day; after it, a cut at
kickoff can see a match still being played.

## The Season Projection's fit

The same Dixon-Coles, sampled rather than maximised, and run at a handful of Prediction Rounds
rather than all 952 (ADR 0007). One posterior fit takes about **four minutes** against the MLE's
0.22 seconds, which is exactly why it is confined to Season Projection points: weekly during a live
Season, and roughly six checkpoints per historical Season for validation.

```
python -m epl.simulate checkpoints   # where a Season is projected from, and where it is not
python -m epl.simulate posterior     # fit one, and read it beside the MLE of the same model
```

**Match probabilities and Season Projections come from formally different fits of the same model**,
and every output that puts them side by side says so. Everything on the scoreboard is the maximum-
likelihood path; only a projection is drawn from the posterior.

The two agree, which is what licenses the split at all: at the first Prediction Round of 2015/16,
Bournemouth v Aston Villa comes out H 0.5340 / D 0.2381 / A 0.2279 by maximum likelihood and
H 0.5430 / D 0.2381 / A 0.2190 at the posterior mean — **0.0090 apart**. What the expensive fit adds
is not a better point estimate but the spread around it: across the draws that same Fixture's Home
probability spans 0.42 to 0.77. Ignoring that spread is what makes a naive season simulator report a
48% title probability where the honest answer is 34%.

There is no second likelihood. The sampler is handed
`epl.models.likelihood.negative_log_likelihood` itself as one opaque node — it already returns its
own analytic gradient — so the arithmetic being sampled is the arithmetic being optimised rather
than a copy of it that could drift.

## The Season Projection

A distribution over final league tables: every Fixture that has not been played is simulated to a
Scoreline, the Season is resolved through the full tiebreaker chain, and **10,000 simulated Seasons**
are counted into a probability of the title, of a European place and of relegation for every Club.

```
python -m epl.simulate project       # one projection: the title, Europe and relegation
python -m epl.simulate validate      # project completed Seasons and see where the champion fell
```

Strengths are **drawn from the posterior on every simulated Season**, never fixed at a point
estimate. That is the whole reason the expensive fit exists, and on real football it is worth what
ADR 0007 says: at 2011/12's first checkpoint Manchester United's title probability is 0.853 from
the posterior mean, 0.850 from the MLE and **0.772 from the draws**. The point estimate and the
posterior mean agree with each other and disagree with the honest answer.

**Two seeds, both recorded** — the sampler's and the walk's — so a published projection is
reproducible exactly. The walk itself is cheap: 10,000 Seasons over half a Season of Fixtures and
4,000 draws takes **2.7 seconds** against the posterior fit's nine minutes in front of it.

The chain that settles a tie is **points, goal difference, goals scored, head-to-head points,
head-to-head away goals, then a play-off at a neutral ground taken as a coin flip**. Ties on points
are routine — 24 of the 26 ingested Seasons had one, 85 pairs in all — and goal difference settles
every one of them: **no real final table in 26 years has ever needed the head-to-head steps.** They
are there for the simulated tables, and each projection reports how many of its own needed them.

The chain here has two more steps than the competition's own regulation, which goes straight from
goals scored to the play-off. That is the ticket's specification, it is a strict refinement — it
changes nothing the regulation decides and replaces some coin flips with a rule — and it is stated
rather than assumed in `epl.simulate.table`.

**European places means the top four of the league table**, which is a simplification stated in the
code: England has had a fifth place in some recent Seasons on UEFA's coefficient, and a European
place can also be won by lifting a cup, which is not a league position at all.

2015/16 at Christmas is the worked example, and it is not a flattering one. Leicester led on 38
points with 210 Fixtures left; the projection gives them **0.058** for the title behind Arsenal's
0.517 and Manchester City's 0.357. They won it — which is why the Season is remembered, and why the
number is quoted here rather than a comfortable one.

### Is it calibrated, or only plausible?

Measured, over **60 projections across the 20 completed Evaluation Window Seasons**:

| | |
|---|---|
| the eventual champion was the projection's favourite | **73%** of the time |
| …and in its top three | **97%** (58 of 60) |
| mean title probability it gave the eventual champion | **0.581** |
| ten-bin calibration error (title / Europe / relegation / pooled) | 0.012 / 0.012 / 0.018 / **0.008** |

For scale, every match Predictor on the scoreboard sits at a ten-bin error of about 0.006 and the
margin map at 0.019, so a Season Projection is about as well calibrated as the model it is built on.

And it tightens, which is what taking six checkpoints across a Season was for:

| Fixtures left | probability given the eventual champion | champion was the favourite |
|---|---|---|
| 331 | 0.405 | 55% |
| 207 | 0.609 | 85% |
| 102 | 0.729 | 80% |

Leicester is the whole exercise in three rows: **0.0008** in September (eighth favourite of twenty),
0.060 at Christmas, **0.526** and favourite by March. The largest miss is 2011/12, where Manchester
City won on goal difference on the final day and the projection never made them favourite at any
checkpoint.

**These points are not independent** — twenty Clubs share one table, three checkpoints share one
champion — so the diagram is a shape rather than a significance test, and `validate` prints that
caveat under every one it produces.

## Calibration

One shared isotonic step wraps every Predictor identically — Elo, Dixon-Coles, both market lines,
the Naive Baseline and the Pundits. A Predictor gets it by being registered; there is no calibration
code in any Predictor and nowhere for a per-Predictor branch to be added (ADR 0006).

It is fitted **walk-forward on out-of-sample Predictions only**. At each Prediction Round the map is
built from the Predictions whose Fixtures kicked off *strictly before* that round's As-Of Instant —
the same cut `Evidence` applies to the corpus, because an Outcome is not knowable until its Fixture
has been played. This is the one thing in the project fitted on results, so it is the one place a
leak could enter with every stored row still auditing clean.

A map needs **380 Predictions** behind it — one Season of Fixtures — before it is fitted at all, so
the first 380 of every track record pass through uncorrected. That is what "out-of-sample only"
costs, and the scoreboard's `corrected` column reports it rather than hiding it: 7,600 of 7,980.

**Every metric is reported twice**, pre-calibration and post-calibration, with the size of the
correction beside them. That rule is why this section can tell you something inconvenient:

| Predictor | RPS | calibrated | ten-bin error | calibrated | mass moved |
|---|---|---|---|---|---|
| Market Line | 0.19362 | 0.19450 | 0.0061 | 0.0124 | 0.034 |
| Ceiling Line | 0.19676 | 0.19800 | 0.0060 | 0.0084 | 0.033 |
| Dixon-Coles | 0.19752 | 0.19793 | 0.0080 | 0.0102 | 0.038 |
| Elo | 0.19943 | 0.20037 | 0.0055 | 0.0097 | 0.031 |
| Naive Baseline | 0.22938 | 0.23087 | 0.0061 | 0.0161 | 0.046 |

**Calibration makes every Predictor worse**, by about 0.001 RPS, moving 3–5% of each Prediction's
probability mass to do it. Two things are behind that:

- **Knot resolution.** A map gets a knot per distinct quote, and market odds and Elo edges are
  nearly continuous — 7,909 distinct Home quotes across 7,980 Fixtures — so most knots rest on one
  Fixture and the map fits noise. Cutting the knots at ten probability bands recovers most of the
  loss (Elo 0.20037 → 0.19968, market 0.19450 → 0.19404). Not shipped: the band count would be a
  hyperparameter, and those are fitted in the Burn-In Window, which holds no stored Prediction.
- **The corpus.** Even coarse, both stay worse than raw. All four are already well calibrated, so
  there is little real miscalibration left to find. A clean split half says the same without the
  walk: fitted on the older half of the market's Fixtures, the map improves that half by 0.0017 RPS
  and costs the later half 0.0005.

The correction is not pointing the wrong way — Elo's draw quote at even Supremacy moves 30.2% →
29.3% against 27.6% observed, exactly the defect it was built for. The noise around it is simply
larger than the signal. So the headline numbers stay pre-calibration, and both columns keep being
published: a 0.001 RPS tax applied silently to every Predictor is precisely what reporting twice
exists to catch.

```
python -m epl.ledger scoreboard    # both tables, and the correction beside the second
python -m epl.ledger reliability   # 10 bins per Predictor per form; outputs/reliability.csv
python -m epl.models draws         # the draw curve, quoted and calibrated, against observed
```

Nothing is stored. A calibrated Prediction is a function of a stored Prediction *and of Outcomes
that happened after it*, so it is derived at scoring time — no row in either ledger store knows an
Outcome, and that is what makes a leaked Prediction distinguishable from a recorded one
([ADR 0005](./docs/adr/0005-split-prediction-ledger.md)).

## Pundits

Nine Seasons of published Scorelines, collected from MyFootballFacts' archive of the BBC column and
**committed as a frozen dataset** at `src/epl/pundits/predictions.csv`, so the accountability
feature is backtestable on a fresh clone rather than in a year's time. Mark Lawrenson worked
2017/18–2021/22 and Chris Sutton 2022/23–2025/26; they are two named Predictors, not one pundit
slot, and each `covers` only the Seasons they worked.

```
python -m epl.pundits fetch    # cache the nine archive pages
python -m epl.pundits build    # parse, reconcile with the corpus, freeze predictions.csv
python -m epl.pundits grades   # exact-score and correct-Outcome rates per Pundit and Season
python -m epl.pundits three-way  # the three-way board and the cost of stating certainty
python -m epl.pundits live     # the Season in progress: what the archive has, and how late
```

**3,408 calls of a possible 3,420.** Only facts are stored — Fixture, predicted Scoreline, Pundit,
date. No prose, no matchday heading, and not the result: a stored Prediction that knew its own
Outcome is what [ADR 0005](./docs/adr/0005-split-prediction-ledger.md) exists to prevent. The result
*is* read at build time, to check the parse against Football-Data — 3,402 of the 3,406 calls that
carry one agree — and then discarded.

| | Calls | Exact score | Correct Outcome | RPS as-stated |
|---|---|---|---|---|
| Lawrenson | 1,896 | 11.0% | 50.9% | 0.3341 |
| Sutton | 1,512 | 9.1% | 49.2% | 0.3343 |

Both readings are published because each alone is an argument
([ADR 0003](./docs/adr/0003-calibrated-pundit-predictor.md)). On the Fixtures they called, the
Market Line scores 0.1946 and 0.1968 and the Naive Baseline 0.2356 and 0.2319 — so as-stated, a
Pundit is a tenth of a point *below the floor*. On accuracy, which asks who they picked rather than
how sure they claimed to be, the same calls beat the floor by seven points and trail the market by
four. Nothing about the Pundit changes between those two sentences; only the question does. That is
the cost of stating certainty, and the fair reading is the Calibrated Pundit below.

## The Calibrated Pundit

A **one-feature model, not a person.** Each published Scoreline is reduced to its predicted goal
margin — 3-0 and 4-1 are both +3 — and quoted the Outcome frequencies a call of that margin has
historically produced, fitted walk-forward on that Pundit's past calls only. It registers as
`margin_map_lawrenson` and `margin_map_sutton`, and each carries a `note` saying in as many words
that it is not the forecaster, because "Sutton beat the model" is a sentence no output of this
project may support ([ADR 0003](./docs/adr/0003-calibrated-pundit-predictor.md)).

```
python -m epl.pundits three-way   # model, market, floor and both readings, on shared Fixtures
python -m epl.pundits calls       # every call ranked by the miss its fair reading still had
python -m epl.pundits map         # what a call of each predicted goal margin is worth
```

| Over | Fixtures | Market Line | Elo | Calibrated Pundit | Naive Baseline | as-stated |
|---|---|---|---|---|---|---|
| Lawrenson's | 1,856 | 0.1943 | 0.2016 | **0.2127** | 0.2356 | 0.3335 |
| Sutton's | 1,472 | 0.1968 | 0.2031 | **0.2111** | 0.2322 | 0.3346 |

**The cost of stating certainty is 0.1209 and 0.1235 RPS**, published with the gap in a column of
that name. Read fairly, the same calls beat the floor they were a tenth of a point below — and
accuracy barely moves across the two readings (0.5102 → 0.5116, 0.4925 → 0.4993), so the 0.12 really
is the format of the question rather than a different set of opinions. Neither Calibrated Pundit
beats Elo; ADR 0003 anticipated that one might, and the naming rule is in the code either way.

The model column here stays Elo now that Dixon-Coles exists, and deliberately. The three-way board
is a **chosen** comparison of three named opponents rather than a view of the registry, so
registering a fifth Predictor puts it on the scoreboard and does not silently rewrite the argument
ADR 0003 is making. `epl.pundits.report.OPPONENTS` says so where the choice is made.

Nothing chooses the buckets. The margins each Pundit actually called *are* the buckets, and the one
rule is that a bucket too thin to carry a rate merges with its neighbour nearer zero — so Lawrenson
ends with `-3,-2 | -1 | 0 | 1 | 2 | 3,4` and Sutton with `-5,-4,-3,-2 | -1 | 0 | 1 | 2 | 3,4,5,6`.
Nothing enforces monotonicity either, and both come out monotone in the Home rate at every step:
+1 goes Home 42% and 48% of the time, +3 or better 83% and 81%.

Unlike the shared layer, a Calibrated Pundit is a real Predictor: its map is fitted on matches that
had already kicked off at the As-Of Instant, and its input was published before it, so it goes
through the ledger and audits like everything else. A map needs 40 past calls behind it, so the
opening 40 of each record are not covered — 1,856 of 1,896 and 1,472 of 1,512.

The shared calibration layer, which costs every other Predictor about 0.001 RPS, **gains an
as-stated Pundit about 0.09** — 0.3341 → 0.2374 and 0.3343 → 0.2473. Put the margin map in front of
it and that gain vanishes: it costs 0.0014 and 0.0015, exactly what it costs Elo and the market.
Both facts point the same way, and it is the one in [Calibration](#calibration): the layer works,
and the other Predictors were already well calibrated.

Parsing nine pages of hand-maintained HTML is not tidy, and none of it is guessed at. Names carry
annotations (`Chelsea*`, `Crystal Palace (19th May)`) which are stripped, and genuine misspellings
(`Wolverhampton Wand`) which are Alias rows. Fifteen Fixtures were postponed, re-listed and called
twice; the call published for the date the Fixture was actually played is the one that stands.
Twelve Fixtures were never listed at all, and `covers` keeps them off the ledger rather than
inventing a Prediction. Every one of those is named and re-derived in
`tests/pundits/test_over_the_corpus.py`.
