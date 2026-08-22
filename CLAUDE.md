# EPL Predictor — read this first

A calibrated Premier League forecasting system: match Outcome probabilities, a simulated final table,
and an honest three-way scoreboard against the betting market and public pundits.

**Before writing any code, read these in order:**

1. [CONTEXT.md](./CONTEXT.md) — the glossary. Use these exact terms in code and in conversation.
2. [docs/DECISIONS.md](./docs/DECISIONS.md) — every decision, the measured evidence behind it, and the open risks.
3. [docs/adr/](./docs/adr/) — nine ADRs explaining the choices that look wrong until you know why.
4. [README.md](./README.md) — target numbers, data sources, build order, layout.

## The rule that outranks everything

**No future data, ever.** A Prediction may use only data timestamped strictly before its As-Of
Instant. If a change would make the model faster, simpler, or more accurate but blurs that line,
the answer is no.

## Things that look like bugs but are deliberate

Do not "fix" these without reading the linked ADR first:

- **Lower-league results are ingested** into an EPL predictor — E0 through E3. [ADR 0004](./docs/adr/0004-rate-the-whole-pyramid.md)
- **The benchmark is pre-match odds, not closing odds**, against convention. [ADR 0001](./docs/adr/0001-pre-match-odds-as-market-benchmark.md)
- **The model predicts weekly in batches** instead of per-fixture, deliberately using less information. [ADR 0002](./docs/adr/0002-weekly-prediction-rounds.md)
- **26 seasons are ingested but only 21 are scored.** [ADR 0008](./docs/adr/0008-burn-in-prefix-frozen-hyperparameters.md)
- **Dixon-Coles is fitted two different ways** — MLE and Bayesian. [ADR 0007](./docs/adr/0007-mle-for-matches-bayesian-for-projections.md)
- **The pundit's published scoreline is transformed before scoring.** [ADR 0003](./docs/adr/0003-calibrated-pundit-predictor.md)
- **Two prediction stores exist, not one**, and one of them must never be rewritten. [ADR 0005](./docs/adr/0005-split-prediction-ledger.md)
- **The Ceiling Line scores *worse* than the Market Line on the scoreboard** — 0.1968 against
  0.1936 — and this is not evidence that closing odds are worse. The two are measured over
  different Fixtures; on the ones they share the Ceiling Line wins by 0.0013 RPS. Its `note` says
  so, and must keep saying so. [ADR 0001](./docs/adr/0001-pre-match-odds-as-market-benchmark.md)
- **The Market Line's stored rows say `inputs_seen = 0`** and carry no `latest_input`. Correct: it
  reads its odds off the Fixture, and a Predictor that consumes no history has no history to leak.
  Do not make it touch the corpus. A Pundit will record the same.
- **Refreshing a cached raw file archives the old bytes** into `superseded/` instead of replacing
  them, and **`Wimbledon`, `Milton Keynes Dons` and `AFC Wimbledon` are three separate Clubs**.
  Both are explained where they live — `src/epl/ingest/football_data.py` and `src/epl/clubs/table.py`.

## Never do this

- Never edit anything under `data/raw/` — it is a byte-identical cache of upstream files.
- Never rewrite a file under `outputs/live/` after its round's first kickoff. Supersede with a new
  As-Of Instant instead. That directory is evidence, not output.
- Never tune a hyperparameter using data from outside the Burn-In Window (2000/01–2004/05).
- Never report accuracy as the headline metric. RPS is primary; accuracy is for lay explanation only.

## Status

Design is complete and grilled.

**Stage 1 is built**: the Miniforge environment (`environment.yml`), the Football-Data ingester
(`src/epl/ingest/`) and the Club/Alias table (`src/epl/clubs/`, 115 Clubs across four tiers).

**Stage 2 is built**: Prediction Rounds (`src/epl/rounds.py`, issue #5) and the metrics module
(`src/epl/metrics/`, issue #6) — RPS, Brier, log loss, accuracy and the 10-bin reliability diagram,
every expected value worked by hand before any model uses it.

**Stage 3 is built**: the Predictor contract (`src/epl/predictors.py`), the split ledger
(`src/epl/ledger/`) and the Naive Baseline (`src/epl/benchmarks/naive.py`), issue #7 — one Predictor
walked over the whole Evaluation Window, stored, audited and scored at **0.22938 RPS** over 7,980
Fixtures and 952 Prediction Rounds.

**Stage 4 is built**: the vig removal (`src/epl/benchmarks/vig.py`), the Market Line and the Ceiling
Line (`src/epl/benchmarks/market.py`), issue #8. The scoreboard now reads:

```
     predictor  fixtures    rps  brier  log_loss  accuracy
   market_line      7980 0.1936 0.5684    0.9582    0.5471
  ceiling_line      2660 0.1968 0.5717    0.9639    0.5498
naive_baseline      7980 0.2294 0.6430    1.0642    0.4556
```

`epl.models`, `epl.pundits` and `epl.simulate` are still documented shells, each naming the issue
that builds it.

Four things about stage 4 worth knowing before building on it:

- **The Ceiling Line's 0.1968 is not worse than the Market Line's 0.1936.** They are measured over
  different Fixtures. On the 2,660 they share, the Market Line scores 0.1981 and the Ceiling Line
  beats it by 0.0013 RPS. That caveat rides onto the scoreboard as the Ceiling Line's `note` and
  must not be dropped — the bare number reads as the opposite of what it means.
- **The Predictor contract grew three optional attributes**, documented above the accessors in
  `src/epl/predictors.py` and read through `predictors.also_sees`, `predictors.note` and
  `schema.covered`. None is on the Protocol (a Protocol member is required of everything claiming
  it, and almost no Predictor wants these):
  `covers(fixtures)` says which Fixtures a Predictor can speak to at all; `also_sees` claims extra
  Fixture columns, checked against `schema.PRIVILEGED_FIXTURE_COLUMNS`; `note` is a caveat the
  scoreboard prints. **Issue #11's Pundits will want the first and the third.**
- **A Predictor that covers nothing writes no file**, and `backfill` says so out loud rather than
  passing over it. That line is also what a broken input column looks like, so read it.
- **Four books in the corpus have an overround below one** — 2025/26 League One closing averages,
  as low as 0.955. `vig.is_book` excludes them; no Premier League Fixture and no pre-match book is
  affected. Do not "fix" this by loosening `as_book`: a book that pays out more than it takes is
  not a book, and that check is what would catch a genuinely shifted column.

Three things about stage 3 worth knowing before building on it:

- **A Predictor is handed `Evidence`, not an As-Of Instant.** Evidence is the corpus already cut at
  the instant, and it records what it hands over, so `inputs_seen` and `latest_input` on every
  stored row are a receipt. Do not add a Predictor that reads the corpus directly — that is the
  one way a leak can enter without an audit failing.
- **The scoreboard has no branch per Predictor and must not grow one.** Registering a Predictor is
  what puts it on the board; the ledger, the audits and the scoring are written once against the
  contract (spec, user story 16).
- **The row audit is two-tier on kickoff**, matching the As-Of rule: strictly-before where a
  kickoff time was recorded, and at-or-before where it was not. 313 Evaluation Window Fixtures are
  played on the very Tuesday or Friday they anchor to and are legitimately equal.

Two things about stage 2 worth knowing before building on it:

- **A Prediction Round's As-Of Instant is midnight at the start of its anchor day**, not the
  afternoon the market samples at. Nothing is played in between, so midnight withholds no data the
  market had — and it is the only choice checkable for the 437 Fixtures that kick off on their own
  anchor day with no recorded kickoff time. Football-Data records no kickoff time at all before
  2019/20 (7,220 of 9,880 Premier League Fixtures).
- **There are 1,189 Prediction Rounds, not the 1,332 the design recorded.** The anchor rule is
  stated as executable code and was kept; the unreproducible count was corrected. See
  [docs/DECISIONS.md](./docs/DECISIONS.md).

## What to build next

**Issue #9 — pyramid-wide Elo.** It is what the README's build order lists first, it is the first
real model, and #8 put a meaningful opponent beside it: an Elo that lands between 0.2294 and 0.1936
is doing something, and one that does not beat 0.2294 has no value at all. Unblocked since #7.

Two things about the contract that will come up in its first hour:

- **Elo reads the whole pyramid.** `evidence.matches()` with no `divisions` is the call — a Club
  promoted into the Premier League must arrive with a rating it earned (ADR 0004). The Naive
  Baseline narrows to the tiers being predicted, and that is the exception rather than the pattern.
- **Hyperparameters are fitted in the Burn-In Window and frozen** (ADR 0008). `backfill` walks the
  Evaluation Window while every round still sees everything before it, so the first scored round
  already has five warmed-up Seasons behind it without one of them being scored.

**Issue #11 — the Pundits** is also unblocked and is the other half of the three-way scoreboard.
It will want `covers` and `note` from the contract stage 4 added: a Pundit published in the Seasons
they worked and no others, and a Pundit scored as-stated needs its caveat travelling with it.

Also ready: **issue #18**, the deferred-v2 stubs (XGBoost, Golden Boot, API-Football). It has been
unblocked since stage 1, needs nothing from the ledger, and is small — pick it up when a stage
lands and there is no appetite to start the next one.

Check the graph before starting:

```
gh issue view <n> | sed -n '/## Blocked by/,$p'
```

```
conda env create -f environment.yml
conda activate epl-predictor
python -m epl.ingest fetch     # fill data/raw/ — 104 files, 26 Seasons x 4 tiers
python -m epl.ingest build     # write matches.csv (52,672) + odds_availability.csv
python -m epl.ledger backfill  # walk every registered Predictor over the Evaluation Window
python -m epl.ledger scoreboard
python -m epl.ledger audit     # re-check both stores and the seal on outputs/live/
python -m epl.benchmarks overround   # the margin in each book, per Season and tier
python -m epl.benchmarks methods     # the three vig removals compared on one book
pytest                         # add --run-network to also hit football-data.co.uk
```
