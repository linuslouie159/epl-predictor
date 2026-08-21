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

The remaining five modules from the README layout exist as documented shells, each naming the issue
that builds it.

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

**Issue #7 — split prediction ledger and the Naive Baseline scored end to end.** It is the only
substantial ticket whose blockers are all closed, and everything else is behind it: #8, #9 and #11
name it directly, and #10 and #12–#17 sit behind those.

Take the tracker's dependency graph over the README's build order where they disagree, and they do
disagree here. The README lists Elo third and never names the ledger as a stage, but #9 is blocked
by #7 — the ledger ticket is the tracer bullet that puts one Predictor through the whole pipeline
and out the other side scored, so that Elo, the benchmarks and the Pundits land on something that
already works end to end. Check the graph before starting:

```
gh issue view <n> | sed -n '/## Blocked by/,$p'
```

Also ready: **issue #18**, the deferred-v2 stubs (XGBoost, Golden Boot, API-Football). It has been
unblocked since stage 1, needs nothing from #7, and is small — pick it up when a stage lands and
there is no appetite to start the next one.

```
conda env create -f environment.yml
conda activate epl-predictor
python -m epl.ingest fetch     # fill data/raw/ — 104 files, 26 Seasons x 4 tiers
python -m epl.ingest build     # write matches.csv (52,672) + odds_availability.csv
pytest                         # add --run-network to also hit football-data.co.uk
```
