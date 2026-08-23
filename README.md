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
| Market Line | 0.1936 | the opponent |
| Ceiling Line | 0.1968* | reference only; knows team news we don't |

\* over its own 2,660 Fixtures from 2019/20, not the 7,980 above — the two numbers are not
comparable. On the Fixtures they share, the Market Line scores 0.1981 and the Ceiling Line beats it
by 0.0013 RPS, which is what a few hours of team news is worth.

The system does **not** need to beat the market. It needs to be leak-free, well calibrated, and
within a stated distance of the market while beating the Naive Baseline and the Pundits.

Elo alone takes **0.030 of the 0.036 RPS** the market takes out of the floor — 84% of the available
edge from ratings and nothing else. The 0.0008 it still sits above the target now rests on
Dixon-Coles alone: the shared calibration layer is built and measured, and it makes every Predictor
on this board slightly *worse* (see [Calibration](#calibration)).

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
| Football-Data.co.uk `fixtures.csv` | upcoming Fixtures + Market Line | rolling ~1 week |
| MyFootballFacts | Pundit backfill (Lawrenson, Sutton) | 2017/18–2025/26, ~3,420 rows |
| BBC Sport | live Pundit predictions | current Season |

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
8. **Live loop** — seal each Prediction Round before kickoff
9. **Frontend** — reads `outputs/` only; no modelling logic

Stages 5 and 7 were moved: the Season Projection needs goal difference (24 of 26 Seasons had points
ties), so it cannot be built on Elo alone; the Pundit tracker needs no model at all, so it comes early.

The list above is the shape of the system, not the order the tickets unblock in. The **Prediction
ledger and the Naive Baseline (#7) come before Elo**, not after the benchmarks: one Predictor has to
go through the whole pipeline and out the other side scored before a real model is worth building on
top of it. The tracker's `Blocked by` fields are authoritative where the two disagree.

## Deferred to v2

XGBoost / ML layer, the Golden Boot player model, and the API-Football client. Each gets a written
stub explaining what it would do and what it needs. API-Football is unnecessary in v1 because
`fixtures.csv` already carries upcoming Fixtures with the Market Line, free and unauthenticated.

## Layout

```
data/raw/               cached downloads, byte-identical, never edited
data/processed/         cleaned tables
src/epl/windows.py      Season identity, Burn-In and Evaluation Windows
src/epl/rounds.py       the As-Of Instant rule and Prediction Rounds
src/epl/predictors.py   the Predictor contract, the Evidence a Predictor sees, the registry
src/epl/calibration.py  the shared isotonic layer every Predictor's output passes through
src/epl/ingest/         football-data, pundit scrapers
src/epl/clubs/          canonical Club table + Alias resolution
src/epl/metrics/        RPS, Brier, log loss, calibration measured
src/epl/models/         elo.py, ordered_logit.py, burn_in.py; dixon-coles
src/epl/benchmarks/     market line + ceiling line (vig.py), naive baseline
outputs/overround.csv   the margin in each book per Season; regenerable, gitignored
src/epl/pundits/        grading + Calibrated Pundit
src/epl/simulate/       Bayesian fit + Monte Carlo Season Projection
src/epl/ledger/         Prediction stores, the row audit, the scoreboard
outputs/backtest/       regenerable, gitignored — one file per Predictor
outputs/live/           SEALED, committed, append-only — one file per Prediction Round
outputs/scoreboard.csv  every Predictor over the Evaluation Window; regenerable, gitignored
outputs/reliability.csv 10-bin diagrams per Predictor, raw and calibrated; regenerable, gitignored
```

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

## Running the ingest

```
python -m epl.ingest fetch              # fill data/raw/ — 104 files, 26 Seasons x 4 tiers, ~16 MB
python -m epl.ingest fetch --refresh    # re-download, including the growing current Season
python -m epl.ingest build              # write matches.csv (52,672) + odds_availability.csv
python -m epl.ingest fixtures           # fetch the rolling forward-Fixture file + Market Line
python -m epl.ingest clubs              # audit: Club spellings the Alias table does not know
python -m epl.clubs.build               # rebuild clubs.csv, aliases.csv, teamname_replacements.json
```

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
pytest --run-network    # also hits football-data.co.uk
```

Tests marked `cache` re-derive the measured facts in [docs/DECISIONS.md](./docs/DECISIONS.md) from
the ingested corpus rather than trusting them, and skip when `data/raw/` is absent. Tests marked
`network` check that upstream still serves the shapes the live loop assumes.

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

## Calibration

One shared isotonic step wraps every Predictor identically — Elo, both market lines, the Naive
Baseline and, later, the Pundits. A Predictor gets it by being registered; there is no calibration
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
