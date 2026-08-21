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
| Naive Baseline | 0.2292 | the floor — beat this or the model has no value |
| **Target** | **≤ 0.1986** | market + 0.005 — this is success |
| Market Line | 0.1936 | the opponent |
| Ceiling Line | — | reference only; knows team news we don't |

The system does **not** need to beat the market. It needs to be leak-free, well calibrated, and
within a stated distance of the market while beating the Naive Baseline and the Pundits.

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
2. **Metrics** — RPS, Brier, log loss, accuracy, calibration. Unit-tested against hand-worked examples
3. **Elo** — pyramid-wide ratings → ordered logit → shared calibration layer
4. **Benchmarks** — Market Line (vig removed) and Naive Baseline
5. **Pundits** — backfill, grading, Calibrated Pundit, three-way scoreboard
6. **Dixon-Coles** — Scoreline probabilities, MLE per Prediction Round
7. **Season Projection** — Bayesian posterior → Monte Carlo → title / top-4 / relegation
8. **Live loop** — seal each Prediction Round before kickoff
9. **Frontend** — reads `outputs/` only; no modelling logic

Stages 5 and 7 were moved: the Season Projection needs goal difference (24 of 26 Seasons had points
ties), so it cannot be built on Elo alone; the Pundit tracker needs no model at all, so it comes early.

## Deferred to v2

XGBoost / ML layer, the Golden Boot player model, and the API-Football client. Each gets a written
stub explaining what it would do and what it needs. API-Football is unnecessary in v1 because
`fixtures.csv` already carries upcoming Fixtures with the Market Line, free and unauthenticated.

## Layout

```
data/raw/            cached downloads, byte-identical, never edited
data/processed/      cleaned tables
src/epl/ingest/      football-data, pundit scrapers
src/epl/clubs/       canonical Club table + Alias resolution
src/epl/metrics/     RPS, Brier, log loss, calibration
src/epl/models/      elo, ordered logit, dixon-coles, calibration layer
src/epl/benchmarks/  market line (vig removal), naive baseline
src/epl/pundits/     grading + Calibrated Pundit
src/epl/simulate/    Bayesian fit + Monte Carlo Season Projection
src/epl/ledger/      Prediction stores
outputs/backtest/    regenerable, gitignored
outputs/live/        SEALED, committed, append-only, never rewritten
```

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

## Tests

```
pytest                  # unit tests, plus corpus integrity checks when data/raw/ is populated
pytest --run-network    # also hits football-data.co.uk
```

Tests marked `cache` re-derive the measured facts in [docs/DECISIONS.md](./docs/DECISIONS.md) from
the ingested corpus rather than trusting them, and skip when `data/raw/` is absent. Tests marked
`network` check that upstream still serves the shapes the live loop assumes.
