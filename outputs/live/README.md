# Sealed Predictions

One file per Prediction Round, written before that round's first kickoff and committed.

**Nothing in this directory is ever rewritten.** Git history is the proof of when each Prediction
existed, which is the only thing that makes the live track record evidence rather than a claim about
what the code would produce today.

`python -m epl.live seal` writes a round here, and refuses outside its window: at or after its As-Of
Instant, and strictly before its first kickoff. `python -m epl.ledger audit` re-checks what is here
against git.

A round is named after itself — `2026-08-28.csv`. Correcting a genuine bug found before kickoff means
`python -m epl.live seal --supersede`, which writes a **new revision** beside it — `2026-08-28.1.csv`
— at a new As-Of Instant, every row stamped strictly later than the one it replaces. Never new bytes
in an old file.

See [ADR 0005](../../docs/adr/0005-split-prediction-ledger.md). Regenerable Backtest Predictions live
in `outputs/backtest/`, which is gitignored.

Empty for now. The loop is built (stage 13, issue #17); Football-Data's rolling `fixtures.csv` has
not yet been seen carrying a Premier League Fixture to seal. `python -m epl.live round` reports what
it holds today, and writes nothing.
