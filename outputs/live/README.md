# Sealed Predictions

One file per Prediction Round, written before that round's first kickoff and committed.

**Nothing in this directory is ever rewritten.** Git history is the proof of when each Prediction
existed, which is the only thing that makes the live track record evidence rather than a claim about
what the code would produce today. Correcting a genuine bug in a sealed round means adding a
superseding row under a new As-Of Instant, never editing history.

See [ADR 0005](../../docs/adr/0005-split-prediction-ledger.md). Regenerable Backtest Predictions live
in `outputs/backtest/`, which is gitignored.

Empty until stage 8 (the live loop).
