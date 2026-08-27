"""The live loop: seal a Prediction Round before kickoff, and score it after (issue #17).

Everything before this stage looked backwards. A Backtest Prediction is regenerable, so its value
is in the aggregate and no single row of it is worth anything (ADR 0005). This package produces the
other kind: a Sealed Prediction, written before its Fixture kicked off and committed, so that git
history rather than a timestamp inside a file is the proof of when it existed.

The loop is three commands and they run at different moments:

===============================  ==============================================================
``python -m epl.live upcoming``  what the rolling fixtures file holds, and what could be sealed
``python -m epl.live seal``      predict the upcoming round, write it, commit it
``python -m epl.live score``     ingest results, then score what was sealed
===============================  ==============================================================

**Three "live" modules exist and they are different things.** :mod:`epl.ledger.live` is the sealed
*store* — where a round is written and how a rewrite is detected. :mod:`epl.pundits.live` is stage
12's spike, which measured how far behind the football the Pundit archive runs. This package is the
loop that uses both.

Three constraints come from earlier stages and none is optional.

**A Pundit cannot be sealed.** Stage 12 measured the only source this project is permitted to read
and found it transcribes a matchday *after* it is played (:mod:`epl.pundits.live`). So the live loop
seals the models and the Market Line, and a Pundit's column on the three-way board is filled in
retrospectively. Nothing here special-cases them: five of the nine registered Predictors declare
they cover no upcoming Fixture — two Pundits, two Calibrated Pundits, and the Ceiling Line, whose
closing odds do not exist yet — and :func:`epl.ledger.schema.covered` is where every one of those
answers is read. The scoreboard has no branch per Predictor and neither does this.

**The Live Season is scored on its own board.** It is ingested, so a Predictor can see it and
a sealed Prediction can be joined to a result — but it is never backfilled and never folded into the
Evaluation Window (:mod:`epl.windows`). ``python -m epl.ledger scoreboard`` therefore says exactly
what it said last week, and ``python -m epl.live score`` is where the live record accumulates.

**A sealed round is corrected by superseding it, never by rewriting it.** ``seal --supersede``
writes a new revision of the round's file at a new As-Of Instant, and
:func:`epl.ledger.live.seal_violations` is what catches the alternative. A superseding Prediction
genuinely knows more than the one it replaces, because it was genuinely made later; recording it at
the round's original instant would be the fiction ADR 0005 exists to prevent.
"""

from epl.live.seal import INSTANT_RESOLUTION, Sealed, run, sealed_predictions
from epl.live.upcoming import (
    CAMPAIGN_FIXTURES,
    KICKED_OFF,
    NOT_OPEN,
    ROUND_STATUS_COLUMNS,
    SEALABLE,
    LiveError,
    NothingToSeal,
    PredictionRound,
    next_round,
    rounds,
    to_predict,
)

__all__ = [
    "CAMPAIGN_FIXTURES",
    "INSTANT_RESOLUTION",
    "KICKED_OFF",
    "NOT_OPEN",
    "ROUND_STATUS_COLUMNS",
    "SEALABLE",
    "LiveError",
    "NothingToSeal",
    "PredictionRound",
    "Sealed",
    "next_round",
    "rounds",
    "run",
    "sealed_predictions",
    "to_predict",
]
