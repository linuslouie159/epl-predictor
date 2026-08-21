"""Metrics: RPS, Brier, log loss, accuracy, calibration.

Built by issue #6, and built **before any model**, because every comparison in this project rests
on it. Expected values are worked out by hand and written as literals — a metric verified against
the code that produces it only proves the code agrees with itself.

RPS is primary. Outcomes are ordinal (Draw sits between Home and Away), and RPS is the metric that
knows it: calling Home when Away happens should cost more than calling Home when Draw happens.
Accuracy is reported for lay explanation only and is never the headline (CLAUDE.md).

The anchors these functions must hit, from ADR 0006 and the spec:

* a certain Prediction wrong in the worst direction scores exactly 1.00 RPS
* wrong by one ordinal step scores 0.50
* correct scores 0.00

Every metric is emitted twice, pre-calibration and post-calibration, so a large correction reads as
a warning about the underlying model rather than a silent fix.
"""

__all__: list[str] = []
