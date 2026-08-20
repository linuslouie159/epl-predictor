# Use pre-match odds, not closing odds, as the Market Line

The forecasting literature treats vig-removed **closing** odds as the standard benchmark, so a
future reader will reasonably ask why we don't. We use the market-average **pre-match** line
(`BbAvH/D/A` 2005/06–2018/19, spliced to `AvgH/D/A` 2019/20–present) instead, because closing
odds absorb team news, late injury reports and lineup leaks from the hours before kickoff — an
information set neither our model nor the Pundits have. Scoring against them would produce a
loss that says nothing about model quality.

Football-Data samples pre-match odds Friday afternoon for weekend fixtures and Tuesday afternoon
for midweek ones, which is also when Pundits publish. Using it puts model, market and pundit on
one information set by construction, and yields a continuous 21-season series.

## Consequences

The vig-removed closing line (`AvgCH/D/A`, 2019/20 onward) is still computed and published as the
**Ceiling Line** — a reference upper bound, explicitly labelled as knowing more than we do. Any
comparison against it must carry that caveat. Seasons 2000/01–2001/02 have no odds columns at all
and therefore have no market comparison; they remain in the Training Window regardless.
