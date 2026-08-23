"""Football-Data.co.uk ingest: E0-E3 Season results, match stats and odds.

Two things make this messier than "read a CSV", and both are era-dependent facts about the source
rather than choices of ours:

* **Column availability moves.** Match stats appear in 2000/01, market-average pre-match odds
  (``BbAv*``) in 2005/06, and in 2019/20 those columns are renamed to ``Avg*`` with market-average
  closing (``AvgC*``) appearing alongside. The two pre-match spellings are the same quantity, so
  they are spliced here into one column; deciding what to *do* with it - vig removal, the Market
  Line - belongs to the benchmarks module, not to ingest.
* **Date formats move**, between ``dd/mm/yy`` and ``dd/mm/yyyy``, sometimes across tiers within one
  Season. Both are parsed, and every parsed date is checked to fall inside its Season's plausible
  window. A silently misparsed year is exactly the kind of corruption that would surface later as a
  leak-free model performing suspiciously well.

All four English tiers are ingested, not just the Premier League. See ADR 0004.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
from pathlib import Path

import pandas as pd

from epl.clubs import ClubResolver
from epl.ingest import cache
from epl.ingest.fetcher import Fetcher, default_fetcher
from epl.paths import processed_dir, raw_dir
from epl.windows import FIRST_SEASON, LAST_SEASON, season_label

#: The four English tiers Football-Data serves with an identical schema (ADR 0004).
DIVISIONS: tuple[str, ...] = ("E0", "E1", "E2", "E3")

#: The source name under which Football-Data's spellings are registered as Aliases.
SOURCE = "football-data"

_BASE_URL = "https://www.football-data.co.uk/mmz4281"

#: Upstream is Windows-authored: newer files carry a UTF-8 BOM, older ones are cp1252.
_ENCODINGS = ("utf-8-sig", "cp1252")

#: Upstream name -> our name. Everything unlisted is dropped. The raw cache keeps the other
#: bookmakers' columns if they are ever wanted; carrying a hundred unused columns through the
#: pipeline only invites someone to model on one by accident.
_COLUMN_MAP = {
    "Date": "date",
    "Time": "time",
    "HomeTeam": "home_club",
    "AwayTeam": "away_club",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "outcome",
    "HTHG": "ht_home_goals",
    "HTAG": "ht_away_goals",
    "HTR": "ht_outcome",
    "Referee": "referee",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_target",
    "AST": "away_shots_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
}

#: Market-average **pre-match** odds - the source of the Market Line (ADR 0001). ``BbAv*`` runs
#: 2005/06-2018/19 and ``Avg*`` from 2019/20. They are the same quantity under two spellings and
#: never appear in the same file.
_PREMATCH_ODDS = {
    "prematch_odds_home": ("BbAvH", "AvgH"),
    "prematch_odds_draw": ("BbAvD", "AvgD"),
    "prematch_odds_away": ("BbAvA", "AvgA"),
}

#: Market-average **closing** odds - the source of the Ceiling Line, 2019/20 onward. Reference
#: only: it knows team news the model does not (ADR 0001).
_CLOSING_ODDS = {
    "closing_odds_home": ("AvgCH",),
    "closing_odds_draw": ("AvgCD",),
    "closing_odds_away": ("AvgCA",),
}

_INT_COLUMNS = (
    "home_goals",
    "away_goals",
    "ht_home_goals",
    "ht_away_goals",
    "home_shots",
    "away_shots",
    "home_shots_target",
    "away_shots_target",
    "home_corners",
    "away_corners",
    "home_fouls",
    "away_fouls",
    "home_yellows",
    "away_yellows",
    "home_reds",
    "away_reds",
)

_FLOAT_COLUMNS = tuple(_PREMATCH_ODDS) + tuple(_CLOSING_ODDS)

_REQUIRED_COLUMNS = frozenset({"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"})

#: Canonical column order for a table of matches.
MATCH_COLUMNS: tuple[str, ...] = (
    "season",
    "division",
    "date",
    "time",
    "home_club",
    "away_club",
    "home_goals",
    "away_goals",
    "outcome",
    "ht_home_goals",
    "ht_away_goals",
    "ht_outcome",
    "home_shots",
    "away_shots",
    "home_shots_target",
    "away_shots_target",
    "home_corners",
    "away_corners",
    "home_fouls",
    "away_fouls",
    "home_yellows",
    "away_yellows",
    "home_reds",
    "away_reds",
    "referee",
    *_PREMATCH_ODDS,
    *_CLOSING_ODDS,
)


#: Canonical column order for the per-Season record of which odds columns exist.
#: ``prematch_average`` holds the spelling actually found - ``BbAv``, ``Avg``, or null.
ODDS_AVAILABILITY_COLUMNS: tuple[str, ...] = (
    "season",
    "season_label",
    "division",
    "bet365",
    "prematch_average",
    "closing_average",
    "has_market_line",
)


class IngestError(Exception):
    """The source data did not look the way this module requires."""


def season_code(season: int) -> str:
    """Football-Data's four-digit code for a Season, identified by its start year.

    >>> season_code(2000)
    '0001'
    >>> season_code(2025)
    '2526'
    >>> season_code(1999)
    '9900'
    """
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def season_csv_url(season: int, division: str) -> str:
    """The upstream URL for one Season of one tier."""
    _check_division(division)
    return f"{_BASE_URL}/{season_code(season)}/{division}.csv"


def raw_season_path(season: int, division: str) -> Path:
    """Where that URL's bytes are cached. The layout mirrors the URL, so provenance is obvious."""
    _check_division(division)
    return raw_dir() / SOURCE / season_code(season) / f"{division}.csv"


def fetch_season(
    season: int,
    division: str,
    *,
    refresh: bool = False,
    fetcher: Fetcher | None = None,
    timeout: float = 60.0,
) -> Path:
    """Download one Season of one tier into the raw cache, and return the cached path.

    A cached file is reused unless ``refresh`` is set. Refreshing matters for the current Season,
    whose upstream file grows weekly and backfills results and odds into rows already published -
    which is precisely why live Predictions are sealed rather than regenerated (ADR 0005).

    That same backfilling is why a refresh does not simply overwrite; :func:`epl.ingest.cache.store`
    archives the cached copy into ``superseded/`` first.
    """
    path = raw_season_path(season, division)
    if path.exists() and not refresh:
        return path

    fetcher = fetcher or default_fetcher(timeout)
    return cache.store(path, fetcher(season_csv_url(season, division)))


def superseded_dir(season: int, division: str) -> Path:
    """Where earlier copies of a Season file go when a refresh brings different bytes."""
    return cache.superseded_dir(raw_season_path(season, division))


def fetch_all(
    seasons: range | list[int] | None = None,
    divisions: tuple[str, ...] = DIVISIONS,
    *,
    refresh: bool = False,
    fetcher: Fetcher | None = None,
    timeout: float = 60.0,
) -> list[Path]:
    """Fill the raw cache for every Season and tier. Returns the cached paths in order."""
    fetcher = fetcher or default_fetcher(timeout)
    return [
        fetch_season(season, division, refresh=refresh, fetcher=fetcher)
        for season in _seasons(seasons)
        for division in divisions
    ]


def read_raw_csv(path: Path | str) -> pd.DataFrame:
    """Read one cached Football-Data CSV as it is, before any interpretation.

    Every cell comes back as a string; nothing is coerced or renamed here.

    The source is not a rectangle. Alongside the BOM on newer files and the cp1252 bytes in older
    ones, Seasons 2002/03-2004/05 carry **ragged rows** - a row may run short of its header, or
    trail extra empty fields where a bookmaker's columns were appended without the header being
    updated. Every column we care about sits at the front of the row and stays aligned, so short
    rows are padded and blank surplus is dropped. Surplus that is *not* blank is refused: that
    would mean the columns had shifted, and quietly reading odds out of the wrong field is the kind
    of bug that produces a plausible number rather than a crash.
    """
    path = Path(path)
    rows = [
        [cell.strip() for cell in row]
        for row in csv.reader(io.StringIO(_decode(path)))
        if any(cell.strip() for cell in row)
    ]
    if not rows:
        return pd.DataFrame()

    header = [name.strip() for name in rows[0]]
    keep = [i for i, name in enumerate(header) if name and not name.startswith("Unnamed:")]
    names = [header[i] for i in keep]

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise IngestError(f"{path}: duplicated column names {duplicates}")

    records = []
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) > len(header):
            surplus = row[len(header) :]
            if any(surplus):
                raise IngestError(
                    f"{path}: line {line_number} has {len(row)} fields against a "
                    f"{len(header)}-column header, and the surplus is not blank: {surplus[:5]}"
                )
            row = row[: len(header)]
        elif len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        records.append([row[i] for i in keep])

    frame = pd.DataFrame(records, columns=names, dtype="object").astype("string")
    frame = frame.mask(frame == "")
    if "Div" in frame.columns:
        frame = frame[frame["Div"].notna()]
    return frame.reset_index(drop=True)


def parse_season_csv(
    path: Path | str,
    season: int,
    division: str,
    *,
    resolver: ClubResolver | None = None,
) -> pd.DataFrame:
    """Turn one cached Season CSV into canonical match rows.

    Club names are resolved to canonical slugs. An unrecognised spelling raises rather than passing
    through, because a silently unmapped Club splits one Club's rating history in two.
    """
    _check_division(division)
    raw = read_raw_csv(path)

    require_columns(raw, set(_REQUIRED_COLUMNS), path)

    frame = pd.DataFrame(index=raw.index)
    frame["season"] = season
    frame["division"] = division

    for source_name, target_name in _COLUMN_MAP.items():
        frame[target_name] = raw[source_name] if source_name in raw.columns else pd.NA

    for target_name, candidates in {**_PREMATCH_ODDS, **_CLOSING_ODDS}.items():
        present = [c for c in candidates if c in raw.columns]
        if len(present) > 1:
            raise IngestError(
                f"{path}: {present} are both present. They are the same quantity under two "
                "spellings, so splicing them would double-count."
            )
        frame[target_name] = raw[present[0]] if present else pd.NA

    frame["date"] = _parse_season_dates(frame["date"], season, path)

    for column in _INT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in _FLOAT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")
    for column in ("time", "referee", "ht_outcome"):
        frame[column] = frame[column].astype("string")

    # Rows without an Outcome are unplayed or abandoned. A Season file occasionally carries one.
    frame = frame[frame["outcome"].isin(["H", "D", "A"])].reset_index(drop=True)
    frame["outcome"] = frame["outcome"].astype("string")

    resolver = resolver or ClubResolver.load()
    frame["home_club"] = resolver.resolve_series(frame["home_club"], source=SOURCE)
    frame["away_club"] = resolver.resolve_series(frame["away_club"], source=SOURCE)

    return frame[list(MATCH_COLUMNS)]


def load_matches(
    seasons: range | list[int] | None = None,
    divisions: tuple[str, ...] = DIVISIONS,
    *,
    resolver: ClubResolver | None = None,
) -> pd.DataFrame:
    """Every cached Season and tier as one table of matches, sorted by date.

    Reads only what is already in the raw cache; call :func:`fetch_all` first.
    """
    resolver = resolver or ClubResolver.load()

    frames = [
        parse_season_csv(path, season, division, resolver=resolver)
        for season in _seasons(seasons)
        for division in divisions
        if (path := raw_season_path(season, division)).exists()
    ]
    if not frames:
        return pd.DataFrame(columns=list(MATCH_COLUMNS))

    matches = pd.concat(frames, ignore_index=True)
    return matches.sort_values(
        ["date", "division", "home_club"], kind="stable"
    ).reset_index(drop=True)


def odds_availability_for(path: Path | str, season: int, division: str) -> pd.Series:
    """Which odds columns one cached Season file actually carries."""
    columns = set(read_raw_csv(path).columns)
    prematch = next((spelling for spelling in ("BbAv", "Avg") if f"{spelling}H" in columns), pd.NA)

    return pd.Series(
        {
            "season": season,
            "season_label": season_label(season),
            "division": division,
            "bet365": {"B365H", "B365D", "B365A"} <= columns,
            "prematch_average": prematch,
            "closing_average": {"AvgCH", "AvgCD", "AvgCA"} <= columns,
            "has_market_line": not pd.isna(prematch),
        }
    )


def odds_availability(
    seasons: range | list[int] | None = None,
    divisions: tuple[str, ...] = DIVISIONS,
) -> pd.DataFrame:
    """One row per cached Season and tier, recording which odds columns it carries.

    The point is that absence and value become indistinguishable downstream once a column is a
    float column full of nulls. A Season with no Market Line is not a Season where the market
    priced a home win at nothing — it is a Season with no market to compare against, and ADR 0001
    says those Seasons simply have no market comparison. Benchmark code reads this table rather
    than re-deriving the era boundaries from the raw files.
    """
    rows = [
        odds_availability_for(path, season, division)
        for season in _seasons(seasons)
        for division in divisions
        if (path := raw_season_path(season, division)).exists()
    ]
    if not rows:
        return pd.DataFrame(columns=list(ODDS_AVAILABILITY_COLUMNS))
    return pd.DataFrame(rows)[list(ODDS_AVAILABILITY_COLUMNS)].reset_index(drop=True)


def club_names_in_raw_cache(
    seasons: range | list[int] | None = None,
    divisions: tuple[str, ...] = DIVISIONS,
) -> list[str]:
    """Every distinct Club spelling in the raw cache, sorted.

    Used to build and to audit the Alias table: a name here that the resolver does not know is a
    hole in the table, not a row to be dropped.
    """
    names: set[str] = set()
    for season in _seasons(seasons):
        for division in divisions:
            path = raw_season_path(season, division)
            if not path.exists():
                continue
            raw = read_raw_csv(path)
            for column in ("HomeTeam", "AwayTeam"):
                if column in raw.columns:
                    names.update(raw[column].dropna().str.strip())
    names.discard("")
    return sorted(names)


def _seasons(seasons: range | list[int] | None) -> list[int]:
    if seasons is None:
        return list(range(FIRST_SEASON, LAST_SEASON + 1))
    return list(seasons)


def _check_division(division: str) -> None:
    if division not in DIVISIONS:
        raise ValueError(f"unknown division {division!r}; expected one of {DIVISIONS}")


def _decode(path: Path) -> str:
    raw = path.read_bytes()
    last_error: UnicodeDecodeError | None = None
    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise IngestError(f"{path}: not decodable as {' or '.join(_ENCODINGS)}") from last_error


def parse_dates(values: pd.Series, path: Path | str) -> pd.Series:
    """Parse Football-Data's two date spellings, ``dd/mm/yy`` and ``dd/mm/yyyy``.

    Both appear, sometimes across tiers within one Season. ``%y`` maps 00-68 to the 2000s and 69-99
    to the 1900s, which is right for every Season we ingest - but only :func:`_parse_season_dates`
    proves it, by checking the result lands inside its Season.
    """
    text = values.astype("string").str.strip()
    parsed = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
    parsed = parsed.fillna(pd.to_datetime(text, format="%d/%m/%y", errors="coerce"))

    unparsed = text.notna() & parsed.isna()
    if unparsed.any():
        raise IngestError(f"{path}: unparseable dates {sorted(set(text[unparsed]))[:5]}")
    return parsed


def require_columns(frame: pd.DataFrame, required: set[str], path: Path | str) -> None:
    """Fail loudly, naming what is absent, before any column is read by position or by guess."""
    missing = required - set(frame.columns)
    if missing:
        raise IngestError(f"{path}: missing required columns {sorted(missing)}")


def _parse_season_dates(values: pd.Series, season: int, path: Path | str) -> pd.Series:
    """Parse dates, then prove the result belongs to this Season.

    A silently misparsed year is the corruption most likely to surface later as a leak-free model
    performing suspiciously well, so the window check is what makes the two-format parse safe to
    rely on rather than merely true today.
    """
    parsed = parse_dates(values, path)

    earliest = pd.Timestamp(dt.date(season, 7, 1))
    latest = pd.Timestamp(dt.date(season + 1, 8, 31))
    outside = parsed.notna() & ((parsed < earliest) | (parsed > latest))
    if outside.any():
        raise IngestError(
            f"{path}: dates outside Season {season_label(season)} "
            f"({earliest.date()}..{latest.date()}): {sorted(set(parsed[outside].dt.date))[:5]}"
        )

    return parsed.dt.date


def match_table(path: Path | None = None) -> pd.DataFrame:
    """The cleaned match table `python -m epl.ingest build` writes, or how to make one.

    Every command that scores or fits anything starts here — the ledger's, the benchmarks' and the
    models' — so the read and the instruction that follows a missing file are stated once. Three
    copies of an eight-line helper is where one of them quietly starts reading `time` as a float
    and stops resolving kickoffs.

    Distinct from :func:`load_matches`, which builds the same table from the raw cache. This reads
    what was already built, which is what a command wants: the corpus every Predictor was walked
    over, rather than a fresh derivation of it.
    """
    source = path or processed_dir() / "matches.csv"
    if not source.exists():
        raise SystemExit(
            f"{source} does not exist. Build it first:\n"
            "    python -m epl.ingest fetch\n"
            "    python -m epl.ingest build"
        )
    return pd.read_csv(source, dtype={"time": "string"})
