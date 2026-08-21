"""The canonical Club table and Alias resolution.

Two CSVs next to this module are the authority:

``clubs.csv``
    One row per Club: a canonical ``slug`` stable across Seasons, tiers and sources, and the
    display ``name`` to show a human.

``aliases.csv``
    One row per (source, spelling): how ``source`` writes that Club. Football-Data alone spells
    Manchester United as ``Man United``, MyFootballFacts as ``Manchester United`` and Understat as
    ``Manchester United``; none of that belongs in model code.

The resolver **raises on an unknown Alias** rather than passing the spelling through. A Club whose
name silently fails to map would have its rating history split in two at the point the spelling
changed, and pyramid-wide Elo (ADR 0004) would carry that split across a promotion without anything
looking wrong.

One judgement call is recorded in the table rather than in code. Football-Data names three Clubs
along the Wimbledon lineage: ``Wimbledon`` (2000/01-2003/04), ``Milton Keynes Dons`` (2004/05
onward) and ``AFC Wimbledon`` (2011/12 onward). Milton Keynes Dons is the legal continuation of
Wimbledon FC, so merging the two would give it an earned rating rather than a cold start. They are
nonetheless kept as three separate Clubs: the identity is contested, and neither of the successors
has ever played in the Premier League, so the merge would buy nothing inside the Evaluation Window
while asserting something the data does not settle.
"""

from __future__ import annotations

import csv
import json
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent


class AliasTableError(Exception):
    """The Club or Alias table is internally inconsistent."""


class UnknownAliasError(KeyError):
    """A source spelled a Club in a way the Alias table does not know."""

    def __init__(self, alias: str, source: str) -> None:
        self.alias = alias
        self.source = source
        super().__init__(
            f"unknown Club alias {alias!r} for source {source!r}. Add it to "
            f"{aliases_path().name} rather than letting it through: an unmapped spelling splits "
            "one Club's rating history in two."
        )


@dataclass(frozen=True, slots=True)
class Club:
    """One Club, as the rest of the system refers to it."""

    slug: str
    name: str


def clubs_path() -> Path:
    """The canonical Club table."""
    return _DATA_DIR / "clubs.csv"


def aliases_path() -> Path:
    """The Alias table: one row per source spelling."""
    return _DATA_DIR / "aliases.csv"


def load_clubs() -> pd.DataFrame:
    """The canonical Club table as ``slug, name``."""
    return pd.read_csv(clubs_path(), dtype=str, keep_default_na=False)


def load_aliases() -> pd.DataFrame:
    """The Alias table as ``source, alias, slug``."""
    return pd.read_csv(aliases_path(), dtype=str, keep_default_na=False)


def normalise_alias(alias: str) -> str:
    """Fold away differences that cannot distinguish two Clubs: case, spacing, accents.

    Deliberately conservative. Anything beyond this - dropping ``FC``, stripping ``United`` -
    could collapse two genuinely different Clubs, so it is left to an explicit Alias row.
    """
    folded = unicodedata.normalize("NFKD", alias)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return " ".join(folded.split()).casefold()


class ClubResolver:
    """Resolves one source's spelling of a Club to its canonical slug."""

    def __init__(self, clubs: pd.DataFrame, aliases: pd.DataFrame) -> None:
        self._clubs = {row.slug: Club(row.slug, row.name) for row in clubs.itertuples()}
        self._aliases = aliases.copy()
        self._by_source: dict[str, dict[str, str]] = {}
        self._validate(clubs, aliases)

        for row in aliases.itertuples():
            self._by_source.setdefault(row.source, {})[normalise_alias(row.alias)] = row.slug

    @classmethod
    def load(cls) -> ClubResolver:
        """Load the tables shipped with the package."""
        return _cached_resolver()

    @property
    def clubs(self) -> dict[str, Club]:
        """Every known Club, keyed by slug."""
        return dict(self._clubs)

    def sources(self) -> list[str]:
        """Every source that has Aliases registered."""
        return sorted(self._by_source)

    def resolve(self, alias: str, source: str) -> str:
        """One spelling to one slug. Raises :class:`UnknownAliasError` if it is not in the table."""
        table = self._by_source.get(source)
        if table is None:
            raise UnknownAliasError(alias, source)
        try:
            return table[normalise_alias(alias)]
        except KeyError:
            raise UnknownAliasError(alias, source) from None

    def resolve_series(self, values: pd.Series, source: str) -> pd.Series:
        """Resolve a whole column, reporting *every* unknown spelling at once.

        One name at a time would mean one round trip through the ingest for each new Club in a new
        Season, which is how an Alias table stops being maintained.
        """
        table = self._by_source.get(source, {})
        keys = values.astype("string").map(normalise_alias, na_action="ignore")
        resolved = keys.map(table)

        unknown = values[resolved.isna() & values.notna()]
        if not unknown.empty:
            missing = sorted(set(unknown.astype(str)))
            raise UnknownAliasError(", ".join(missing), source)

        return resolved.astype("string")

    def knows(self, alias: str, source: str) -> bool:
        """Whether this spelling resolves, without raising."""
        return normalise_alias(alias) in self._by_source.get(source, {})

    def export_teamname_replacements(self, path: Path | str, source: str) -> Path:
        """Write soccerdata's ``teamname_replacements.json`` for one source.

        soccerdata is used only for Understat and FBref (decision 5), and it expects a mapping of
        canonical name to the spellings that should be replaced by it. Exporting rather than
        hand-maintaining that file keeps one authority for Aliases.
        """
        path = Path(path)
        if source not in self._by_source:
            raise AliasTableError(f"no aliases registered for source {source!r}")

        replacements: dict[str, list[str]] = {}
        for row in self._aliases.itertuples():
            if row.source != source:
                continue
            canonical = self._clubs[row.slug].name
            if row.alias != canonical:
                replacements.setdefault(canonical, []).append(row.alias)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({k: sorted(v) for k, v in sorted(replacements.items())}, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def _validate(self, clubs: pd.DataFrame, aliases: pd.DataFrame) -> None:
        expected_club_columns = ["slug", "name"]
        if list(clubs.columns) != expected_club_columns:
            raise AliasTableError(
                f"{clubs_path().name} columns are {list(clubs.columns)}, "
                f"expected {expected_club_columns}"
            )
        expected_alias_columns = ["source", "alias", "slug"]
        if list(aliases.columns) != expected_alias_columns:
            raise AliasTableError(
                f"{aliases_path().name} columns are {list(aliases.columns)}, "
                f"expected {expected_alias_columns}"
            )

        duplicate_slugs = clubs["slug"][clubs["slug"].duplicated()].tolist()
        if duplicate_slugs:
            raise AliasTableError(f"duplicate Club slugs: {sorted(set(duplicate_slugs))}")

        orphans = sorted(set(aliases["slug"]) - set(self._clubs))
        if orphans:
            raise AliasTableError(f"aliases point at slugs with no Club row: {orphans}")

        keys = aliases.assign(key=aliases["alias"].map(normalise_alias))
        collisions = keys.groupby(["source", "key"])["slug"].nunique()
        ambiguous = collisions[collisions > 1]
        if not ambiguous.empty:
            raise AliasTableError(
                f"one spelling maps to several Clubs: {sorted(ambiguous.index.tolist())}"
            )


@lru_cache(maxsize=1)
def _cached_resolver() -> ClubResolver:
    return ClubResolver(load_clubs(), load_aliases())


def write_clubs(clubs: list[Club], path: Path | str | None = None) -> Path:
    """Rewrite the canonical Club table. Used by the table-building tooling, not by the pipeline."""
    path = Path(path) if path is not None else clubs_path()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["slug", "name"])
        for club in sorted(clubs, key=lambda c: c.slug):
            writer.writerow([club.slug, club.name])
    return path


def write_aliases(rows: list[tuple[str, str, str]], path: Path | str | None = None) -> Path:
    """Rewrite the Alias table. Used by the table-building tooling, not by the pipeline."""
    path = Path(path) if path is not None else aliases_path()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "alias", "slug"])
        for row in sorted(set(rows)):
            writer.writerow(list(row))
    return path
