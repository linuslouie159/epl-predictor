"""The Club table and Alias resolution.

The invariants here are the ones that keep a Club's rating history in one piece across 26 Seasons,
four tiers and several sources.
"""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from epl.clubs import (
    AliasTableError,
    ClubResolver,
    UnknownAliasError,
    load_aliases,
    load_clubs,
)
from epl.clubs.table import normalise_alias

FOOTBALL_DATA = "football-data"


def _resolver(clubs: list[tuple[str, str]], aliases: list[tuple[str, str, str]]) -> ClubResolver:
    return ClubResolver(
        pd.DataFrame(clubs, columns=["slug", "name"]),
        pd.DataFrame(aliases, columns=["source", "alias", "slug"]),
    )


class TestResolve:
    def test_maps_a_source_spelling_to_a_slug(self) -> None:
        resolver = _resolver(
            [("man_united", "Manchester United")],
            [(FOOTBALL_DATA, "Man United", "man_united")],
        )
        assert resolver.resolve("Man United", FOOTBALL_DATA) == "man_united"

    def test_ignores_case_and_stray_whitespace(self) -> None:
        resolver = _resolver(
            [("man_united", "Manchester United")],
            [(FOOTBALL_DATA, "Man United", "man_united")],
        )
        assert resolver.resolve("  man   UNITED ", FOOTBALL_DATA) == "man_united"

    def test_ignores_accents(self) -> None:
        resolver = _resolver([("alaves", "Alaves")], [("x", "Alavés", "alaves")])
        assert resolver.resolve("Alaves", "x") == "alaves"

    def test_two_sources_may_spell_one_club_differently(self) -> None:
        resolver = _resolver(
            [("man_united", "Manchester United")],
            [
                (FOOTBALL_DATA, "Man United", "man_united"),
                ("bbc", "Manchester Utd", "man_united"),
            ],
        )
        assert resolver.resolve("Man United", FOOTBALL_DATA) == "man_united"
        assert resolver.resolve("Manchester Utd", "bbc") == "man_united"

    def test_an_alias_does_not_leak_across_sources(self) -> None:
        resolver = _resolver(
            [("man_united", "Manchester United")],
            [("bbc", "Manchester Utd", "man_united")],
        )
        with pytest.raises(UnknownAliasError):
            resolver.resolve("Manchester Utd", FOOTBALL_DATA)

    def test_unknown_spelling_raises_rather_than_passing_through(self) -> None:
        resolver = _resolver([("arsenal", "Arsenal")], [(FOOTBALL_DATA, "Arsenal", "arsenal")])
        with pytest.raises(UnknownAliasError, match="Woking"):
            resolver.resolve("Woking", FOOTBALL_DATA)

    def test_unknown_source_raises(self) -> None:
        resolver = _resolver([("arsenal", "Arsenal")], [(FOOTBALL_DATA, "Arsenal", "arsenal")])
        with pytest.raises(UnknownAliasError):
            resolver.resolve("Arsenal", "some-new-scraper")

    def test_knows_answers_without_raising(self) -> None:
        resolver = _resolver([("arsenal", "Arsenal")], [(FOOTBALL_DATA, "Arsenal", "arsenal")])
        assert resolver.knows("Arsenal", FOOTBALL_DATA)
        assert not resolver.knows("Woking", FOOTBALL_DATA)


class TestResolveSeries:
    def test_resolves_a_whole_column(self) -> None:
        resolver = _resolver(
            [("arsenal", "Arsenal"), ("chelsea", "Chelsea")],
            [(FOOTBALL_DATA, "Arsenal", "arsenal"), (FOOTBALL_DATA, "Chelsea", "chelsea")],
        )
        resolved = resolver.resolve_series(pd.Series(["Arsenal", "Chelsea"]), FOOTBALL_DATA)
        assert list(resolved) == ["arsenal", "chelsea"]

    def test_reports_every_unknown_spelling_at_once(self) -> None:
        resolver = _resolver([("arsenal", "Arsenal")], [(FOOTBALL_DATA, "Arsenal", "arsenal")])
        with pytest.raises(UnknownAliasError) as excinfo:
            resolver.resolve_series(
                pd.Series(["Arsenal", "Woking", "Yeovil", "Woking"]), FOOTBALL_DATA
            )
        message = str(excinfo.value)
        assert "Woking" in message and "Yeovil" in message


class TestTableIntegrity:
    def test_duplicate_slugs_are_rejected(self) -> None:
        with pytest.raises(AliasTableError, match="duplicate"):
            _resolver([("arsenal", "Arsenal"), ("arsenal", "Arsenal FC")], [])

    def test_an_alias_pointing_at_no_club_is_rejected(self) -> None:
        with pytest.raises(AliasTableError, match="no Club row"):
            _resolver([("arsenal", "Arsenal")], [(FOOTBALL_DATA, "Woking", "woking")])

    def test_one_spelling_cannot_mean_two_clubs_in_one_source(self) -> None:
        with pytest.raises(AliasTableError, match="several Clubs"):
            _resolver(
                [("bristol_city", "Bristol City"), ("bristol_rovers", "Bristol Rovers")],
                [
                    (FOOTBALL_DATA, "Bristol", "bristol_city"),
                    (FOOTBALL_DATA, "bristol", "bristol_rovers"),
                ],
            )


class TestShippedTable:
    """The tables committed alongside the code, which the whole pipeline depends on."""

    def test_loads_and_validates(self) -> None:
        assert ClubResolver.load().clubs

    def test_slugs_are_lower_snake_case(self) -> None:
        bad = [s for s in load_clubs()["slug"] if not re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)*", s)]
        assert bad == []

    def test_every_club_has_at_least_one_alias(self) -> None:
        clubs = set(load_clubs()["slug"])
        aliased = set(load_aliases()["slug"])
        assert sorted(clubs - aliased) == []

    def test_display_names_are_unique(self) -> None:
        names = load_clubs()["name"]
        assert names[names.duplicated()].tolist() == []

    def test_football_data_is_a_registered_source(self) -> None:
        assert FOOTBALL_DATA in ClubResolver.load().sources()

    def test_covers_the_premier_league_regulars(self) -> None:
        resolver = ClubResolver.load()
        for alias, slug in [
            ("Man United", "man_united"),
            ("Man City", "man_city"),
            ("Arsenal", "arsenal"),
            ("Liverpool", "liverpool"),
            ("Tottenham", "tottenham"),
            ("Nott'm Forest", "nottm_forest"),
        ]:
            assert resolver.resolve(alias, FOOTBALL_DATA) == slug


class TestExport:
    def test_writes_teamname_replacements(self, tmp_path) -> None:
        resolver = _resolver(
            [("man_united", "Manchester United")],
            [(FOOTBALL_DATA, "Man United", "man_united")],
        )
        path = resolver.export_teamname_replacements(
            tmp_path / "teamname_replacements.json", FOOTBALL_DATA
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload == {"Manchester United": ["Man United"]}

    def test_omits_a_spelling_that_already_matches_the_canonical_name(self, tmp_path) -> None:
        resolver = _resolver([("arsenal", "Arsenal")], [(FOOTBALL_DATA, "Arsenal", "arsenal")])
        path = resolver.export_teamname_replacements(tmp_path / "out.json", FOOTBALL_DATA)
        assert json.loads(path.read_text(encoding="utf-8")) == {}

    def test_exports_the_shipped_table(self, tmp_path) -> None:
        resolver = ClubResolver.load()
        path = resolver.export_teamname_replacements(tmp_path / "out.json", FOOTBALL_DATA)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["Manchester United"] == ["Man United"]
        assert payload["Sheffield Wednesday"] == ["Sheffield Wed", "Sheffield Weds"]

    def test_refuses_a_source_with_no_aliases(self, tmp_path) -> None:
        resolver = ClubResolver.load()
        with pytest.raises(AliasTableError, match="no aliases"):
            resolver.export_teamname_replacements(tmp_path / "out.json", "nonexistent-source")


class TestNormalise:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Man United", "man united"),
            ("  Man   United  ", "man united"),
            ("MAN UNITED", "man united"),
            ("Alavés", "alaves"),
        ],
    )
    def test_folds_only_what_cannot_distinguish_two_clubs(self, raw: str, expected: str) -> None:
        assert normalise_alias(raw) == expected

    def test_keeps_clubs_that_differ_only_by_suffix_apart(self) -> None:
        assert normalise_alias("Bristol City") != normalise_alias("Bristol Rovers")
