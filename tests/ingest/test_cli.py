"""The ingest command line, which is how the raw cache is actually filled."""

from __future__ import annotations

import pytest

from epl.ingest import __main__ as cli
from epl.ingest.football_data import raw_season_path


class TestSeasonRange:
    def test_reads_a_range(self) -> None:
        assert cli._parse_seasons("2005-2008") == [2005, 2006, 2007, 2008]

    def test_reads_a_single_season(self) -> None:
        assert cli._parse_seasons("2019") == [2019]


class TestBuild:
    def test_writes_the_cleaned_match_table(self, project_root, data_dir, capsys) -> None:
        target = raw_season_path(2019, "E0")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((data_dir / "E0_1920_sample.csv").read_bytes())

        out = project_root / "processed" / "matches.csv"
        assert cli.main(["--seasons", "2019", "--divisions", "E0", "build", "--out", str(out)]) == 0

        text = out.read_text(encoding="utf-8")
        assert text.splitlines()[0].startswith("season,division,date,time,home_club,away_club")
        assert "liverpool" in text
        assert "3 matches" in capsys.readouterr().out

    def test_warns_when_the_build_is_not_the_canonical_corpus(
        self, project_root, data_dir, capsys
    ) -> None:
        """ADR 0004: a subset table must not be mistaken for the corpus Predictors are scored on."""
        target = raw_season_path(2019, "E0")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((data_dir / "E0_1920_sample.csv").read_bytes())

        out = project_root / "processed" / "matches.csv"
        cli.main(["--seasons", "2019", "--divisions", "E0", "build", "--out", str(out)])

        printed = capsys.readouterr().out
        assert "not the canonical corpus" in printed
        assert "ADR 0004" in printed
        assert "do not score a Predictor on this table" in printed


class TestClubsAudit:
    def test_reports_success_when_every_spelling_is_known(self, project_root, data_dir) -> None:
        target = raw_season_path(2019, "E0")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((data_dir / "E0_1920_sample.csv").read_bytes())
        assert cli.main(["--seasons", "2019", "--divisions", "E0", "clubs"]) == 0

    def test_exits_nonzero_when_a_spelling_is_unknown(self, project_root, capsys) -> None:
        """An unknown Club must fail the audit rather than be dropped from the corpus."""
        target = raw_season_path(2019, "E0")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\r\n"
            b"E0,09/08/2019,Liverpool,Real Madrid,4,1,H\r\n"
        )
        assert cli.main(["--seasons", "2019", "--divisions", "E0", "clubs"]) == 1
        assert "Real Madrid" in capsys.readouterr().out


class TestFetch:
    def test_fills_the_cache(self, project_root, monkeypatch, capsys) -> None:
        payload = (
            b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\r\n"
            b"E0,09/08/2019,Liverpool,Norwich,4,1,H\r\n"
        )
        monkeypatch.setattr(cli, "fetch_all", lambda s, d, refresh: [raw_season_path(2019, "E0")])
        target = raw_season_path(2019, "E0")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        assert cli.main(["--seasons", "2019", "--divisions", "E0", "fetch"]) == 0
        assert "cached 1 files" in capsys.readouterr().out


class TestArguments:
    def test_requires_a_command(self) -> None:
        with pytest.raises(SystemExit):
            cli.main([])
