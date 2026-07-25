"""Offline tests for the results importer.

Nothing here touches the network -- the parsers are fed the exact byte layout
football-data.co.uk publishes, captured as fixtures. A test suite that needs
the internet to pass is a test suite that fails on a train.
"""

import pytest

from claudebet.data.sources import load_match_log
from claudebet.fetch import (
    LEAGUES,
    normalise_season,
    parse_extra_csv,
    parse_main_csv,
    write_results_csv,
)

# A real slice of the Premier League file, columns and all.
MAIN_CSV = (
    "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,Referee,"
    "HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR,B365H,B365D,B365A\n"
    "E0,16/08/2024,20:00,Man United,Fulham,1,0,H,0,0,D,R Jones,"
    "14,10,5,2,12,10,7,8,2,3,0,0,1.53,4.20,6.00\n"
    "E0,17/08/2024,12:30,Ipswich,Liverpool,0,2,A,0,0,D,T Robinson,"
    "8,17,2,7,11,9,3,9,1,0,0,1,7.50,4.75,1.40\n"
)

EXTRA_CSV = (
    "Country,League,Season,Date,Time,Home,Away,HG,AG,Res,PSCH,PSCD,PSCA\n"
    "USA,MLS,2024,24/02/2024,20:30,Inter Miami,Real Salt Lake,2,0,H,1.7,4.0,4.5\n"
    "USA,MLS,2024,02/03/2024,19:30,LA Galaxy,Inter Miami,1,1,D,3.1,3.6,2.3\n"
    "USA,MLS,2023,25/02/2023,20:30,Inter Miami,CF Montreal,2,1,H,2.0,3.5,3.6\n"
)


class TestSeasonCodes:
    @pytest.mark.parametrize("given,expected", [
        ("2425", "2425"), ("2024-25", "2425"), ("2024/25", "2425"),
        ("24-25", "2425"), ("2024", "2425"), ("24", "2425"),
        ("1920", "1920"), ("2019-20", "1920"),
    ])
    def test_accepts_the_spellings_people_use(self, given, expected):
        assert normalise_season(given) == expected

    def test_rejects_nonsense(self):
        with pytest.raises(ValueError, match="cannot read season"):
            normalise_season("last year")


class TestMainLeagueParsing:
    def test_reads_every_paired_statistic(self):
        rows = parse_main_csv(MAIN_CSV)
        assert len(rows) == 2
        first = rows[0]
        assert first["date"] == "2024-08-16"
        assert first["home"] == "Man United" and first["away"] == "Fulham"
        assert (first["home_goals"], first["away_goals"]) == (1, 0)
        assert (first["home_corners"], first["away_corners"]) == (7, 8)
        assert (first["home_shots"], first["away_shots"]) == (14, 10)
        assert first["home_shots_on_target"] == 5
        assert (first["home_fouls"], first["away_fouls"]) == (12, 10)

    def test_total_cards_combines_yellows_and_reds(self):
        second = parse_main_csv(MAIN_CSV)[1]
        assert (second["home_yellows"], second["home_reds"]) == (1, 0)
        assert (second["away_yellows"], second["away_reds"]) == (0, 1)
        assert second["home_cards"] == 1
        assert second["away_cards"] == 1  # the red counts as a card

    def test_two_digit_years_still_parse(self):
        old = MAIN_CSV.replace("16/08/2024", "16/08/02")
        assert parse_main_csv(old)[0]["date"] == "2002-08-16"

    def test_rows_without_a_score_are_dropped(self):
        blank = MAIN_CSV.replace(
            "E0,16/08/2024,20:00,Man United,Fulham,1,0,H", "E0,16/08/2024,20:00,A,B,,,"
        )
        assert len(parse_main_csv(blank)) == 1

    def test_missing_columns_are_skipped_not_invented(self):
        minimal = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nE0,16/08/2024,A,B,1,0\n"
        row = parse_main_csv(minimal)[0]
        assert row["home_goals"] == 1
        assert "home_corners" not in row
        assert "home_cards" not in row

    def test_empty_input(self):
        assert parse_main_csv("") == []


class TestExtraLeagueParsing:
    def test_reads_goals_only(self):
        rows = parse_extra_csv(EXTRA_CSV)
        assert len(rows) == 3
        first = rows[0]
        assert first["home"] == "Inter Miami"
        assert first["home_goals"] == 2
        # The feed simply has no corners in it; nothing should be fabricated.
        assert not any(k.endswith("corners") for k in first)

    def test_season_filter(self):
        rows = parse_extra_csv(EXTRA_CSV, seasons=["2024"])
        assert len(rows) == 2
        assert all(r["date"].startswith("2024") for r in rows)

    def test_no_filter_returns_everything(self):
        assert len(parse_extra_csv(EXTRA_CSV, seasons=[])) == 3


class TestLeagueRegistry:
    def test_detailed_and_goals_only_are_both_present(self):
        assert LEAGUES["E0"].detailed
        assert not LEAGUES["USA"].detailed

    def test_codes_are_self_consistent(self):
        for code, league in LEAGUES.items():
            assert league.code == code
            assert league.name


class TestWriting:
    def test_output_loads_back_as_a_match_log(self, tmp_path):
        path = write_results_csv(parse_main_csv(MAIN_CSV), tmp_path / "m.csv")
        log = load_match_log(path)
        assert len(log) == 2
        assert "corners" in log.stats()
        assert "cards" in log.stats()
        assert "shots_on_target" in log.stats()
        assert not log.is_example_data

    def test_goals_only_output_loads_too(self, tmp_path):
        path = write_results_csv(parse_extra_csv(EXTRA_CSV), tmp_path / "m.csv")
        log = load_match_log(path)
        assert log.stats() == ["goals"]
        assert "Inter Miami" in log.teams()

    def test_every_written_column_is_paired(self, tmp_path):
        path = write_results_csv(parse_main_csv(MAIN_CSV), tmp_path / "m.csv")
        header = path.read_text().split("\n")[0].split(",")
        for column in header:
            if column.startswith("home_"):
                assert f"away_{column[5:]}" in header

    def test_refuses_to_write_nothing(self, tmp_path):
        with pytest.raises(ValueError, match="nothing to write"):
            write_results_csv([], tmp_path / "m.csv")


class TestFetchCli:
    def test_list_shows_both_kinds(self, capsys):
        from claudebet.cli import main

        assert main(["fetch-results", "--list"]) == 0
        out = capsys.readouterr().out
        assert "corners, shots and cards" in out
        assert "goals only" in out
        assert "E0" in out and "USA" in out
        assert "Peru" in out  # the gap is stated, not hidden

    def test_league_is_required(self):
        from claudebet.cli import main

        with pytest.raises(SystemExit, match="which league"):
            main(["fetch-results"])

    def test_unknown_league_is_rejected(self, tmp_path):
        from claudebet.cli import main

        with pytest.raises(SystemExit, match="unknown league"):
            main(["fetch-results", "--league", "ZZ9",
                  "--out", str(tmp_path / "m.csv")])

    def test_will_not_clobber_your_own_data(self, tmp_path):
        from claudebet.cli import main

        path = tmp_path / "m.csv"
        write_results_csv(parse_main_csv(MAIN_CSV), path)
        with pytest.raises(SystemExit, match="already exists"):
            main(["fetch-results", "--league", "E0", "--out", str(path)])

    def test_example_data_may_be_replaced_freely(self, tmp_path, monkeypatch):
        from claudebet.cli import main
        from claudebet.data.sources import write_match_log_template

        path = write_match_log_template(tmp_path / "m.csv")
        # The example file is not precious, so the guard must not block it.
        # Stub the download so the test stays offline.
        monkeypatch.setattr(
            "claudebet.fetch._download", lambda url, timeout=30.0: MAIN_CSV
        )
        assert main(["fetch-results", "--league", "E0", "--out", str(path)]) == 0
        assert load_match_log(path).is_example_data is False
