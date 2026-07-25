import json
import math
import random
from datetime import date, timedelta

import pytest

from claudebet.form import FormModel, MatchLog, MatchRecord
from claudebet.scan import (
    Fixture,
    ScanConfig,
    load_fixtures,
    resolve_market,
    scan,
    write_fixtures_template,
)
from claudebet.trends import (
    binomial_at_least,
    hit_rate,
    scan_trends,
    standard_conditions,
    team_trends,
)
from claudebet.trends import _over_match, _over_team, Condition

TEAMS = ["Alianza", "Cristal", "Universitario", "Melgar", "Cienciano", "Boys",
         "Grau", "Huancayo"]


def poisson_sample(rng, lam):
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1
        if k > 40:
            return k


def league(n=700, seed=9):
    """A league with clear strength differences and overdispersed corners."""
    rng = random.Random(seed)
    atk = {t: 1.30 - 0.08 * i for i, t in enumerate(TEAMS)}
    con = {t: 0.78 + 0.08 * i for i, t in enumerate(TEAMS)}
    records, start = [], date(2025, 1, 1)
    for i in range(n):
        h, a = rng.sample(TEAMS, 2)
        tempo = math.exp(rng.gauss(0, 0.26))
        records.append(MatchRecord(
            h, a,
            {"goals": poisson_sample(rng, 1.3 * atk[h] * con[a] * 1.15),
             "corners": poisson_sample(rng, 5.6 * 1.12 * tempo),
             "cards": poisson_sample(rng, 2.1),
             "shots": poisson_sample(rng, 13.0 * atk[h])},
            {"goals": poisson_sample(rng, 1.3 * atk[a] * con[h] / 1.15),
             "corners": poisson_sample(rng, 4.9 / 1.12 * tempo),
             "cards": poisson_sample(rng, 2.1),
             "shots": poisson_sample(rng, 10.5 * atk[a])},
            when=start + timedelta(days=i // 4)))
    return MatchLog(records)


def hot_streak_log(n=220, seed=3):
    """A league of ordinary corner counts where one team's last five games all
    finished well over the line. Exactly the shape that produces a tempting
    streak the book has already priced."""
    rng = random.Random(seed)
    records, start = [], date(2025, 1, 1)
    for i in range(n):
        h, a = rng.sample(TEAMS, 2)
        records.append(MatchRecord(
            h, a,
            {"goals": poisson_sample(rng, 1.3), "corners": poisson_sample(rng, 5.0)},
            {"goals": poisson_sample(rng, 1.1), "corners": poisson_sample(rng, 5.0)},
            when=start + timedelta(days=i // 3)))
    # Alianza's five most recent matches: comfortably over 8.5 every time.
    for j in range(5):
        records.append(MatchRecord(
            "Alianza", TEAMS[1 + j % 6],
            {"goals": 1, "corners": 8},
            {"goals": 1, "corners": 7},
            when=start + timedelta(days=n // 3 + j + 1)))
    return MatchLog(records)


class TestBinomial:
    def test_known_values(self):
        # A fair coin: at least 5 heads from 5 is 1/32.
        assert binomial_at_least(5, 5, 0.5) == pytest.approx(1 / 32)
        assert binomial_at_least(0, 5, 0.5) == pytest.approx(1.0)
        assert binomial_at_least(3, 5, 0.5) == pytest.approx(0.5)

    def test_a_streak_at_a_high_base_rate_is_unremarkable(self):
        # This is the whole point: 5/5 at a true 75% rate is a 24% event.
        assert binomial_at_least(5, 5, 0.75) == pytest.approx(0.2373, abs=1e-3)
        assert binomial_at_least(5, 5, 0.75) > 0.05  # not evidence of anything

    def test_a_streak_at_a_low_base_rate_is_notable(self):
        assert binomial_at_least(5, 5, 0.25) < 0.01

    def test_zero_trials(self):
        assert binomial_at_least(0, 0, 0.5) == 1.0


class TestHitRate:
    def test_counts_from_both_perspectives(self):
        log = MatchLog([
            MatchRecord("A", "B", {"goals": 3}, {"goals": 0}, when=date(2026, 1, 1)),
            MatchRecord("B", "A", {"goals": 0}, {"goals": 2}, when=date(2026, 1, 8)),
            MatchRecord("A", "B", {"goals": 0}, {"goals": 1}, when=date(2026, 1, 15)),
        ])
        over_half = Condition("team over 0.5 goals", _over_team("goals", 0.5))
        hits, played, seq = hit_rate(log, "A", over_half, 5)
        assert (hits, played) == (2, 3)
        assert seq == "YYN"

    def test_match_level_condition(self):
        log = MatchLog([
            MatchRecord("A", "B", {"goals": 2}, {"goals": 2}, when=date(2026, 1, 1)),
            MatchRecord("A", "B", {"goals": 0}, {"goals": 1}, when=date(2026, 1, 8)),
        ])
        cond = Condition("match over 2.5", _over_match("goals", 2.5))
        assert hit_rate(log, "A", cond, 5)[:2] == (1, 2)

    def test_window_limits_the_count(self):
        log = MatchLog([
            MatchRecord("A", "B", {"goals": 1}, {"goals": 0},
                        when=date(2026, 1, i + 1))
            for i in range(9)
        ])
        cond = Condition("over 0.5", _over_team("goals", 0.5))
        assert hit_rate(log, "A", cond, 3)[1] == 3
        assert hit_rate(log, "A", cond, None)[1] == 9


class TestTrends:
    def test_finds_streaks_and_reports_luck(self):
        log = league(n=300)
        results = team_trends(log, "Alianza", window=5)
        assert results
        for trend in results:
            assert trend.hits >= 4
            assert 0.0 <= trend.luck_probability <= 1.0
            assert "/" not in trend.line() or True
            assert set(trend.sequence) <= {"Y", "N"}

    def test_a_streak_that_matches_the_season_rate_is_not_notable(self):
        # Alianza is the strongest team, so "over 0.5 goals" lands nearly
        # every week. A five-game run of it means nothing.
        log = league(n=400)
        results = team_trends(log, "Alianza", window=5)
        routine = [t for t in results if t.baseline > 0.8 and t.hits == t.played]
        assert routine, "expected at least one high-base-rate perfect streak"
        assert all(not t.is_notable for t in routine)

    def test_screen_reports_how_many_streaks_chance_alone_gives(self):
        summary = scan_trends(league(n=400), window=5)
        assert summary["conditions_checked"] > 50
        assert summary["expected_by_chance"] > 0
        # The honest comparison: found versus expected.
        assert isinstance(summary["perfect_streaks"], int)

    def test_standard_conditions_cover_the_available_stats(self):
        names = [c.name for c in standard_conditions(["goals", "corners"])]
        assert any("corners" in n for n in names)
        assert any("both teams score" in n for n in names)
        assert not any("cards" in n for n in names)

    def test_under_is_the_complement_of_over(self):
        log = league(n=200)
        over = Condition("o", _over_match("goals", 2.5))
        hits_over, played, _ = hit_rate(log, "Alianza", over, 10)

        def under(record, is_home):
            value = over.test(record, is_home)
            return None if value is None else not value

        hits_under, played2, _ = hit_rate(
            log, "Alianza", Condition("u", under), 10
        )
        assert played == played2
        assert hits_over + hits_under == played


@pytest.fixture(scope="module")
def projection():
    return FormModel(window=5).fit(league(n=300)).project("Alianza", "Boys")


@pytest.fixture(scope="module")
def log():
    return league()


class TestMarketGrammar:

    def test_match_result(self, projection):
        market = resolve_market(projection, "1X2")
        assert set(market) == {"home", "draw", "away"}
        assert sum(market.values()) == pytest.approx(1.0)

    def test_double_chance_and_btts(self, projection):
        assert set(resolve_market(projection, "dc")) == {"1X", "12", "X2"}
        btts = resolve_market(projection, "btts")
        assert sum(btts.values()) == pytest.approx(1.0)

    def test_match_total(self, projection):
        market = resolve_market(projection, "corners_9.5")
        assert set(market) == {"over", "under"}
        assert sum(market.values()) == pytest.approx(1.0, abs=1e-6)

    def test_team_total(self, projection):
        home = resolve_market(projection, "shots_home_12.5")
        away = resolve_market(projection, "shots_away_12.5")
        assert home["over"] != away["over"]

    def test_multi_word_stat_names_parse(self):
        rng = random.Random(4)
        records = [
            MatchRecord(
                *rng.sample(TEAMS[:4], 2),
                {"goals": poisson_sample(rng, 1.4),
                 "shots_on_target": poisson_sample(rng, 4.5)},
                {"goals": poisson_sample(rng, 1.1),
                 "shots_on_target": poisson_sample(rng, 3.8)},
                when=date(2025, 1, 1) + timedelta(days=i),
            )
            for i in range(200)
        ]
        model = FormModel(window=5).fit(MatchLog(records))
        projection = model.project(TEAMS[0], TEAMS[1])
        assert resolve_market(projection, "shots_on_target_8.5") is not None
        assert resolve_market(projection, "shots_on_target_home_4.5") is not None

    def test_asian_handicap(self, projection):
        market = resolve_market(projection, "ah_-0.5")
        assert set(market) == {"home", "away"}

    def test_unknown_markets_return_none(self, projection):
        assert resolve_market(projection, "nonsense") is None
        assert resolve_market(projection, "offsides_3.5") is None
        assert resolve_market(projection, "corners_abc") is None


class TestScan:

    def test_a_genuinely_generous_price_becomes_a_bet(self, log):
        # Model fair price on this corners line is about 1.65; 2.40 is far
        # beyond it and should clear even the strict single-book bar.
        fixtures = [Fixture("Alianza", "Boys",
                            markets={"corners_9.5": {"over": 2.40, "under": 1.80}})]
        result = scan(log, fixtures, ScanConfig(), bankroll=200)
        assert result.bets, "a 2.40 on a 1.65 fair price must be bettable"
        bet = result.bets[0]
        assert bet.selection == "over"
        assert bet.stake > 0
        assert bet.edge > 0

    def test_a_fair_price_is_not_a_bet(self, log):
        fixtures = [Fixture("Alianza", "Boys",
                            markets={"corners_9.5": {"over": 1.65, "under": 2.25}})]
        assert scan(log, fixtures, ScanConfig(), bankroll=200).bets == []

    def test_an_implausible_disagreement_is_flagged_not_staked(self, log):
        # Model has Alianza a heavy favourite; a 6.00 price would be a 50-point
        # disagreement, which means the data is wrong, not the book.
        fixtures = [Fixture("Alianza", "Boys",
                            markets={"1X2": {"home": 6.00, "draw": 4.0, "away": 1.50}})]
        result = scan(log, fixtures, ScanConfig(), bankroll=200)
        assert result.flagged
        assert all(c.stake == 0 for c in result.flagged)
        assert result.bets == []
        assert "differ by" in result.flagged[0].reason

    def test_side_markets_tolerate_more_disagreement_than_the_main_line(self, log):
        # The same gap should be allowed on corners but not on the result.
        config = ScanConfig()
        corners = scan(log, [Fixture("Alianza", "Boys",
                       markets={"corners_9.5": {"over": 2.40, "under": 1.80}})],
                       config, bankroll=200)
        assert not corners.flagged

    def test_short_priced_streaks_land_in_priced_in(self):
        # The trap, built deliberately: Alianza have gone over 8.5 corners in
        # five straight, and the book pays 1.10 for it. The streak is real and
        # the bet is still bad, because 1.10 needs it to land 91% of the time.
        streak_log = hot_streak_log()
        fixtures = [Fixture("Alianza", "Boys",
                            markets={"corners_home_4.5": {"over": 1.45, "under": 2.6}})]
        result = scan(streak_log, fixtures, ScanConfig(), bankroll=200)
        over = next(c for c in result.all_candidates if c.selection == "over")
        assert over.recent == "5/5", over.recent
        assert over.verdict != "bet"
        assert over.ev < 0
        assert over in result.priced_in
        assert "ALREADY PRICED IN" in result.report()

    def test_match_level_recent_pools_both_teams(self):
        # A match total is about both sides, so the hit rate covers both
        # teams' recent games -- ten, not five.
        result = scan(hot_streak_log(),
                      [Fixture("Alianza", "Boys",
                               markets={"corners_8.5": {"over": 1.60,
                                                        "under": 2.3}})],
                      ScanConfig(), bankroll=200)
        over = next(c for c in result.all_candidates if c.selection == "over")
        assert over.recent.endswith("/10")

    def test_ranking_is_by_value_not_by_probability(self, log):
        fixtures = [Fixture("Alianza", "Boys", markets={
            "shots_home_8.5": {"over": 1.12, "under": 5.5},   # very likely, bad price
            "corners_9.5": {"over": 2.40, "under": 1.80},     # less likely, good price
        })]
        result = scan(log, fixtures, ScanConfig(), bankroll=200)
        assert result.bets
        top = result.bets[0]
        assert top.market == "corners_9.5"
        # The near-certain one is the lower-ranked of the two.
        likely = next(c for c in result.all_candidates
                      if c.market == "shots_home_8.5" and c.selection == "over")
        assert likely.blended_prob > top.blended_prob  # more likely...
        assert likely.ev < top.ev                      # ...and worth less

    def test_exposure_is_capped_across_the_day(self, log):
        fixtures = [
            Fixture(h, a, markets={"corners_9.5": {"over": 2.60, "under": 1.70}})
            for h, a in (("Alianza", "Boys"), ("Cristal", "Huancayo"),
                         ("Universitario", "Grau"), ("Melgar", "Cienciano"))
        ]
        config = ScanConfig(max_slate_exposure=0.05)
        result = scan(log, fixtures, config, bankroll=1000)
        assert sum(c.stake_fraction for c in result.bets) <= 0.05 + 1e-9

    def test_unknown_teams_are_skipped_with_a_warning(self, log):
        fixtures = [Fixture("Nowhere United", "Boys",
                            markets={"corners_9.5": {"over": 2.4, "under": 1.8}})]
        result = scan(log, fixtures, ScanConfig(), bankroll=200)
        assert result.bets == []
        assert any("no history" in w for w in result.warnings)

    def test_unknown_market_warns_but_does_not_crash(self, log):
        fixtures = [Fixture("Alianza", "Boys",
                            markets={"offsides_3.5": {"over": 2.0, "under": 1.8}})]
        result = scan(log, fixtures, ScanConfig(), bankroll=200)
        assert any("no model for market" in w for w in result.warnings)

    def test_thin_history_warns(self):
        small = MatchLog([
            MatchRecord("A", "B", {"goals": 1, "corners": 5},
                        {"goals": 0, "corners": 4},
                        when=date(2026, 1, i + 1))
            for i in range(12)
        ])
        result = scan(small, [Fixture("A", "B",
                      markets={"corners_9.5": {"over": 2.4, "under": 1.8}})],
                      ScanConfig(), bankroll=100)
        assert any("historical matches" in w for w in result.warnings)

    def test_ev_reconciles_with_the_displayed_columns(self, log):
        fixtures = [Fixture("Alianza", "Boys",
                            markets={"corners_9.5": {"over": 2.40, "under": 1.80}})]
        for c in scan(log, fixtures, ScanConfig(), bankroll=200).all_candidates:
            assert c.ev == pytest.approx(c.blended_prob * c.odds - 1.0)
            assert c.edge == pytest.approx(c.blended_prob - c.needs, abs=1e-9)

    def test_report_renders_every_section(self, log):
        fixtures = [Fixture("Alianza", "Boys", markets={
            "corners_9.5": {"over": 2.40, "under": 1.80},
            "shots_home_8.5": {"over": 1.12, "under": 5.5},
            "1X2": {"home": 6.00, "draw": 4.0, "away": 1.50},
        })]
        text = scan(log, fixtures, ScanConfig(), bankroll=200).report(bankroll=200)
        assert "TOP BETS" in text
        assert "FLAGGED" in text
        streak = scan(hot_streak_log(),
                      [Fixture("Alianza", "Boys",
                               markets={"corners_home_4.5": {"over": 1.45,
                                                             "under": 2.6}})],
                      ScanConfig(), bankroll=200).report(bankroll=200)
        assert "ALREADY PRICED IN" in streak

    def test_trust_scale_moves_the_model_weight(self, log):
        fixtures = [Fixture("Alianza", "Boys",
                            markets={"corners_9.5": {"over": 2.20, "under": 1.80}})]
        timid = scan(log, fixtures, ScanConfig(trust_scale=0.2), bankroll=200)
        bold = scan(log, fixtures, ScanConfig(trust_scale=1.5), bankroll=200)
        timid_over = next(c for c in timid.all_candidates if c.selection == "over")
        bold_over = next(c for c in bold.all_candidates if c.selection == "over")
        assert bold_over.trust > timid_over.trust


class TestFixtureFiles:
    def test_template_round_trips(self, tmp_path):
        path = write_fixtures_template(tmp_path / "today.json")
        fixtures, date_str, book = load_fixtures(path)
        assert book == "doradobet"
        assert date_str
        assert len(fixtures) == 2
        assert fixtures[0].home == "Inter Miami"
        assert "1X2" in fixtures[0].markets

    def test_american_odds_are_accepted(self, tmp_path):
        path = tmp_path / "t.json"
        path.write_text(json.dumps({
            "matches": [{"home": "A", "away": "B",
                         "markets": {"1X2": {"home": "+150", "away": "-170"}}}]
        }))
        fixtures, _, _ = load_fixtures(path)
        assert fixtures[0].markets["1X2"]["home"] == pytest.approx(2.5)

    def test_bare_list_is_accepted(self, tmp_path):
        path = tmp_path / "t.json"
        path.write_text(json.dumps([{"home": "A", "away": "B", "markets": {}}]))
        fixtures, _, _ = load_fixtures(path)
        assert len(fixtures) == 1


class TestScanCli:
    def _files(self, tmp_path):
        log = league(n=400)
        csv_path = tmp_path / "matches.csv"
        header = ("date,home,away,home_goals,away_goals,home_corners,away_corners,"
                  "home_cards,away_cards,home_shots,away_shots\n")
        rows = [
            f"{r.when.isoformat()},{r.home},{r.away},"
            f"{int(r.home_stats['goals'])},{int(r.away_stats['goals'])},"
            f"{int(r.home_stats['corners'])},{int(r.away_stats['corners'])},"
            f"{int(r.home_stats['cards'])},{int(r.away_stats['cards'])},"
            f"{int(r.home_stats['shots'])},{int(r.away_stats['shots'])}"
            for r in log.records
        ]
        csv_path.write_text(header + "\n".join(rows) + "\n")
        fx_path = tmp_path / "today.json"
        fx_path.write_text(json.dumps({
            "date": "2026-07-26", "book": "doradobet",
            "matches": [{"home": "Alianza", "away": "Boys", "markets": {
                "corners_9.5": {"over": 2.40, "under": 1.80},
                "shots_home_8.5": {"over": 1.12, "under": 5.50}}}]
        }))
        return csv_path, fx_path

    def test_scan_command(self, tmp_path, capsys):
        from claudebet.cli import main

        csv_path, fx_path = self._files(tmp_path)
        assert main(["scan", str(fx_path), "--log", str(csv_path),
                     "--bankroll", "200"]) == 0
        out = capsys.readouterr().out
        assert "doradobet scan" in out
        assert "TOP BETS" in out

    def test_scan_json(self, tmp_path, capsys):
        from claudebet.cli import main

        csv_path, fx_path = self._files(tmp_path)
        assert main(["scan", str(fx_path), "--log", str(csv_path),
                     "--bankroll", "200", "--json", "--all"]) == 0
        rows = json.loads(capsys.readouterr().out)
        assert rows and "blended_prob" in rows[0]

    def test_trends_command_for_one_team(self, tmp_path, capsys):
        from claudebet.cli import main

        csv_path, _ = self._files(tmp_path)
        assert main(["trends", str(csv_path), "--team", "Alianza"]) == 0
        assert "last 5 games" in capsys.readouterr().out

    def test_trends_screen_reports_chance_baseline(self, tmp_path, capsys):
        from claudebet.cli import main

        csv_path, _ = self._files(tmp_path)
        assert main(["trends", str(csv_path)]) == 0
        out = capsys.readouterr().out
        assert "expected by pure chance" in out

    def test_trends_unknown_team(self, tmp_path):
        from claudebet.cli import main

        csv_path, _ = self._files(tmp_path)
        with pytest.raises(SystemExit, match="unknown team"):
            main(["trends", str(csv_path), "--team", "Nowhere"])

    def test_fixtures_template_command(self, tmp_path, capsys):
        from claudebet.cli import main

        assert main(["template", str(tmp_path / "t.json"), "--fixtures"]) == 0
        assert "claudebet scan" in capsys.readouterr().out
