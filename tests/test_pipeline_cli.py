import json

import pytest

from claudebet.cli import main
from claudebet.data.sources import from_the_odds_api, load_csv, load_json, write_template
from claudebet.market import Market
from claudebet.pipeline import AnalysisConfig, analyze, analyze_slate
from claudebet.store import BetLog


# A realistic three-way market. Two sharp books at ~3% hold agree that the
# home side is a shade under 42.5%; the soft book is generous on the home price
# and pads the other two to a ~6% hold, as a soft book actually does.
#
# The soft price cannot be moved much beyond 2.4675 without creating a genuine
# cross-book arbitrage against the sharp books' draw and away prices -- which is
# itself the reason large edges on liquid two- and three-way markets are rare.
SHARP = {
    "pinnacle": {"H": 2.30, "D": 3.45, "A": 3.30},
    "circa": {"H": 2.29, "D": 3.44, "A": 3.28},
}

# The default thresholds are deliberately strict enough to decline the largest
# realistic outlier, so tests that exercise the bet path relax them explicitly.
LENIENT = AnalysisConfig(min_edge=0.003, confidence_z=0.5)
OPEN = AnalysisConfig(min_edge=0.001, confidence_z=0.0)


def market_with(soft_h, key="game1", book="hardrock", hold=1.06):
    """SHARP plus one soft book priced generously on H at the given hold."""
    rest = hold - 1.0 / soft_h
    prices = {
        **SHARP,
        book: {"H": soft_h, "D": 1.0 / (rest * 0.489), "A": 1.0 / (rest * 0.511)},
    }
    return Market.from_book_prices(key, prices)


class TestAnalyze:
    def test_flags_a_soft_book_outlier(self):
        result = analyze(market_with(2.45), config=LENIENT, bankroll=1000)
        bets = result.bets
        assert bets, "2.45 against a 42.5% consensus should be bettable"
        assert bets[0].book == "hardrock"
        assert bets[0].outcome == "H"
        assert bets[0].evaluation.stake > 0

    def test_defaults_decline_the_same_marginal_edge(self):
        # 1.7 points of raw edge does not survive the default haircut. This is
        # the intended behaviour, not a miss: at that size the probability the
        # true edge is negative is substantial.
        result = analyze(market_with(2.45), bankroll=1000)
        assert result.bets == []
        opp = next(o for o in result.opportunities if o.book == "hardrock"
                   and o.outcome == "H")
        assert opp.evaluation.edge_pp > 0  # raw edge is real
        assert opp.evaluation.used_prob < opp.evaluation.fair_prob

    def test_no_bet_when_every_book_agrees(self):
        result = analyze(Market.from_book_prices("g", SHARP), config=OPEN,
                         bankroll=1000)
        assert result.bets == []

    def test_benchmark_excludes_the_book_being_evaluated(self):
        market = market_with(2.45)
        opp = next(o for o in analyze(market).opportunities
                   if o.book == "hardrock" and o.outcome == "H")
        assert opp.n_books_in_benchmark == 2  # pinnacle + circa, not itself

        # The soft book's generous H price *is* a low opinion of H (39.5% once
        # devigged). Leaving it in the benchmark therefore pulls the fair
        # probability down toward the very price under test and understates
        # the edge -- which is exactly what leave-one-out exists to prevent.
        with_itself = analyze(market, config=AnalysisConfig(exclude_own_book=False))
        opp_incl = next(o for o in with_itself.opportunities
                        if o.book == "hardrock" and o.outcome == "H")
        assert opp.consensus_prob > opp_incl.consensus_prob
        assert opp.evaluation.edge_pp > opp_incl.evaluation.edge_pp

    def test_thin_market_warns_and_still_answers(self):
        thin = Market.from_book_prices("g", {"pinnacle": {"A": 2.0, "B": 2.0}})
        result = analyze(thin, bankroll=1000)
        assert any("book" in w for w in result.warnings)
        assert result.fair_probs["A"] == pytest.approx(0.5)

    def test_model_shifts_the_fair_price(self):
        market = Market.from_book_prices("g", SHARP)
        baseline = analyze(market).fair_probs["H"]
        result = analyze(
            market,
            model_probs={"H": 0.70, "D": 0.20, "A": 0.10},
            config=AnalysisConfig(model_weight=0.4),
        )
        assert result.fair_probs["H"] > baseline
        assert result.fair_probs["H"] < 0.70  # blended, not adopted wholesale
        assert sum(result.fair_probs.values()) == pytest.approx(1.0)

    def test_model_weight_zero_ignores_the_model(self):
        market = Market.from_book_prices("g", SHARP)
        result = analyze(market, model_probs={"H": 0.98, "D": 0.01, "A": 0.01},
                         config=AnalysisConfig(model_weight=0.0))
        assert result.fair_probs["H"] == pytest.approx(analyze(market).fair_probs["H"])

    def test_partial_model_is_ignored_with_a_warning(self):
        market = Market.from_book_prices("g", SHARP)
        result = analyze(market, model_probs={"H": 0.7},
                         config=AnalysisConfig(model_weight=0.5))
        assert any("no opinion" in w for w in result.warnings)

    def test_strict_mode_rejects_devig_dependent_edges(self):
        market = market_with(2.45)
        loose = analyze(market, config=LENIENT)
        strict = analyze(
            market,
            config=AnalysisConfig(min_edge=0.003, confidence_z=0.5,
                                  require_positive_worst_case=True),
        )
        assert loose.bets and not strict.bets
        assert "worst-case" in strict.opportunities[0].evaluation.reason

    def test_higher_confidence_z_bets_less(self):
        market = market_with(2.45)
        timid = analyze(market, config=AnalysisConfig(min_edge=0.001, confidence_z=3.0),
                        bankroll=1000)
        bold = analyze(market, config=OPEN, bankroll=1000)
        assert sum(b.evaluation.stake for b in timid.bets) < sum(
            b.evaluation.stake for b in bold.bets
        )

    def test_report_renders(self):
        text = analyze(market_with(2.45), config=LENIENT, bankroll=1000).report()
        assert "consensus from" in text
        assert "hardrock" in text

    def test_arbitrage_surfaces_in_the_report(self):
        market = Market.from_book_prices(
            "g", {"a": {"A": 2.20, "B": 1.70}, "b": {"A": 1.70, "B": 2.30}}
        )
        result = analyze(market)
        assert result.arbitrage is not None
        assert "ARBITRAGE" in result.report()

    def test_no_arbitrage_in_the_realistic_fixture(self):
        # Guards the fixture itself: a soft price above ~2.4675 would arb the
        # sharp books and stop being a realistic test of edge detection.
        assert analyze(market_with(2.45)).arbitrage is None

    def test_line_shopping_gain_is_positive_when_prices_differ(self):
        assert analyze(market_with(2.45)).shopping_gain["H"] > 0


class TestSlate:
    def test_joint_staking_cuts_total_exposure(self):
        markets = [market_with(2.45, key=f"g{i}") for i in range(4)]
        cfg = AnalysisConfig(min_edge=0.001, confidence_z=0.0, max_slate_exposure=0.06)
        slate = analyze_slate(markets, bankroll=1000, config=cfg)
        assert len(slate["recommendations"]) == 4
        assert slate["total_exposure"] <= 0.06 + 1e-9
        solo = sum(r["standalone_fraction"] for r in slate["recommendations"])
        assert slate["total_exposure"] < solo

    def test_same_selection_at_two_books_counted_once(self):
        market = market_with(2.45)
        # A second soft book on the same side at a slightly worse price.
        second = market_with(2.44, book="espnbet")
        for q in second.quotes:
            if q.book == "espnbet":
                market.quotes.append(q)
        slate = analyze_slate([market], bankroll=1000, config=OPEN)
        picks = [(r["market"], r["outcome"]) for r in slate["recommendations"]]
        assert picks and len(picks) == len(set(picks))
        assert slate["recommendations"][0]["book"] == "hardrock"  # the better price

    def test_empty_slate_is_safe(self):
        slate = analyze_slate([Market.from_book_prices("g", SHARP)], bankroll=1000)
        assert slate["recommendations"] == []
        assert slate["total_exposure"] == 0.0


class TestDataSources:
    def test_json_round_trip(self, tmp_path):
        path = write_template(tmp_path / "m.json")
        markets = load_json(path)
        assert len(markets) == 1
        assert markets[0].outcomes == ("KC", "BUF")
        assert markets[0].quote("pinnacle", "KC").limit == 25000

    def test_csv_loading_groups_by_market(self, tmp_path):
        path = tmp_path / "q.csv"
        path.write_text(
            "market,book,outcome,odds\n"
            "g1,pinnacle,A,2.00\ng1,pinnacle,B,2.00\n"
            "g2,pinnacle,X,1.50\ng2,pinnacle,Y,2.80\n"
        )
        markets = load_csv(path)
        assert {m.key for m in markets} == {"g1", "g2"}

    def test_american_odds_in_files_are_parsed(self, tmp_path):
        path = tmp_path / "q.csv"
        path.write_text("market,book,outcome,odds\ng,pin,A,+150\ng,pin,B,-170\n")
        market = load_csv(path)[0]
        assert market.quote("pin", "A").odds == pytest.approx(2.5)

    def test_the_odds_api_shape(self):
        payload = [
            {
                "home_team": "BUF",
                "away_team": "KC",
                "commence_time": "2026-01-04T18:00:00Z",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "KC", "price": 2.05},
                                    {"name": "BUF", "price": 1.87},
                                ],
                            },
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {"name": "KC", "price": 1.91, "point": 1.5},
                                    {"name": "BUF", "price": 1.91, "point": -1.5},
                                ],
                            },
                        ],
                    }
                ],
            }
        ]
        markets = from_the_odds_api(payload)
        keys = [m.key for m in markets]
        assert any("moneyline" in k for k in keys)
        assert any("spread 1.5" in k for k in keys)
        spread = next(m for m in markets if "spread" in m.key)
        assert set(spread.outcomes) == {"KC +1.5", "BUF -1.5"}


class TestBetLog:
    def test_full_lifecycle(self, tmp_path):
        log = BetLog(tmp_path / "bets.db")
        bet_id = log.record("g1", "A", "pinnacle", 2.10, 50, fair_prob=0.52,
                            tags=("nfl",))
        assert log.pending()[0]["id"] == bet_id
        assert log.needs_closing_line()
        log.set_closing(bet_id, 1.95, closing_fair_prob=0.53)
        log.grade(bet_id, "win")
        assert log.pending() == []
        result = log.performance(1000)
        assert result.profit == pytest.approx(55.0)
        assert result.clv["mean_pp"] > 0

    def test_calibration_needs_graded_bets(self, tmp_path):
        log = BetLog(tmp_path / "bets.db")
        assert log.calibration()["n"] == 0
        with pytest.raises(ValueError):
            log.performance()

    def test_calibration_reports_once_graded(self, tmp_path):
        log = BetLog(tmp_path / "bets.db")
        for i in range(30):
            bet_id = log.record("g", "A", "b", 2.0, 10, fair_prob=0.5)
            log.set_closing(bet_id, 2.0)
            log.grade(bet_id, "win" if i % 2 else "loss")
        report = log.calibration()
        assert report["n"] == 30
        assert report["brier"] == pytest.approx(0.25)
        assert "warning" in report  # too small a sample to conclude


class TestCli:
    def test_devig_runs(self, capsys):
        assert main(["devig", "1.91", "1.91"]) == 0
        out = capsys.readouterr().out
        assert "50.00%" in out
        assert "hold" in out

    def test_devig_all_methods(self, capsys):
        assert main(["devig", "1.05", "21.0", "--all", "--labels", "fav", "dog"]) == 0
        out = capsys.readouterr().out
        assert "shin" in out and "power" in out
        assert "max disagreement" in out

    def test_kelly_command(self, capsys):
        assert main(["kelly", "--prob", "0.60", "--odds", "2.0",
                     "--bankroll", "1000", "--stderr", "0"]) == 0
        out = capsys.readouterr().out
        assert "BET" in out
        assert "full Kelly" in out

    def test_kelly_declines_a_bad_price(self, capsys):
        assert main(["kelly", "--prob", "0.40", "--odds", "2.0"]) == 0
        assert "NO-EDGE" in capsys.readouterr().out

    @staticmethod
    def _market_file(tmp_path):
        market = market_with(2.45)
        path = tmp_path / "m.json"
        path.write_text(json.dumps({
            "key": market.key,
            "outcomes": list(market.outcomes),
            "prices": market.by_book(),
        }))
        return path

    def test_analyze_command(self, tmp_path, capsys):
        path = self._market_file(tmp_path)
        assert main(["analyze", str(path), "--bankroll", "1000",
                     "--min-edge", "0.003", "--confidence", "0.5"]) == 0
        out = capsys.readouterr().out
        assert "hardrock" in out and "bets:" in out

    def test_analyze_defaults_decline(self, tmp_path, capsys):
        path = self._market_file(tmp_path)
        assert main(["analyze", str(path), "--bankroll", "1000"]) == 0
        assert "no bet" in capsys.readouterr().out

    def test_analyze_json_output(self, tmp_path, capsys):
        path = self._market_file(tmp_path)
        assert main(["analyze", str(path), "--json", "--bankroll", "1000",
                     "--min-edge", "0.003", "--confidence", "0.5"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["fair_probs"]
        assert payload["bets"][0]["verdict"] == "bet"

    def test_parlay_command(self, capsys):
        assert main(["parlay", "--leg", "a:0.6:1.6", "--leg", "b:0.5:1.9"]) == 0
        out = capsys.readouterr().out
        assert "independent prob" in out
        assert "negative EV" in out

    def test_parlay_with_correlation(self, capsys):
        assert main(["parlay", "--leg", "a:0.6:1.6", "--leg", "b:0.5:1.9",
                     "--corr", "a,b:0.4"]) == 0
        assert "from correlation" in capsys.readouterr().out

    def test_parlay_rejects_bad_leg(self):
        with pytest.raises(SystemExit):
            main(["parlay", "--leg", "nonsense"])

    def test_clv_command(self, capsys):
        assert main(["clv", "--taken", "2.10", "--closing", "1.95"]) == 0
        assert "beat the close" in capsys.readouterr().out

    def test_template_command(self, tmp_path, capsys):
        assert main(["template", str(tmp_path / "t.json")]) == 0
        assert "wrote" in capsys.readouterr().out

    def test_log_commands(self, tmp_path, capsys):
        db = str(tmp_path / "b.db")
        assert main(["log", "--db", db, "record", "--market", "g",
                     "--outcome", "A", "--odds", "2.0", "--stake", "10",
                     "--fair-prob", "0.55"]) == 0
        assert main(["log", "--db", db, "pending"]) == 0
        assert main(["log", "--db", db, "close", "1", "1.90"]) == 0
        assert main(["log", "--db", db, "grade", "1", "win"]) == 0
        assert main(["log", "--db", db, "report"]) == 0
        out = capsys.readouterr().out
        assert "profit" in out

    def test_missing_file_exits(self):
        with pytest.raises(SystemExit):
            main(["analyze", "/nonexistent/path.json"])
