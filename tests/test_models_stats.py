import math
import random
from datetime import date, timedelta

import pytest

from claudebet.backtest import SettledBet, clv_report, run
from claudebet.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    brier_score,
    expected_calibration_error,
    log_loss,
    reliability_table,
    skill_score,
)
from claudebet.correlation import (
    ParlayLeg,
    bivariate_normal_cdf,
    norm_cdf,
    norm_ppf,
    parlay,
)
from claudebet.models.elo import Elo, EloConfig, GameResult, tune_k
from claudebet.models.poisson import DixonColes, Match
from claudebet.models.ratings import Game, PowerRatings
from claudebet.portfolio import Candidate, optimize, scenarios_for


class TestNormalMath:
    def test_norm_cdf_known_points(self):
        assert norm_cdf(0.0) == pytest.approx(0.5)
        assert norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
        assert norm_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)

    @pytest.mark.parametrize("p", [0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999])
    def test_ppf_inverts_cdf(self, p):
        assert norm_cdf(norm_ppf(p)) == pytest.approx(p, abs=1e-9)

    @pytest.mark.parametrize("rho", [-0.8, -0.3, 0.0, 0.3, 0.8])
    def test_bivariate_matches_closed_form_at_origin(self, rho):
        # Phi2(0, 0, rho) = 1/4 + arcsin(rho) / (2*pi), exactly.
        expected = 0.25 + math.asin(rho) / (2 * math.pi)
        assert bivariate_normal_cdf(0.0, 0.0, rho) == pytest.approx(expected, abs=1e-8)

    def test_bivariate_independence_factorises(self):
        assert bivariate_normal_cdf(0.5, -0.3, 0.0) == pytest.approx(
            norm_cdf(0.5) * norm_cdf(-0.3)
        )


class TestParlay:
    legs = [ParlayLeg("a", 0.6, 1.60), ParlayLeg("b", 0.5, 1.90)]

    def test_independent_is_the_product(self):
        result = parlay(self.legs)
        assert result.fair_prob == pytest.approx(0.30)
        assert result.independent_prob == pytest.approx(0.30)
        assert result.correlation_gain == pytest.approx(0.0)

    def test_positive_correlation_raises_the_joint_probability(self):
        result = parlay(self.legs, correlations={("a", "b"): 0.4})
        assert result.fair_prob > result.independent_prob
        assert result.correlation_gain > 0

    def test_negative_correlation_lowers_it(self):
        result = parlay(self.legs, correlations={("a", "b"): -0.4})
        assert result.fair_prob < result.independent_prob

    def test_marginals_are_preserved(self):
        # A perfectly correlated pair collapses to the smaller marginal.
        result = parlay(self.legs, correlations={("a", "b"): 0.999})
        assert result.fair_prob == pytest.approx(0.5, abs=0.02)

    def test_parlay_of_vigged_legs_is_negative_ev(self):
        # Two -110 legs at fair 50% each: the parlay carries compounded hold.
        legs = [ParlayLeg("x", 0.5, 1.9091), ParlayLeg("y", 0.5, 1.9091)]
        result = parlay(legs)
        assert result.ev_per_unit < 0
        assert result.hold > 0.08  # roughly double a single leg's hold

    def test_three_legs_use_monte_carlo_and_stay_sane(self):
        legs = self.legs + [ParlayLeg("c", 0.4, 2.4)]
        result = parlay(
            legs, correlations={("a", "b"): 0.3, ("a", "c"): 0.2}, samples=40000
        )
        assert "copula" in result.method
        assert 0.0 < result.fair_prob < min(leg.prob for leg in legs)

    def test_reproducible(self):
        kwargs = dict(
            correlations={("a", "b"): 0.3, ("a", "c"): 0.2}, samples=20000
        )
        legs = self.legs + [ParlayLeg("c", 0.4, 2.4)]
        assert parlay(legs, **kwargs).fair_prob == parlay(legs, **kwargs).fair_prob

    def test_duplicate_leg_names_rejected(self):
        with pytest.raises(ValueError):
            parlay([ParlayLeg("a", 0.5, 2.0), ParlayLeg("a", 0.5, 2.0)])

    def test_single_leg_rejected(self):
        with pytest.raises(ValueError):
            parlay([ParlayLeg("a", 0.5, 2.0)])


class TestPortfolio:
    def test_single_bet_recovers_standalone_kelly(self):
        result = optimize(
            [Candidate("only", 0.6, 2.0)],
            kelly_multiplier=1.0,
            max_exposure=1.0,
            max_per_bet=1.0,
        )
        assert result.stakes["only"] == pytest.approx(0.2, abs=1e-3)

    def test_simultaneous_bets_are_sized_below_standalone(self):
        cands = [Candidate(f"b{i}", 0.58, 2.0) for i in range(4)]
        result = optimize(
            cands, kelly_multiplier=1.0, max_exposure=1.0, max_per_bet=1.0
        )
        standalone = 0.16  # (0.58*1 - 0.42)/1
        assert all(v < standalone for v in result.stakes.values())
        assert result.total_exposure < 4 * standalone

    def test_mutually_exclusive_bets_never_both_win(self):
        cands = [
            Candidate("home", 0.55, 2.0, group="game"),
            Candidate("away", 0.45, 2.3, group="game"),
        ]
        scenarios = scenarios_for(cands)
        for prob, payoff in scenarios:
            assert not (payoff[0] > 0 and payoff[1] > 0)
        assert sum(p for p, _ in scenarios) == pytest.approx(1.0)

    def test_group_probabilities_over_one_rejected(self):
        with pytest.raises(ValueError):
            scenarios_for(
                [
                    Candidate("a", 0.7, 2.0, group="g"),
                    Candidate("b", 0.7, 2.0, group="g"),
                ]
            )

    def test_exposure_cap_is_respected(self):
        cands = [Candidate(f"b{i}", 0.65, 2.0) for i in range(5)]
        result = optimize(cands, kelly_multiplier=1.0, max_exposure=0.10,
                          max_per_bet=0.05)
        assert result.total_exposure <= 0.10 + 1e-9
        assert all(v <= 0.05 + 1e-9 for v in result.stakes.values())

    def test_negative_ev_bets_get_nothing(self):
        result = optimize(
            [Candidate("good", 0.60, 2.0), Candidate("bad", 0.40, 2.0)],
            kelly_multiplier=1.0,
            max_exposure=1.0,
            max_per_bet=1.0,
        )
        assert result.stakes["bad"] == pytest.approx(0.0, abs=1e-6)
        assert result.stakes["good"] > 0

    def test_empty_slate(self):
        result = optimize([])
        assert result.stakes == {}
        assert result.method == "exact"

    def test_hedging_a_market_justifies_more_size_than_standalone(self):
        # Two outcomes of one market cannot both lose, so the pair is far less
        # risky than either leg alone. Log-optimal stakes are therefore larger
        # than standalone Kelly, not smaller -- the opposite of the
        # independent-bets case, and a direct check on the covariance terms.
        cands = [
            Candidate("H", 0.45, 2.50, group="g"),
            Candidate("D", 0.30, 3.60, group="g"),
        ]
        result = optimize(cands, kelly_multiplier=1.0, max_exposure=1.0,
                          max_per_bet=1.0)
        assert result.stakes["H"] > result.standalone["H"]
        assert result.stakes["D"] > result.standalone["D"]

    @pytest.mark.parametrize(
        "n,prob,odds", [(4, 0.55, 2.0), (6, 0.55, 2.0), (8, 0.60, 2.0), (10, 0.55, 2.5)]
    )
    def test_approximation_tracks_the_exact_solver(self, n, prob, odds):
        cands = [Candidate(f"g{i}", prob, odds) for i in range(n)]
        kwargs = dict(kelly_multiplier=1.0, max_exposure=1.0, max_per_bet=1.0)
        exact = optimize(cands, **kwargs)
        approx = optimize(cands, max_scenarios=1, **kwargs)  # force the fallback
        assert exact.method == "exact"
        assert approx.method == "quadratic+line-search"
        assert approx.total_exposure == pytest.approx(exact.total_exposure, rel=0.05)

    def test_large_slate_does_not_blow_up(self):
        # Twenty independent bets is 2**20 joint outcomes -- a normal Sunday,
        # and far past what can be enumerated.
        cands = [Candidate(f"g{i}", 0.55, 2.00) for i in range(20)]
        result = optimize(cands, max_exposure=0.15, max_per_bet=0.02)
        assert result.method == "quadratic+line-search"
        assert result.total_exposure <= 0.15 + 1e-9
        assert all(v <= 0.02 + 1e-9 for v in result.stakes.values())
        assert result.growth_rate > 0

    def test_scenarios_for_still_refuses_a_huge_space(self):
        cands = [Candidate(f"g{i}", 0.55, 2.00) for i in range(30)]
        with pytest.raises(ValueError, match="exceeds"):
            scenarios_for(cands)

    def test_large_slate_ignores_negative_ev_bets(self):
        cands = [Candidate(f"good{i}", 0.60, 2.00) for i in range(18)]
        cands += [Candidate(f"bad{i}", 0.40, 2.00) for i in range(4)]
        result = optimize(cands, max_exposure=0.20, max_per_bet=0.02)
        assert result.method == "quadratic+line-search"
        assert all(result.stakes[f"bad{i}"] == pytest.approx(0.0, abs=1e-9)
                   for i in range(4))
        assert sum(result.stakes[f"good{i}"] for i in range(18)) > 0

    def test_approximation_is_reproducible(self):
        cands = [Candidate(f"g{i}", 0.55, 2.00) for i in range(20)]
        a = optimize(cands, max_exposure=0.15)
        b = optimize(cands, max_exposure=0.15)
        assert a.stakes == b.stakes


class TestCalibration:
    def test_brier_and_log_loss_at_perfection(self):
        assert brier_score([1 - 1e-12, 1e-12], [1, 0]) == pytest.approx(0.0, abs=1e-9)
        assert log_loss([1 - 1e-9, 1e-9], [1, 0]) == pytest.approx(0.0, abs=1e-6)

    def test_brier_of_a_coin_flip(self):
        assert brier_score([0.5, 0.5], [1, 0]) == pytest.approx(0.25)

    def test_skill_score_sign(self):
        outcomes = [1, 1, 0, 0]
        good = [0.9, 0.8, 0.2, 0.1]
        poor = [0.6, 0.55, 0.45, 0.4]
        assert skill_score(good, outcomes, poor) > 0
        assert skill_score(poor, outcomes, good) < 0

    def test_reliability_bins_recover_the_truth(self):
        rng = random.Random(1)
        probs, outcomes = [], []
        for _ in range(4000):
            p = rng.choice([0.2, 0.5, 0.8])
            probs.append(p)
            outcomes.append(1 if rng.random() < p else 0)
        table = reliability_table(probs, outcomes, bins=10)
        for b in table:
            assert b.observed == pytest.approx(b.mean_pred, abs=0.05)
        assert expected_calibration_error(probs, outcomes) < 0.03

    def test_platt_leaves_a_calibrated_model_alone(self):
        rng = random.Random(7)
        probs, outcomes = [], []
        for _ in range(3000):
            p = rng.uniform(0.05, 0.95)
            probs.append(p)
            outcomes.append(1 if rng.random() < p else 0)
        cal = PlattCalibrator().fit(probs, outcomes)
        assert cal.a == pytest.approx(1.0, abs=0.15)
        assert cal.b == pytest.approx(0.0, abs=0.15)

    def test_platt_corrects_overconfidence(self):
        # Predictions are the truth stretched in log-odds: over-confident.
        rng = random.Random(11)
        true_p, probs, outcomes = [], [], []
        for _ in range(4000):
            t = rng.uniform(0.15, 0.85)
            stretched = 1 / (1 + math.exp(-1.8 * math.log(t / (1 - t))))
            true_p.append(t)
            probs.append(stretched)
            outcomes.append(1 if rng.random() < t else 0)
        cal = PlattCalibrator().fit(probs, outcomes)
        assert cal.a < 0.85  # flattening, as it should
        before = brier_score(probs, outcomes)
        after = brier_score([cal(p) for p in probs], outcomes)
        assert after < before
        assert "over-confident" in cal.describe()

    def test_isotonic_is_monotone_and_improves_fit(self):
        rng = random.Random(3)
        probs, outcomes = [], []
        for _ in range(2000):
            t = rng.uniform(0.1, 0.9)
            probs.append(min(0.99, t * 0.6 + 0.2))  # compressed predictions
            outcomes.append(1 if rng.random() < t else 0)
        cal = IsotonicCalibrator().fit(probs, outcomes)
        grid = [i / 50 for i in range(1, 50)]
        mapped = [cal(x) for x in grid]
        assert all(b >= a - 1e-9 for a, b in zip(mapped, mapped[1:]))
        assert brier_score([cal(p) for p in probs], outcomes) < brier_score(
            probs, outcomes
        )


class TestBacktest:
    def test_profit_arithmetic(self):
        bets = [
            SettledBet("A", 2.0, 100, "win"),
            SettledBet("B", 2.0, 100, "loss"),
            SettledBet("C", 3.0, 100, "win"),
        ]
        result = run(bets, starting_bankroll=1000)
        assert result.profit == pytest.approx(100 - 100 + 200)
        assert result.turnover == pytest.approx(300)
        assert result.roi == pytest.approx(200 / 300)
        assert result.final_bankroll == pytest.approx(1200)

    def test_push_and_void_return_the_stake(self):
        result = run([SettledBet("A", 2.0, 100, "push"),
                      SettledBet("B", 2.0, 100, "void")])
        assert result.profit == 0.0
        assert result.n_settled == 0

    def test_half_results(self):
        assert SettledBet("A", 3.0, 100, "half-win").profit == pytest.approx(100)
        assert SettledBet("A", 3.0, 100, "half-loss").profit == pytest.approx(-50)

    def test_drawdown_is_measured_from_the_peak(self):
        bets = [
            SettledBet("A", 2.0, 100, "win"),   # 1100
            SettledBet("B", 2.0, 100, "loss"),  # 1000
            SettledBet("C", 2.0, 100, "loss"),  # 900
        ]
        result = run(bets, starting_bankroll=1000)
        assert result.max_drawdown == pytest.approx(200)
        assert result.longest_losing_streak == 2

    def test_clv_positive_when_you_beat_the_close(self):
        bets = [SettledBet("A", 2.10, 100, "loss", closing_odds=1.95)] * 60
        report = clv_report(bets)
        assert report["mean_pp"] > 0
        assert report["beat_rate"] == 1.0
        result = run(bets)
        assert "beating the closing line" in result.verdict()

    def test_losing_to_the_close_is_called_out(self):
        bets = [SettledBet("A", 1.90, 100, "win", closing_odds=2.10)] * 60
        result = run(bets)
        assert "not beating the closing line" in result.verdict()

    def test_confidence_interval_brackets_the_point_estimate(self):
        rng = random.Random(5)
        bets = [
            SettledBet("A", 2.0, 100, "win" if rng.random() < 0.55 else "loss")
            for _ in range(500)
        ]
        result = run(bets, bootstrap_draws=400)
        lo, hi = result.roi_ci95
        assert lo < result.roi < hi

    def test_small_sample_refuses_to_conclude(self):
        result = run([SettledBet("A", 2.0, 10, "win")] * 20)
        assert "no conclusion" in result.verdict()

    def test_fractional_restaking_compounds(self):
        bets = [SettledBet("A", 2.0, 0.1, "win"),
                SettledBet("B", 2.0, 0.1, "win")]
        result = run(bets, starting_bankroll=1000, restake_fractionally=True)
        assert result.final_bankroll == pytest.approx(1000 * 1.1 * 1.1)

    def test_tags_split_the_record(self):
        bets = [
            SettledBet("A", 2.0, 100, "win", tags=("nfl",)),
            SettledBet("B", 2.0, 100, "loss", tags=("nba",)),
        ]
        result = run(bets)
        assert result.by_tag["nfl"]["roi"] == pytest.approx(1.0)
        assert result.by_tag["nba"]["roi"] == pytest.approx(-1.0)

    def test_bad_result_string_rejected(self):
        with pytest.raises(ValueError):
            SettledBet("A", 2.0, 100, "kind of won")


class TestElo:
    def test_winner_gains_and_loser_loses_the_same(self):
        elo = Elo(EloConfig(k=20, home_advantage=0, use_mov=False))
        elo.update(GameResult("A", "B", 3, 1))
        assert elo.rating("A") > 1500 > elo.rating("B")
        assert elo.rating("A") - 1500 == pytest.approx(1500 - elo.rating("B"))

    def test_home_advantage_favours_the_host(self):
        elo = Elo(EloConfig(home_advantage=55))
        assert elo.win_probability("A", "B") > 0.5
        assert elo.win_probability("A", "B", neutral=True) == pytest.approx(0.5)

    def test_bigger_win_moves_ratings_more(self):
        narrow = Elo(EloConfig(home_advantage=0))
        narrow.update(GameResult("A", "B", 1, 0))
        blowout = Elo(EloConfig(home_advantage=0))
        blowout.update(GameResult("A", "B", 40, 0))
        assert blowout.ratings["A"] > narrow.ratings["A"]

    def test_ratings_converge_on_a_dominant_team(self):
        elo = Elo(EloConfig(k=20, home_advantage=0))
        elo.fit([GameResult("A", "B", 2, 0) for _ in range(40)])
        assert elo.win_probability("A", "B", neutral=True) > 0.85

    def test_three_way_probabilities_sum_to_one(self):
        elo = Elo(EloConfig(draw_nu=1.0))
        probs = elo.probabilities("A", "B")
        assert set(probs) == {"home", "draw", "away"}
        assert sum(probs.values()) == pytest.approx(1.0)
        assert probs["draw"] > 0.2

    def test_draw_probability_peaks_between_equals(self):
        elo = Elo(EloConfig(draw_nu=1.0, home_advantage=0),
                  ratings={"A": 1500, "B": 1500, "C": 1900})
        even = elo.probabilities("A", "B")["draw"]
        lopsided = elo.probabilities("A", "C")["draw"]
        assert even > lopsided

    def test_season_regression_pulls_toward_the_mean(self):
        elo = Elo(ratings={"A": 1700, "B": 1300})
        elo.regress_to_mean(0.5)
        assert elo.ratings["A"] == pytest.approx(1600)
        assert elo.ratings["B"] == pytest.approx(1400)

    def test_walk_forward_backtest_beats_a_coin_flip(self):
        rng = random.Random(42)
        strength = {"A": 0.75, "B": 0.5, "C": 0.25}
        teams = list(strength)
        games = []
        for _ in range(600):
            h, a = rng.sample(teams, 2)
            p = strength[h] / (strength[h] + strength[a])
            margin = 1 if rng.random() < p else -1
            games.append(GameResult(h, a, 1 if margin > 0 else 0,
                                    0 if margin > 0 else 1))
        result = Elo(EloConfig(k=20, home_advantage=0)).backtest(games)
        assert result["brier"] < 0.25  # better than always saying 50%

    def test_tune_k_returns_a_candidate(self):
        rng = random.Random(9)
        games = [
            GameResult("A", "B", 1, 0) if rng.random() < 0.7
            else GameResult("A", "B", 0, 1)
            for _ in range(200)
        ]
        best, scores = tune_k(games, candidates=(8, 20, 40))
        assert best in (8, 20, 40)
        assert len(scores) == 3


class TestDixonColes:
    @staticmethod
    def synthetic(n=600, seed=4):
        rng = random.Random(seed)
        strength = {"strong": 0.5, "mid": 0.0, "weak": -0.5}
        teams = list(strength)
        out = []
        start = date(2025, 1, 1)
        for i in range(n):
            h, a = rng.sample(teams, 2)
            lam = math.exp(0.3 + strength[h] * 1.0 + strength[a] * 0.5)
            mu = math.exp(0.1 + strength[a] * 1.0 + strength[h] * 0.5)
            out.append(
                Match(
                    h,
                    a,
                    _poisson_sample(rng, lam),
                    _poisson_sample(rng, mu),
                    when=start + timedelta(days=i // 4),
                )
            )
        return out

    def test_matrix_is_a_distribution(self):
        model = DixonColes().fit(self.synthetic(200))
        sl = model.predict("strong", "weak")
        assert sum(sum(row) for row in sl.matrix) == pytest.approx(1.0)
        assert all(v >= 0 for row in sl.matrix for v in row)

    def test_derived_markets_are_consistent(self):
        model = DixonColes().fit(self.synthetic(300))
        sl = model.predict("strong", "mid")
        odds1x2 = sl.match_odds()
        assert sum(odds1x2.values()) == pytest.approx(1.0)
        totals = sl.totals(2.5)
        assert totals["over"] + totals["under"] + totals["push"] == pytest.approx(1.0)
        assert totals["push"] == pytest.approx(0.0)  # half line cannot push
        whole = sl.totals(3.0)
        assert whole["push"] > 0
        ah = sl.asian_handicap(-0.5)
        assert ah["home"] + ah["away"] + ah["push"] == pytest.approx(1.0)
        btts = sl.both_teams_to_score()
        assert sum(btts.values()) == pytest.approx(1.0)

    def test_stronger_team_is_favoured(self):
        model = DixonColes().fit(self.synthetic(600))
        favoured = model.predict("strong", "weak").match_odds()
        assert favoured["home"] > favoured["away"]
        reversed_fixture = model.predict("weak", "strong").match_odds()
        assert reversed_fixture["away"] > reversed_fixture["home"]

    def test_recovers_the_ordering_of_team_strengths(self):
        model = DixonColes().fit(self.synthetic(800))
        ranked = [row["team"] for row in model.strengths()]
        assert ranked.index("strong") < ranked.index("weak")

    def test_home_advantage_is_positive(self):
        model = DixonColes().fit(self.synthetic(600))
        assert model.home_advantage > 0

    def test_xg_override_moves_the_line(self):
        model = DixonColes().fit(self.synthetic(200))
        base = model.predict("mid", "mid").match_odds()["home"]
        boosted = model.predict("mid", "mid", home_xg=4.0, away_xg=0.5).match_odds()["home"]
        assert boosted > base

    def test_needs_data(self):
        with pytest.raises(ValueError):
            DixonColes().fit([Match("a", "b", 1, 0)])

    def test_totals_line_ordering(self):
        model = DixonColes().fit(self.synthetic(200))
        sl = model.predict("strong", "weak")
        assert sl.totals(1.5)["over"] > sl.totals(4.5)["over"]


class TestPowerRatings:
    def test_recovers_synthetic_ratings(self):
        rng = random.Random(2)
        true = {"A": 7.0, "B": 0.0, "C": -7.0}
        teams = list(true)
        games = []
        for _ in range(900):
            h, a = rng.sample(teams, 2)
            margin = true[h] - true[a] + 2.5 + rng.gauss(0, 10)
            games.append(Game(h, a, max(0, round(20 + margin / 2)),
                              max(0, round(20 - margin / 2))))
        pr = PowerRatings(sport="nfl", ridge=1.0).fit(games)
        assert pr.rating("A") > pr.rating("B") > pr.rating("C")
        assert pr.rating("A") - pr.rating("C") == pytest.approx(14.0, abs=4.0)
        assert pr.home_advantage == pytest.approx(2.5, abs=2.0)

    def test_win_probability_tracks_the_margin(self):
        pr = PowerRatings(sport="nfl")
        pr.ratings = {"A": 7.0, "B": 0.0}
        pr.home_advantage = 2.0
        pr.sigma = 13.2
        assert pr.expected_margin("A", "B") == pytest.approx(9.0)
        assert 0.7 < pr.win_probability("A", "B") < 0.8
        assert pr.win_probability("A", "B") + pr.win_probability("B", "A", neutral=True) > 1

    def test_cover_probability_sums_to_one(self):
        pr = PowerRatings(sport="nfl")
        pr.ratings = {"A": 7.0, "B": 0.0}
        pr.home_advantage = 2.0
        result = pr.cover_probability("A", "B", -9.0)
        assert sum(result.values()) == pytest.approx(1.0)
        assert result["push"] > 0
        half = pr.cover_probability("A", "B", -9.5)
        assert half["push"] == 0.0

    def test_laying_the_exact_expected_margin_is_a_coin_flip(self):
        pr = PowerRatings(sport="nfl")
        pr.ratings = {"A": 5.0, "B": 0.0}
        pr.home_advantage = 0.0
        result = pr.cover_probability("A", "B", -5.5)
        assert result["home"] == pytest.approx(0.5, abs=0.02)

    def test_ridge_shrinks_a_thin_sample(self):
        games = [Game("A", "B", 50, 0)]  # one absurd result
        heavy = PowerRatings(ridge=50.0).fit(games + [Game("B", "A", 21, 20)])
        light = PowerRatings(ridge=0.5).fit(games + [Game("B", "A", 21, 20)])
        assert abs(heavy.rating("A")) < abs(light.rating("A"))

    def test_margin_cap_limits_blowout_influence(self):
        games = [Game("A", "B", 70, 0), Game("B", "A", 21, 20)]
        capped = PowerRatings(ridge=1.0).fit(games, cap_margin=21)
        uncapped = PowerRatings(ridge=1.0).fit(games)
        assert capped.rating("A") < uncapped.rating("A")

    def test_sigma_falls_back_to_the_sport_prior_on_thin_data(self):
        pr = PowerRatings(sport="nba").fit([Game("A", "B", 100, 90),
                                            Game("B", "A", 95, 99)])
        assert 5.0 < pr.sigma < 15.0


def _poisson_sample(rng: random.Random, lam: float) -> int:
    """Knuth's algorithm -- fine for the small rates used in these tests."""
    limit = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1
        if k > 40:
            return k
