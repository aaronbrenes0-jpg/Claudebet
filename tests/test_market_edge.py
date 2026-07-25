from datetime import datetime, timedelta, timezone

import pytest

from claudebet.edge import (
    cents_of_edge,
    evaluate,
    expected_roi,
    kelly_fraction,
    risk_of_drawdown,
)
from claudebet.market import Market, Quote, arbitrage, best_prices, consensus, movement
from claudebet.blend import blend, log_linear_pool, weight_from_evidence, shrink_toward


def make_market(prices, key="test", asof=None):
    return Market.from_book_prices(key, prices, asof=asof)


class TestConsensus:
    prices = {
        "pinnacle": {"A": 2.00, "B": 1.95},
        "draftkings": {"A": 2.10, "B": 1.83},
        "fanduel": {"A": 1.98, "B": 1.93},
    }

    def test_probabilities_sum_to_one(self):
        c = consensus(make_market(self.prices))
        assert sum(c.probs.values()) == pytest.approx(1.0)

    def test_sharp_book_dominates_the_pool(self):
        # Two soft books both leaning one way should not outvote Pinnacle.
        skewed = {
            "pinnacle": {"A": 2.00, "B": 1.95},
            "hardrock": {"A": 3.00, "B": 1.40},
            "espnbet": {"A": 3.00, "B": 1.40},
        }
        c = consensus(make_market(skewed))
        pin_only = consensus(make_market({"pinnacle": skewed["pinnacle"]}))
        soft_only = consensus(
            make_market({k: v for k, v in skewed.items() if k != "pinnacle"})
        )
        gap_to_pin = abs(c.probs["A"] - pin_only.probs["A"])
        gap_to_soft = abs(c.probs["A"] - soft_only.probs["A"])
        assert gap_to_pin < gap_to_soft

    def test_leave_one_out_changes_the_benchmark(self):
        market = make_market(self.prices)
        full = consensus(market)
        without = consensus(market, exclude="draftkings")
        assert without.probs["A"] != pytest.approx(full.probs["A"])
        assert without.n_books == full.n_books - 1
        assert without.excluded == ("draftkings",)

    def test_excluding_everything_raises(self):
        market = make_market({"pinnacle": {"A": 2.0, "B": 2.0}})
        with pytest.raises(ValueError):
            consensus(market, exclude="pinnacle")

    def test_incomplete_books_are_dropped(self):
        market = Market(
            key="k",
            outcomes=("A", "B"),
            quotes=[
                Quote("pinnacle", "A", 2.0),
                Quote("pinnacle", "B", 2.0),
                Quote("halfbook", "A", 5.0),  # only one side
            ],
        )
        c = consensus(market)
        assert c.n_books == 1
        assert "halfbook" not in c.per_book

    def test_dispersion_is_zero_when_books_agree(self):
        agreed = {"pinnacle": {"A": 2.0, "B": 2.0}, "circa": {"A": 2.0, "B": 2.0}}
        c = consensus(make_market(agreed))
        assert c.dispersion["A"] == pytest.approx(0.0, abs=1e-9)
        assert c.confidence() > 0

    def test_dispersion_rises_when_books_disagree(self):
        split = {"pinnacle": {"A": 1.5, "B": 3.0}, "circa": {"A": 3.0, "B": 1.5}}
        tight = consensus(make_market({"pinnacle": {"A": 2.0, "B": 2.0},
                                       "circa": {"A": 2.0, "B": 2.0}}))
        wide = consensus(make_market(split))
        assert wide.dispersion["A"] > tight.dispersion["A"]
        assert wide.confidence() < tight.confidence()

    def test_effective_books_penalises_concentration(self):
        c = consensus(make_market(self.prices))
        assert 1.0 <= c.effective_books <= c.n_books + 1e-9

    def test_stale_quotes_lose_weight(self):
        now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        market = Market(
            key="k",
            outcomes=("A", "B"),
            asof=now,
            quotes=[
                Quote("pinnacle", "A", 2.00, timestamp=now),
                Quote("pinnacle", "B", 2.00, timestamp=now),
                Quote("circa", "A", 3.00, timestamp=now - timedelta(hours=6)),
                Quote("circa", "B", 1.50, timestamp=now - timedelta(hours=6)),
            ],
        )
        c = consensus(market)
        assert c.weights["circa"] < c.weights["pinnacle"] / 10


class TestBestPriceAndArb:
    def test_best_price_picks_the_highest(self):
        market = make_market(
            {"a": {"X": 2.0, "Y": 2.0}, "b": {"X": 2.4, "Y": 1.7}}
        )
        best = best_prices(market)
        assert best["X"] == (2.4, "b")
        assert best["Y"] == (2.0, "a")

    def test_no_arb_in_a_normal_market(self):
        assert arbitrage(make_market({"a": {"X": 1.9, "Y": 1.9}})) is None

    def test_arb_detected_and_stakes_balance(self):
        market = make_market({"a": {"X": 2.2, "Y": 1.7}, "b": {"X": 1.7, "Y": 2.3}})
        arb = arbitrage(market)
        assert arb is not None
        assert arb["roi"] > 0
        assert sum(arb["stakes"].values()) == pytest.approx(1.0)
        # Every leg returns the same amount -- that is what makes it risk-free.
        returns = [
            arb["stakes"][o] * arb["legs"][o]["odds"] for o in market.outcomes
        ]
        assert max(returns) == pytest.approx(min(returns))


class TestMovement:
    def test_tracks_drift_and_flags_steam(self):
        t0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        snaps = [
            make_market({"pinnacle": {"A": 2.00, "B": 1.95}}, asof=t0),
            make_market({"pinnacle": {"A": 1.80, "B": 2.15}}, asof=t0 + timedelta(minutes=5)),
        ]
        result = movement(snaps)
        assert result["drift"]["A"] > 0  # price shortened, probability rose
        assert any(s["outcome"] == "A" for s in result["steam"])
        assert result["steam"][0]["direction"] == "shortening"

    def test_no_steam_when_nothing_moves(self):
        t0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        snaps = [
            make_market({"pinnacle": {"A": 2.0, "B": 2.0}}, asof=t0),
            make_market({"pinnacle": {"A": 2.0, "B": 2.0}}, asof=t0 + timedelta(minutes=5)),
        ]
        assert movement(snaps)["steam"] == []


class TestEdge:
    def test_kelly_matches_the_closed_form(self):
        # p=0.6 at even money: f* = (0.6*1 - 0.4)/1 = 0.2
        assert kelly_fraction(0.6, 2.0) == pytest.approx(0.2)
        assert kelly_fraction(0.5, 2.0) == pytest.approx(0.0)
        assert kelly_fraction(0.4, 2.0) == pytest.approx(-0.2)

    def test_expected_roi(self):
        assert expected_roi(0.6, 2.0) == pytest.approx(0.2)
        assert expected_roi(0.5, 2.0) == pytest.approx(0.0)

    def test_fractional_kelly_scales_the_stake(self):
        full = evaluate(0.60, 2.0, kelly_multiplier=1.0, max_stake_fraction=1.0)
        quarter = evaluate(0.60, 2.0, kelly_multiplier=0.25, max_stake_fraction=1.0)
        assert quarter.stake_fraction == pytest.approx(full.stake_fraction / 4)

    def test_uncertainty_haircut_shrinks_the_edge(self):
        certain = evaluate(0.55, 2.0, prob_stderr=0.0)
        uncertain = evaluate(0.55, 2.0, prob_stderr=0.25)
        assert uncertain.used_prob < certain.used_prob
        assert uncertain.stake_fraction < certain.stake_fraction

    def test_haircut_can_kill_a_marginal_bet(self):
        result = evaluate(0.52, 2.0, prob_stderr=0.30, min_edge=0.01)
        assert result.verdict != "bet"
        assert "estimation error" in result.reason or "threshold" in result.reason

    def test_negative_edge_is_refused(self):
        result = evaluate(0.45, 2.0)
        assert result.verdict == "no-edge"
        assert result.stake_fraction == 0.0

    def test_stake_cap_binds(self):
        # A big edge at long odds asks for far too much; the cap must hold.
        result = evaluate(0.50, 5.0, max_stake_fraction=0.02, kelly_multiplier=1.0)
        assert result.kelly_full > 0.02
        assert result.stake_fraction == pytest.approx(0.02)

    def test_stake_scales_with_bankroll(self):
        result = evaluate(0.60, 2.0, bankroll=5000)
        assert result.stake == pytest.approx(result.stake_fraction * 5000)

    def test_edge_in_cents_is_zero_at_a_fair_price(self):
        assert evaluate(0.5, 2.0).edge_cents == pytest.approx(0.0, abs=1e-6)

    def test_cents_of_edge_is_sane_either_side_of_even_money(self):
        # American odds jump from -100 to +100 with nothing in between, so a
        # naive subtraction across evens overstates the gap by exactly 200.
        # 2.05 against a fair 1.9865 is about six cents, not 206.
        assert cents_of_edge(0.5034, 2.05) == pytest.approx(6.4, abs=0.3)
        # Both sides negative: plain American difference, no crossing.
        assert cents_of_edge(1 / 1.5701, 1.62) == pytest.approx(14.1, abs=0.3)
        # Both sides positive.
        assert cents_of_edge(1 / 2.60, 2.70) == pytest.approx(10.0, abs=0.3)

    def test_cents_of_edge_grows_smoothly_across_even_money(self):
        # Sweep a price across the boundary; the measure must not jump.
        previous = None
        for price in [1.94 + 0.01 * i for i in range(13)]:
            value = cents_of_edge(0.5, price)
            if previous is not None:
                assert abs(value - previous) < 3.0
            previous = value

    def test_drawdown_risk_falls_with_smaller_kelly(self):
        full = risk_of_drawdown(0.02, 1.0, 1.0, 0.5)
        quarter = risk_of_drawdown(0.02, 1.0, 0.25, 0.5)
        assert full == pytest.approx(0.5)
        assert quarter < 0.05


class TestBlend:
    def test_weight_zero_returns_the_market(self):
        market = {"A": 0.4, "B": 0.6}
        assert blend({"A": 0.9, "B": 0.1}, market, 0.0) == pytest.approx(market)

    def test_weight_one_returns_the_model(self):
        model = {"A": 0.9, "B": 0.1}
        assert blend(model, {"A": 0.4, "B": 0.6}, 1.0) == pytest.approx(model)

    def test_blend_lands_between_and_normalises(self):
        out = blend({"A": 0.8, "B": 0.2}, {"A": 0.4, "B": 0.6}, 0.5)
        assert 0.4 < out["A"] < 0.8
        assert sum(out.values()) == pytest.approx(1.0)

    def test_pooling_is_symmetric_in_log_odds(self):
        # Equal weights on mirrored opinions must land exactly in the middle.
        out = log_linear_pool([{"A": 0.7, "B": 0.3}, {"A": 0.3, "B": 0.7}], [1, 1])
        assert out["A"] == pytest.approx(0.5)

    def test_evidence_weight_grows_with_sample(self):
        small = weight_from_evidence(50, 0.03)
        large = weight_from_evidence(5000, 0.03)
        assert 0 < small < large <= 0.45

    def test_no_skill_means_no_weight(self):
        assert weight_from_evidence(10000, calibration_score=0.0) == 0.0
        assert weight_from_evidence(10000, calibration_score=-0.01) == 0.0

    def test_unproven_model_gets_a_small_weight(self):
        assert 0 < weight_from_evidence(100, None) < 0.2

    def test_shrink_toward_anchor(self):
        assert shrink_toward(0.8, 0.5, 1.0) == pytest.approx(0.5)
        assert shrink_toward(0.8, 0.5, 0.0) == pytest.approx(0.8)
        assert 0.5 < shrink_toward(0.8, 0.5, 0.5) < 0.8
