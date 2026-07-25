import pytest

from claudebet import odds
from claudebet.devig import METHODS, devig, devig_spread, worst_case_prob


class TestConversions:
    def test_american_to_decimal_known_values(self):
        assert odds.american_to_decimal(100) == pytest.approx(2.0)
        assert odds.american_to_decimal(150) == pytest.approx(2.5)
        assert odds.american_to_decimal(-150) == pytest.approx(1 + 100 / 150)
        assert odds.american_to_decimal(-110) == pytest.approx(1.909090909)

    @pytest.mark.parametrize("american", [-500, -200, -110, -101, 100, 110, 250, 1000])
    def test_american_round_trip(self, american):
        d = odds.american_to_decimal(american)
        assert odds.decimal_to_american(d) == pytest.approx(american)

    def test_fractional(self):
        assert odds.fractional_to_decimal("5/2") == pytest.approx(3.5)
        assert odds.fractional_to_decimal("1/1") == pytest.approx(2.0)
        assert odds.decimal_to_fractional(3.5) == "5/2"

    def test_parse_odds_disambiguates(self):
        assert odds.parse_odds("+150") == pytest.approx(2.5)
        assert odds.parse_odds("-150") == pytest.approx(1 + 100 / 150)
        assert odds.parse_odds("2.5") == pytest.approx(2.5)
        assert odds.parse_odds(2.5) == pytest.approx(2.5)
        assert odds.parse_odds("5/2") == pytest.approx(3.5)
        assert odds.parse_odds("40%") == pytest.approx(2.5)
        assert odds.parse_odds(0.4) == pytest.approx(2.5)
        assert odds.parse_odds(-150) == pytest.approx(1 + 100 / 150)
        assert odds.parse_odds("evens") == pytest.approx(2.0)

    def test_parse_odds_rejects_nonsense(self):
        for bad in ["", "abc", 0.0, 1.0, -50, "0/1"]:
            with pytest.raises(ValueError):
                odds.parse_odds(bad)

    def test_overround_and_hold(self):
        pair = [1.909090909, 1.909090909]  # -110 both sides
        assert odds.overround(pair) == pytest.approx(1.047619, rel=1e-5)
        # Hold is the share of handle kept, always below the overround excess.
        assert odds.hold(pair) == pytest.approx(0.0454545, rel=1e-4)
        assert odds.hold(pair) < odds.overround(pair) - 1


class TestDevig:
    two_way = [1.909090909, 1.909090909]
    three_way = [2.30, 3.40, 3.30]
    longshot = [1.03, 15.0]  # heavy favourite / big dog, ~3.8% overround
    # A field wide enough that flat subtraction would push the longshot
    # negative -- the case that breaks a naive additive implementation.
    wide_field = [2.3, 3.8, 6.0, 10.0, 15.0, 25.0, 200.0]

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("market", [two_way, three_way, longshot])
    def test_probabilities_sum_to_one(self, method, market):
        result = devig(market, method)
        assert sum(result.probs) == pytest.approx(1.0, abs=1e-9)
        assert all(0.0 < p < 1.0 for p in result.probs)

    @pytest.mark.parametrize("method", METHODS)
    def test_fair_prob_below_booked(self, method):
        result = devig(self.three_way, method)
        booked = [1 / o for o in self.three_way]
        # Every fair probability must be no larger than the booked one, since
        # the whole point is removing margin.
        assert all(f <= b + 1e-9 for f, b in zip(result.probs, booked))

    def test_symmetric_market_gives_even_split(self):
        for method in METHODS:
            result = devig(self.two_way, method)
            assert result.probs[0] == pytest.approx(0.5, abs=1e-6)

    def test_shin_z_is_a_valid_proportion(self):
        result = devig(self.three_way, "shin")
        assert 0.0 <= result.params["z"] < 1.0

    def test_power_k_exceeds_one(self):
        result = devig(self.three_way, "power")
        assert result.params["k"] > 1.0

    def test_shin_shades_the_longshot_below_multiplicative(self):
        # The favourite-longshot correction is the entire reason to prefer
        # Shin; if this ever flips, the implementation is wrong.
        mult = devig(self.longshot, "multiplicative").probs[1]
        shin = devig(self.longshot, "shin").probs[1]
        power = devig(self.longshot, "power").probs[1]
        assert shin < mult
        assert power < mult

    def test_additive_survives_a_market_it_would_break(self):
        # The flat per-outcome charge here exceeds the 200.0 shot's booked
        # probability, so the unguarded formula would return a negative.
        surplus = (sum(1 / o for o in self.wide_field) - 1) / len(self.wide_field)
        assert surplus > 1 / self.wide_field[-1]
        result = devig(self.wide_field, "additive")
        assert all(p > 0 for p in result.probs)
        assert sum(result.probs) == pytest.approx(1.0)

    def test_no_vig_market_passes_through(self):
        result = devig([2.0, 2.0], "shin")
        assert result.overround == pytest.approx(1.0)
        assert result.probs == pytest.approx((0.5, 0.5))

    def test_hold_matches_odds_module(self):
        result = devig(self.two_way, "shin")
        assert result.hold == pytest.approx(odds.hold(self.two_way))

    def test_single_outcome_rejected(self):
        with pytest.raises(ValueError):
            devig([2.0])

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError):
            devig([2.0, 2.0], "vibes")

    def test_spread_grows_with_market_width(self):
        tight = devig_spread(self.two_way)["max_spread"]
        wide = devig_spread(self.longshot)["max_spread"]
        assert wide > tight

    def test_worst_case_is_the_most_pessimistic(self):
        market = [1.80, 2.10]
        wc = worst_case_prob(market, 0)
        for method in METHODS:
            assert wc <= devig(market, method).probs[0] + 1e-9

    def test_fair_odds_invert_the_probabilities(self):
        result = devig(self.three_way, "shin")
        for p, o in zip(result.probs, result.fair_odds):
            assert p * o == pytest.approx(1.0)
