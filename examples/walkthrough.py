"""End-to-end walkthrough. Run with: python3 examples/walkthrough.py

Builds a market, finds the soft price, sizes it, prices a parlay, fits a model,
and grades a fake record -- so you can see what each stage actually returns.
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta

from claudebet import (
    AnalysisConfig,
    Market,
    ParlayLeg,
    SettledBet,
    analyze,
    analyze_slate,
    backtest,
    devig_spread,
    parlay,
    weight_from_evidence,
)
from claudebet.devig import devig
from claudebet.models import DixonColes, Elo, PowerRatings
from claudebet.models.elo import EloConfig, GameResult
from claudebet.models.poisson import Match
from claudebet.models.ratings import Game


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ---------------------------------------------------------------- 1. devig
rule("1. The same prices, five different fair probabilities")

prices = [2.30, 3.45, 3.30]  # a 1X2 market at about 2.7% hold
for method in ("multiplicative", "additive", "power", "shin", "odds_ratio"):
    result = devig(prices, method)
    params = " ".join(f"{k}={v:.3f}" for k, v in result.params.items())
    print(f"  {method:<16} "
          + "  ".join(f"{p:6.2%}" for p in result.probs)
          + (f"   ({params})" if params else ""))
print(f"\n  Methods disagree by up to "
      f"{devig_spread(prices)['max_spread']:.2%} on this market.")
print("  An 'edge' smaller than that is an artefact of your devig choice.")


# ------------------------------------------------------- 2. market analysis
rule("2. Finding the soft price")

market = Market.from_book_prices(
    "Arsenal v Chelsea | 1X2",
    {
        "pinnacle": {"Arsenal": 2.30, "Draw": 3.45, "Chelsea": 3.30},
        "circa": {"Arsenal": 2.29, "Draw": 3.44, "Chelsea": 3.28},
        "betmgm": {"Arsenal": 2.34, "Draw": 3.30, "Chelsea": 3.15},
        "hardrock": {"Arsenal": 2.45, "Draw": 3.14, "Chelsea": 3.00},
    },
)
print(analyze(market, config=AnalysisConfig(min_edge=0.003, confidence_z=0.5),
              bankroll=5000).report())

print("\n  With the default (stricter) settings:")
strict = analyze(market, bankroll=5000)
print(f"  {len(strict.bets)} bets -- the 1.7-point edge does not survive the "
      "estimation-error haircut.")


# ------------------------------------------------------- 3. slate staking
rule("3. Six good bets are not six full-sized bets")

markets = []
for i in range(6):
    markets.append(
        Market.from_book_prices(
            f"game{i} | moneyline",
            {
                "pinnacle": {"home": 2.05, "away": 1.90},
                "circa": {"home": 2.06, "away": 1.89},
                "hardrock": {"home": 2.22, "away": 1.74},
            },
        )
    )
slate = analyze_slate(
    markets,
    bankroll=5000,
    config=AnalysisConfig(min_edge=0.005, confidence_z=0.5, max_slate_exposure=0.08),
)
solo = sum(r["standalone_fraction"] for r in slate["recommendations"])
print(f"  bets found:            {len(slate['recommendations'])}")
print(f"  sum of solo stakes:    {solo:.2%} of bankroll")
print(f"  jointly-sized total:   {slate['total_exposure']:.2%} of bankroll")
print(f"  expected log-growth:   {slate['expected_growth']:+.4%} per slate")
from claudebet.portfolio import Candidate, optimize

print("\n  How much does simultaneity actually cost? For *independent* bets,")
print("  less than the folklore suggests -- the second-order terms cancel:")
for n in (2, 6, 12):
    bets = [Candidate(f"g{i}", 0.55, 2.00) for i in range(n)]
    joint = optimize(bets, kelly_multiplier=1.0, max_exposure=1.0, max_per_bet=1.0)
    naive = 0.10 * n
    print(f"    {n:>2} bets at full Kelly: naive {naive:6.2%}  ->"
          f"  optimal {joint.total_exposure:6.2%}"
          f"  ({joint.total_exposure / naive:.0%} of naive)")
print("  The correction only bites once the naive total approaches the whole")
print("  bankroll. What it protects you from there is real: at 12 bets the")
print("  naive answer is 120% of bankroll, which is not a portfolio, it is a")
print("  margin call.")

print("\n  Correlation is where joint sizing earns its keep. Two outcomes of")
print("  one market cannot both lose, so the pair is safer than either leg:")
hedge = optimize(
    [Candidate("home", 0.45, 2.50, group="g"),
     Candidate("draw", 0.30, 3.60, group="g")],
    kelly_multiplier=1.0, max_exposure=1.0, max_per_bet=1.0,
)
for leg in ("home", "draw"):
    print(f"    {leg:<5} standalone {hedge.standalone[leg]:6.2%}"
          f"  ->  jointly optimal {hedge.stakes[leg]:6.2%}")
print("  Larger, not smaller -- the negative covariance is a genuine risk")
print("  reduction, and a naive per-bet Kelly cannot see it.")

print("\n  (Drop min_edge to 0.001 and this same slate reports twelve bets --")
print("   both sides of every game. That is what a threshold below the noise")
print("   floor buys you, and why the defaults sit where they do.)")


# ------------------------------------------------------------- 4. parlays
rule("4. Why the parlay is a tax, and when it is not")

legs = [
    ParlayLeg("Arsenal win", 0.425, 2.30),
    ParlayLeg("over 2.5 goals", 0.540, 1.80),
]
plain = parlay(legs)
correlated = parlay(legs, correlations={("Arsenal win", "over 2.5 goals"): 0.30})
print(f"  book's parlay price (legs multiplied)  {plain.offered_odds:.2f}")
print(f"  true probability if legs independent   {plain.independent_prob:.2%}"
      f"  -> EV {plain.ev_per_unit:+.2%}")
print(f"  true probability at +0.30 correlation  {correlated.fair_prob:.2%}"
      f"  -> EV {correlated.ev_per_unit:+.2%}")
print(f"  correlation is worth                   "
      f"{correlated.correlation_gain:+.2%} of probability")
print(f"  hold on the independent pricing        {plain.hold:.2%}")
print("\n  Read that second line first: multiply two vigged legs together and")
print("  you are paying roughly double the hold to win less often. The only")
print("  thing that flips it is correlation the book has failed to price --")
print("  which is why books apply their own same-game haircut, and why you")
print("  need a correlation estimate you can defend before taking one.")


# -------------------------------------------------------------- 5. models
rule("5. Models that give you a second opinion")

rng = random.Random(11)

# Elo, on results generated from known Elo ratings plus a home edge, so we can
# check what it recovers rather than just that it runs.
true_elo = {"Alpha": 1660, "Bravo": 1560, "Charlie": 1460, "Delta": 1360}
true_hfa = 40
teams = list(true_elo)
games = []
for _ in range(3000):
    h, a = rng.sample(teams, 2)
    diff = true_elo[h] - true_elo[a] + true_hfa
    home_win = rng.random() < 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
    games.append(GameResult(h, a, 1 if home_win else 0, 0 if home_win else 1))

elo = Elo(EloConfig(k=16, home_advantage=true_hfa, use_mov=False))
report = elo.backtest(games)
print(f"  Elo walk-forward over {report['n']} games: Brier {report['brier']:.4f}, "
      f"accuracy {report['accuracy']:.1%}")
centre = sum(true_elo.values()) / len(true_elo)
print("  recovered vs true (both centred on their own mean):")
for team, rating in elo.table():
    print(f"    {team:<9} {rating - 1500:+6.0f}   true {true_elo[team] - centre:+6.0f}")
print("  Elo is a random walk around the truth -- with K=16 and 3000 games the")
print("  ordering is stable but individual ratings still wander by tens of points.")

# Power ratings on synthetic margins.
truth = {"Alpha": 7.0, "Bravo": 2.0, "Charlie": -2.0, "Delta": -7.0}
margin_games = []
for _ in range(1200):
    h, a = rng.sample(teams, 2)
    margin = truth[h] - truth[a] + 2.5 + rng.gauss(0, 13)
    margin_games.append(
        Game(h, a, max(0, round(21 + margin / 2)), max(0, round(21 - margin / 2)))
    )
pr = PowerRatings(sport="nfl", ridge=2.0).fit(margin_games, cap_margin=28)
print("\n  Power ratings (true spread Alpha-Delta = 14.0):")
print("  " + ", ".join(f"{t} {r:+.1f}" for t, r in pr.table()))
print(f"  home advantage {pr.home_advantage:+.1f} (true 2.5), "
      f"sigma {pr.sigma:.1f}, R^2 {pr.r_squared:.3f}")
cover = pr.cover_probability("Alpha", "Delta", -13.5)
print(f"  Alpha -13.5 vs Delta: covers {cover['home']:.1%}")

# Dixon-Coles on synthetic scorelines.
def poisson_sample(lam: float) -> int:
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1
        if k > 30:
            return k

attack = {"City": 0.25, "United": 0.08, "Rovers": -0.08, "County": -0.25}
defence = {"City": 0.22, "United": 0.05, "Rovers": -0.05, "County": -0.22}
clubs = list(attack)
base, hfa = math.log(1.30), 0.20
matches = []
start = date(2025, 8, 1)
for i in range(900):
    h, a = rng.sample(clubs, 2)
    lam = math.exp(base + attack[h] - defence[a] + hfa)
    mu = math.exp(base + attack[a] - defence[h])
    matches.append(
        Match(h, a, poisson_sample(lam), poisson_sample(mu),
              when=start + timedelta(days=i // 4))
    )
dc = DixonColes().fit(matches)
sl = dc.predict("City", "County")
odds = sl.match_odds()
print(f"\n  Dixon-Coles, City v County (xG {sl.home_xg:.2f} - {sl.away_xg:.2f}):")
print(f"    1X2         {odds['home']:.1%} / {odds['draw']:.1%} / {odds['away']:.1%}")
print(f"    over 2.5    {sl.totals(2.5)['over']:.1%}")
print(f"    BTTS        {sl.both_teams_to_score()['yes']:.1%}")
print(f"    -1.5 AH     {sl.asian_handicap(-1.5)['home']:.1%}")
print("    top scores  "
      + ", ".join(f"{s} {p:.1%}" for s, p in sl.top_scorelines(4)))
print(f"    low-score correction rho = {dc.rho:+.3f}")


# ------------------------------------------------------- 6. how much weight
rule("6. How much should you trust your model?")

for n, skill in ((50, None), (400, 0.01), (2000, 0.02), (8000, 0.04)):
    w = weight_from_evidence(n, skill, market_confidence=0.9)
    label = "unproven" if skill is None else f"skill {skill:+.2f}"
    print(f"  {n:>5} graded bets, {label:<12} -> model weight {w:.1%}")
print("\n  A model with no out-of-sample record barely moves the market number,")
print("  and that is the correct amount for it to move it.")


# ------------------------------------------------------------ 7. grading
rule("7. Grading a record: CLV first, profit last")

rng = random.Random(3)
bets = []
for _ in range(500):
    # A genuine 2% edge: taken at 2.00, closes at 1.96.
    won = rng.random() < 0.51
    bets.append(
        SettledBet("pick", 2.00, 100, "win" if won else "loss", closing_odds=1.96)
    )
result = backtest(bets, starting_bankroll=10000)
print(result.summary())
print("\n  Note the confidence interval: after 500 bets at a real 2% edge, the")
print("  measured ROI could plausibly be anywhere in that range. CLV settled")
print("  the question in the same sample that left profit ambiguous.")
