"""Turning a probability estimate and a price into a decision and a stake.

Expected value is the easy half. The hard half is that your probability is an
estimate with error, and the Kelly criterion is savagely sensitive to that
error: overestimate your edge by a factor of two and full Kelly does not halve
your growth, it can turn it negative. Everything here is built to be
conservative about that.

Three defences, all on by default:

1. **Uncertainty haircut.** You supply (or the pipeline derives) a standard
   error on the probability, in log-odds. Staking uses a lower confidence bound
   on the probability, not the point estimate.
2. **Fractional Kelly.** A fixed fraction -- 0.25 by default -- of the already
   hairclipped Kelly stake. This costs a quarter of the theoretical growth rate
   and removes most of the drawdown.
3. **A minimum edge.** Below it, the bet is declined outright, because at small
   measured edges the probability that the true edge is negative is large.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

from .blend import expit, logit
from .odds import decimal_to_american, decimal_to_prob, parse_odds

__all__ = [
    "BetEvaluation",
    "expected_value",
    "expected_roi",
    "kelly_fraction",
    "evaluate",
    "risk_of_drawdown",
    "cents_of_edge",
    "summarize",
]


def expected_value(prob: float, odds: float, stake: float = 1.0) -> float:
    """Expected profit in units of currency for a given stake."""
    d = parse_odds(odds)
    return stake * (prob * (d - 1.0) - (1.0 - prob))


def expected_roi(prob: float, odds: float) -> float:
    """Expected profit per unit staked. 0.03 means a 3% return on turnover."""
    return prob * parse_odds(odds) - 1.0


def kelly_fraction(prob: float, odds: float) -> float:
    """Full-Kelly stake as a fraction of bankroll. Negative means lay/pass."""
    d = parse_odds(odds)
    b = d - 1.0
    return (prob * b - (1.0 - prob)) / b


def cents_of_edge(fair_prob: float, odds: float) -> float:
    """Edge expressed in American cents, the way traders quote it.

    The distance between the price you got and the fair price for the same
    probability. Roughly: 20 cents on a -110 market is a very large edge; 5-10
    cents is a normal good bet; under 3 is inside the noise.
    """
    d = parse_odds(odds)
    fair_d = 1.0 / max(min(fair_prob, 1 - 1e-9), 1e-9)
    return abs(decimal_to_american(d) - decimal_to_american(fair_d))


@dataclass(frozen=True)
class BetEvaluation:
    """Everything needed to decide on a single price."""

    outcome: str
    odds: float
    fair_prob: float
    used_prob: float  # after the uncertainty haircut
    prob_stderr: float  # in log-odds
    breakeven_prob: float
    edge_pp: float  # fair_prob - breakeven, in probability points
    edge_cents: float
    ev_per_unit: float  # expected profit per unit staked, at used_prob
    ev_point_estimate: float  # same, at fair_prob (no haircut)
    kelly_full: float
    stake_fraction: float  # of bankroll, after fraction + caps
    stake: float  # in currency, if a bankroll was supplied
    verdict: str  # "bet" | "pass" | "no-edge"
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def american(self) -> float:
        return decimal_to_american(self.odds)


def evaluate(
    fair_prob: float,
    odds: float,
    *,
    outcome: str = "",
    bankroll: float = 0.0,
    prob_stderr: float = 0.0,
    kelly_multiplier: float = 0.25,
    confidence_z: float = 1.0,
    min_edge: float = 0.01,
    max_stake_fraction: float = 0.02,
    min_stake: float = 0.0,
) -> BetEvaluation:
    """Score one price against one fair probability.

    ``prob_stderr``
        Standard error of the probability estimate *in log-odds*. Market
        dispersion (:attr:`claudebet.market.Consensus.dispersion`) is a decent
        empirical proxy; 0.10-0.20 is typical for a liquid market.
    ``confidence_z``
        How many standard errors to give back before staking. 1.0 stakes off
        roughly the 16th percentile of your probability estimate.
    ``min_edge``
        Minimum edge in probability points, measured *after* the haircut.
    ``max_stake_fraction``
        Hard ceiling on any single bet as a share of bankroll, applied last.
        Kelly on a genuine 8% edge at long odds will happily suggest far more
        than you should ever have on one game.
    """
    d = parse_odds(odds)
    p = min(max(float(fair_prob), 1e-9), 1.0 - 1e-9)
    breakeven = decimal_to_prob(d)

    # Haircut: walk the probability down by z standard errors in log-odds.
    se = max(0.0, float(prob_stderr))
    used = expit(logit(p) - confidence_z * se) if se > 0 else p

    edge_pp = p - breakeven
    ev_point = expected_roi(p, d)
    ev_used = expected_roi(used, d)
    k_full = kelly_fraction(used, d)
    edge_after_haircut = used - breakeven

    verdict, reason = "bet", "positive expected value after haircut"
    if edge_after_haircut <= 0:
        verdict = "no-edge"
        reason = (
            "price is worse than fair"
            if edge_pp <= 0
            else "edge disappears once estimation error is priced in"
        )
    elif edge_after_haircut < min_edge:
        verdict = "pass"
        reason = f"edge {edge_after_haircut:.3%} below the {min_edge:.3%} threshold"

    if verdict == "bet":
        fraction = max(0.0, k_full) * float(kelly_multiplier)
        fraction = min(fraction, float(max_stake_fraction))
    else:
        fraction = 0.0

    stake = fraction * float(bankroll)
    if stake > 0 and min_stake > 0 and stake < min_stake:
        # Too small to be worth the ticket; do not round up into a worse bet.
        stake = 0.0
        fraction = 0.0
        if verdict == "bet":
            verdict = "pass"
            reason = f"stake {fraction * bankroll:.2f} below minimum {min_stake:.2f}"

    return BetEvaluation(
        outcome=outcome,
        odds=d,
        fair_prob=p,
        used_prob=used,
        prob_stderr=se,
        breakeven_prob=breakeven,
        edge_pp=edge_pp,
        edge_cents=cents_of_edge(p, d),
        ev_per_unit=ev_used,
        ev_point_estimate=ev_point,
        kelly_full=k_full,
        stake_fraction=fraction,
        stake=stake,
        verdict=verdict,
        reason=reason,
    )


def risk_of_drawdown(
    edge: float, variance: float, fraction_of_kelly: float, depth: float = 0.5
) -> float:
    """Probability of ever losing ``depth`` of the bankroll under continuous
    fractional-Kelly betting.

    Uses the standard diffusion result ``P = depth ** (2/f - 1)`` for a
    fraction ``f`` of full Kelly, which is independent of the edge itself --
    a genuinely useful fact. Quarter Kelly gives roughly a 3% chance of ever
    halving; full Kelly gives 50%.

    ``edge`` and ``variance`` are accepted for interface completeness and to
    reject degenerate inputs; they cancel out of the result.
    """
    f = float(fraction_of_kelly)
    if not 0 < f <= 1:
        raise ValueError("fraction_of_kelly must be in (0, 1]")
    if variance <= 0:
        raise ValueError("variance must be positive")
    if not 0 < depth < 1:
        raise ValueError("depth must be in (0, 1)")
    exponent = (2.0 / f) - 1.0
    return float(depth**exponent)


def summarize(evals: Sequence[BetEvaluation]) -> dict:
    """Roll a slate of evaluations into a portfolio-level summary."""
    bets = [e for e in evals if e.verdict == "bet"]
    exposure = sum(e.stake_fraction for e in bets)
    return {
        "n_candidates": len(evals),
        "n_bets": len(bets),
        "total_exposure_fraction": exposure,
        "expected_return_fraction": sum(
            e.stake_fraction * e.ev_per_unit for e in bets
        ),
        "mean_edge_pp": (sum(e.edge_pp for e in bets) / len(bets)) if bets else 0.0,
        "mean_edge_cents": (
            sum(e.edge_cents for e in bets) / len(bets) if bets else 0.0
        ),
    }
