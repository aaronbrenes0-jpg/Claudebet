"""Grading a betting record honestly.

Profit is a terrible short-run measure of a betting strategy. At a 2% edge the
standard deviation of a 500-bet sample is around 4-5% of turnover, so two years
of genuinely good work can show a loss, and a bad strategy can show a profit for
just as long. Judge on these, in order:

1. **Closing line value.** Did you consistently get a better price than the
   market's final one? CLV converges roughly an order of magnitude faster than
   profit and it is the closest thing to a leading indicator that exists. If
   you beat the close by 2% on average, you are a winner and the results will
   catch up. If you do not, a profit so far was luck.
2. **Drawdown against the plan.** Whether the bankroll path stayed inside what
   your staking scheme implies.
3. **Profit**, last, with a confidence interval attached -- and this module
   refuses to report a point estimate of ROI without one.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Sequence

from .odds import decimal_to_prob, parse_odds

__all__ = ["SettledBet", "BacktestResult", "run", "clv_report"]

_RESULTS = ("win", "loss", "push", "void", "half-win", "half-loss")


@dataclass
class SettledBet:
    """One graded bet.

    ``closing_odds`` is the price on the same selection when the market closed,
    at the same book or the sharpest one you track. Supplying it unlocks the
    CLV analysis, which is the most valuable output here -- record it even when
    it is inconvenient.
    """

    outcome: str
    odds: float
    stake: float
    result: str
    timestamp: datetime | None = None
    market: str = ""
    closing_odds: float | None = None
    fair_prob: float | None = None  # your estimate when you bet
    closing_fair_prob: float | None = None  # devigged closing consensus
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.odds = parse_odds(self.odds)
        if self.closing_odds is not None:
            self.closing_odds = parse_odds(self.closing_odds)
        self.result = str(self.result).strip().lower()
        if self.result not in _RESULTS:
            raise ValueError(f"result must be one of {_RESULTS}, got {self.result!r}")
        if self.stake < 0:
            raise ValueError("stake cannot be negative")

    @property
    def profit(self) -> float:
        b = self.odds - 1.0
        return {
            "win": self.stake * b,
            "loss": -self.stake,
            "push": 0.0,
            "void": 0.0,
            "half-win": self.stake * b / 2.0,
            "half-loss": -self.stake / 2.0,
        }[self.result]

    @property
    def settled(self) -> bool:
        return self.result not in ("push", "void")

    def clv(self) -> float | None:
        """Closing line value in probability points.

        The difference between the closing market's implied probability and
        the one you were paid at. Positive means you got the better number.
        Both sides are devigged crudely (a flat half-hold assumption) when only
        raw odds are available; supply ``closing_fair_prob`` for an exact
        figure.
        """
        if self.closing_fair_prob is not None and self.fair_prob is not None:
            return self.closing_fair_prob - decimal_to_prob(self.odds)
        if self.closing_odds is None:
            return None
        return decimal_to_prob(self.closing_odds) - decimal_to_prob(self.odds)

    def clv_cents(self) -> float | None:
        """CLV expressed as a percentage price improvement, the way a trader
        would read it: ``odds_taken / closing_odds - 1``."""
        if self.closing_odds is None:
            return None
        return self.odds / self.closing_odds - 1.0


@dataclass
class BacktestResult:
    n_bets: int
    n_settled: int
    turnover: float
    profit: float
    roi: float
    win_rate: float
    bankroll_path: list[float]
    final_bankroll: float
    max_drawdown: float
    max_drawdown_pct: float
    longest_losing_streak: int
    roi_ci95: tuple[float, float]
    roi_t_stat: float
    clv: dict = field(default_factory=dict)
    by_tag: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"{self.n_bets} bets, {self.turnover:.2f} staked",
            f"profit {self.profit:+.2f}  ROI {self.roi:+.2%} "
            f"(95% CI {self.roi_ci95[0]:+.2%} to {self.roi_ci95[1]:+.2%})",
            f"bankroll {self.bankroll_path[0]:.2f} -> {self.final_bankroll:.2f}, "
            f"max drawdown {self.max_drawdown_pct:.1%}",
        ]
        if self.clv:
            lines.append(
                f"CLV: beat the close {self.clv['beat_rate']:.1%} of the time, "
                f"mean {self.clv['mean_pp']:+.2%} in probability points"
            )
        verdict = self.verdict()
        lines.append(f"verdict: {verdict}")
        return "\n".join(lines)

    def verdict(self) -> str:
        if self.clv and self.clv.get("n", 0) >= 50:
            mean_pp = self.clv["mean_pp"]
            if mean_pp > 0.01:
                return (
                    "beating the closing line convincingly; the edge looks real "
                    "regardless of the profit column"
                )
            if mean_pp > 0.002:
                return "slightly ahead of the closing line; promising, keep going"
            return (
                "not beating the closing line; any profit here is most likely "
                "variance, not edge"
            )
        if self.n_settled < 200:
            return f"only {self.n_settled} settled bets: no conclusion is available yet"
        if self.roi_ci95[0] > 0:
            return "profitable with the confidence interval clear of zero"
        return "no statistically meaningful edge demonstrated"


def _bootstrap_roi(
    profits: Sequence[float], stakes: Sequence[float], draws: int, seed: int
) -> tuple[float, float]:
    if not profits:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(profits)
    rois: list[float] = []
    idx = range(n)
    for _ in range(draws):
        picks = [rng.choice(idx) for _ in range(n)]
        staked = sum(stakes[i] for i in picks)
        if staked <= 0:
            continue
        rois.append(sum(profits[i] for i in picks) / staked)
    if not rois:
        return (0.0, 0.0)
    rois.sort()
    lo = rois[max(0, int(0.025 * len(rois)) - 1)]
    hi = rois[min(len(rois) - 1, int(0.975 * len(rois)))]
    return (lo, hi)


def clv_report(bets: Iterable[SettledBet]) -> dict:
    """Closing line value across a record. The headline number is
    ``mean_pp``: your average edge over the closing price, in probability
    points. Sustained values above 1% are strong."""
    values = []
    pct = []
    for bet in bets:
        v = bet.clv()
        if v is not None:
            values.append(v)
        p = bet.clv_cents()
        if p is not None:
            pct.append(p)
    if not values:
        return {}
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / max(1, n - 1)
    se = math.sqrt(var / n) if n > 1 else 0.0
    return {
        "n": n,
        "mean_pp": mean,
        "median_pp": sorted(values)[n // 2],
        "beat_rate": sum(1 for v in values if v > 0) / n,
        "stderr": se,
        "t_stat": mean / se if se > 0 else 0.0,
        "mean_price_improvement": (sum(pct) / len(pct)) if pct else None,
    }


def run(
    bets: Sequence[SettledBet],
    starting_bankroll: float = 1000.0,
    *,
    restake_fractionally: bool = False,
    bootstrap_draws: int = 2000,
    seed: int = 7,
) -> BacktestResult:
    """Replay a record and measure it.

    ``restake_fractionally``
        With ``False`` (default) each bet's recorded stake is used as-is, which
        is what you want for grading what actually happened. With ``True`` the
        stakes are treated as *fractions of the bankroll at the time* and
        compound, which is what you want for asking "what would Kelly have
        done" -- and note the ordering then matters, so timestamps should be
        present and correct.
    """
    ordered = sorted(
        bets, key=lambda b: (b.timestamp is None, b.timestamp or datetime.min)
    )
    bankroll = float(starting_bankroll)
    path = [bankroll]
    profits: list[float] = []
    stakes: list[float] = []
    peak = bankroll
    max_dd = 0.0
    streak = worst_streak = 0
    wins = 0
    settled = 0

    for bet in ordered:
        if restake_fractionally:
            stake = bet.stake * bankroll
            scaled = SettledBet(
                outcome=bet.outcome,
                odds=bet.odds,
                stake=stake,
                result=bet.result,
                timestamp=bet.timestamp,
                market=bet.market,
                closing_odds=bet.closing_odds,
                fair_prob=bet.fair_prob,
                closing_fair_prob=bet.closing_fair_prob,
                tags=bet.tags,
            )
            pnl = scaled.profit
        else:
            stake = bet.stake
            pnl = bet.profit

        bankroll += pnl
        path.append(bankroll)
        profits.append(pnl)
        stakes.append(stake)

        if bet.settled:
            settled += 1
            if pnl > 0:
                wins += 1
                streak = 0
            else:
                streak += 1
                worst_streak = max(worst_streak, streak)

        peak = max(peak, bankroll)
        max_dd = max(max_dd, peak - bankroll)

    turnover = sum(stakes)
    profit = sum(profits)
    roi = profit / turnover if turnover > 0 else 0.0

    if len(profits) > 1 and turnover > 0:
        per_unit = [
            p / s for p, s in zip(profits, stakes) if s > 0
        ]  # ROI contribution per bet
        m = sum(per_unit) / len(per_unit)
        var = sum((x - m) ** 2 for x in per_unit) / (len(per_unit) - 1)
        se = math.sqrt(var / len(per_unit)) if var > 0 else 0.0
        t_stat = m / se if se > 0 else 0.0
    else:
        t_stat = 0.0

    ci = _bootstrap_roi(profits, stakes, bootstrap_draws, seed)

    by_tag: dict[str, dict] = {}
    for bet, pnl, stake in zip(ordered, profits, stakes):
        for tag in bet.tags:
            entry = by_tag.setdefault(tag, {"n": 0, "staked": 0.0, "profit": 0.0})
            entry["n"] += 1
            entry["staked"] += stake
            entry["profit"] += pnl
    for entry in by_tag.values():
        entry["roi"] = entry["profit"] / entry["staked"] if entry["staked"] else 0.0

    return BacktestResult(
        n_bets=len(ordered),
        n_settled=settled,
        turnover=turnover,
        profit=profit,
        roi=roi,
        win_rate=wins / settled if settled else 0.0,
        bankroll_path=path,
        final_bankroll=bankroll,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd / peak if peak > 0 else 0.0,
        longest_losing_streak=worst_streak,
        roi_ci95=ci,
        roi_t_stat=t_stat,
        clv=clv_report(ordered),
        by_tag=by_tag,
    )
