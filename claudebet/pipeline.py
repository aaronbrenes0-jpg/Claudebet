"""The end-to-end path: prices in, ranked bets out.

This is the module to use if you only use one. It does, in order:

1. Devig every book on the market and pool them into a consensus fair price.
2. For each individual price, rebuild that consensus *without the book quoting
   it*, so a soft book's outlier is measured against the rest of the market
   rather than partly against itself.
3. Optionally blend in your own model, with a weight derived from how much
   out-of-sample evidence the model has earned.
4. Derive an error bar from how much the books disagree and how much the devig
   methods disagree, and take that error out of the probability before staking.
5. Size with fractional Kelly under a per-bet and per-slate exposure cap.

Step 4 is the one that stops this from being a bet-generating machine. Most of
what looks like edge on a screen is one book being slow, one devig assumption,
or one thin market -- and a haircut sized from the market's own disagreement
kills most of those without needing to identify which is which.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .blend import blend as blend_distributions
from .devig import devig_spread, worst_case_prob
from .edge import BetEvaluation, evaluate
from .market import Consensus, Market, arbitrage, best_prices, consensus, line_shopping_gain
from .portfolio import Candidate, Allocation, optimize

__all__ = ["AnalysisConfig", "Opportunity", "MarketAnalysis", "analyze", "analyze_slate"]


@dataclass
class AnalysisConfig:
    """Every knob in one place. The defaults are deliberately conservative."""

    devig_method: str = "shin"
    model_weight: float = 0.0  # 0 = pure market; see blend.weight_from_evidence
    kelly_multiplier: float = 0.25
    min_edge: float = 0.015  # 1.5 probability points after the haircut
    max_stake_fraction: float = 0.02
    max_slate_exposure: float = 0.10
    confidence_z: float = 1.0
    min_books: int = 3  # below this the consensus is not a consensus
    extra_stderr: float = 0.05  # floor on estimate error, in log-odds
    exclude_own_book: bool = True
    require_positive_worst_case: bool = False  # survive the harshest devig too
    sharpness: Mapping[str, float] | None = None


@dataclass
class Opportunity:
    """One bettable price, fully diagnosed."""

    market: str
    book: str
    outcome: str
    evaluation: BetEvaluation
    consensus_prob: float
    model_prob: float | None
    blended_prob: float
    worst_case_prob: float
    dispersion: float
    n_books_in_benchmark: int
    notes: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return self.evaluation.verdict

    @property
    def edge(self) -> float:
        return self.evaluation.edge_pp

    def describe(self) -> str:
        e = self.evaluation
        head = (
            f"{self.outcome} @ {e.odds:.3f} ({e.american:+.0f}) at {self.book}: "
            f"fair {self.blended_prob:.2%} vs breakeven {e.breakeven_prob:.2%}"
        )
        body = (
            f"  edge {e.edge_pp:+.2%} ({e.edge_cents:.1f} cents), "
            f"EV {e.ev_per_unit:+.2%} per unit, stake {e.stake_fraction:.3%} of bankroll"
        )
        tail = f"  {e.verdict}: {e.reason}"
        extra = "".join(f"\n  note: {n}" for n in self.notes)
        return f"{head}\n{body}\n{tail}{extra}"


@dataclass
class MarketAnalysis:
    market: str
    consensus: Consensus
    fair_probs: dict[str, float]
    best: dict[str, tuple[float, str]]
    opportunities: list[Opportunity]
    arbitrage: dict | None
    shopping_gain: dict[str, float]
    devig_sensitivity: dict
    warnings: list[str] = field(default_factory=list)

    @property
    def bets(self) -> list[Opportunity]:
        return [o for o in self.opportunities if o.verdict == "bet"]

    def report(self) -> str:
        lines = [f"== {self.market} =="]
        if self.warnings:
            lines += [f"!  {w}" for w in self.warnings]
        lines.append(
            f"consensus from {self.consensus.n_books} books "
            f"(effective {self.consensus.effective_books:.1f}), "
            f"mean hold {self.consensus.mean_hold:.2%}, "
            f"confidence {self.consensus.confidence():.2f}"
        )
        for outcome, prob in self.fair_probs.items():
            price, book = self.best.get(outcome, (float("nan"), "-"))
            lines.append(
                f"  {outcome:<24} fair {prob:6.2%}  "
                f"best {price:6.3f} ({book})  "
                f"disagreement {self.consensus.dispersion.get(outcome, 0):.3f}"
            )
        if self.arbitrage:
            lines.append(
                f"  ARBITRAGE: {self.arbitrage['roi']:+.2%} risk-free across "
                + ", ".join(
                    f"{o} at {leg['book']}"
                    for o, leg in self.arbitrage["legs"].items()
                )
            )
        bets = self.bets
        if bets:
            lines.append("  bets:")
            lines += ["    " + line for b in bets for line in b.describe().split("\n")]
        else:
            lines.append("  no bet: nothing clears the edge threshold")
        return "\n".join(lines)


def _stderr_for(
    outcome: str, cons: Consensus, sensitivity: dict, index: int, config: AnalysisConfig
) -> float:
    """Standard error of the fair probability, in log-odds.

    Three sources, combined in quadrature because they are largely independent:
    how much the books disagree, how much the devig method choice matters, and
    a floor for everything we have not modelled (stale quotes, injuries not yet
    priced, our own bugs).
    """
    book_disagreement = cons.dispersion.get(outcome, 0.0)
    # Convert the devig spread from probability points into log-odds at the
    # current level, so it means the same thing at 5% as at 50%.
    p = min(max(cons.probs.get(outcome, 0.5), 1e-6), 1 - 1e-6)
    spread_pp = sensitivity["outcomes"][index]["spread"] if sensitivity else 0.0
    method_disagreement = spread_pp / (p * (1 - p)) if p * (1 - p) > 0 else 0.0

    # Thin consensus: inflate the error rather than refusing to answer.
    thinness = 1.0 / math.sqrt(max(1.0, cons.effective_books))

    return math.sqrt(
        (book_disagreement * thinness) ** 2
        + (0.5 * method_disagreement) ** 2
        + config.extra_stderr**2
    )


def analyze(
    market: Market,
    model_probs: Mapping[str, float] | None = None,
    config: AnalysisConfig | None = None,
    bankroll: float = 0.0,
) -> MarketAnalysis:
    """Full analysis of one market across every book quoting it."""
    cfg = config or AnalysisConfig()
    warnings: list[str] = []

    overall = consensus(market, method=cfg.devig_method, sharpness=cfg.sharpness)
    if overall.n_books < cfg.min_books:
        warnings.append(
            f"only {overall.n_books} book(s) quote a complete market; the fair "
            "price is barely better than a guess and stakes are cut accordingly"
        )

    fair = dict(overall.probs)
    if model_probs:
        missing = set(market.outcomes) - set(model_probs)
        if missing:
            warnings.append(f"model has no opinion on {sorted(missing)}; market used")
        else:
            fair = blend_distributions(
                dict(model_probs), overall.probs, cfg.model_weight
            )

    # Devig sensitivity is computed on the sharpest complete book, since that
    # is the one carrying most of the consensus weight.
    table = market.by_book(complete_only=True)
    sensitivity: dict = {}
    if table:
        sharpest = max(table, key=lambda b: overall.weights.get(b, 0.0))
        sensitivity = devig_spread([table[sharpest][o] for o in market.outcomes])
        if sensitivity["max_spread"] > 0.03:
            warnings.append(
                f"devig methods disagree by up to {sensitivity['max_spread']:.1%} on "
                "this market; treat small edges as unmeasurable"
            )

    opportunities: list[Opportunity] = []
    for quote in market.quotes:
        if quote.outcome not in market.outcomes:
            continue
        index = market.outcomes.index(quote.outcome)

        # Leave-one-out benchmark.
        bench = overall
        if cfg.exclude_own_book:
            try:
                bench = consensus(
                    market,
                    method=cfg.devig_method,
                    sharpness=cfg.sharpness,
                    exclude=quote.book,
                )
            except ValueError:
                bench = overall

        bench_prob = bench.probs.get(quote.outcome, fair[quote.outcome])
        if model_probs and set(model_probs) >= set(market.outcomes):
            blended = blend_distributions(
                dict(model_probs), bench.probs, cfg.model_weight
            )[quote.outcome]
        else:
            blended = bench_prob

        stderr = _stderr_for(quote.outcome, bench, sensitivity, index, cfg)

        notes: list[str] = []
        if cfg.exclude_own_book and bench is overall:
            notes.append("could not exclude this book from the benchmark")
        if bench.n_books < cfg.min_books:
            notes.append(f"benchmark rests on only {bench.n_books} book(s)")

        wc = float("nan")
        if quote.book in table:
            wc = worst_case_prob([table[quote.book][o] for o in market.outcomes], index)
            if cfg.require_positive_worst_case and wc * quote.odds <= 1.0:
                notes.append(
                    "fails the worst-case devig test: the edge depends on how "
                    "the vig is assumed to be distributed"
                )

        ev = evaluate(
            blended,
            quote.odds,
            outcome=quote.outcome,
            bankroll=bankroll,
            prob_stderr=stderr,
            kelly_multiplier=cfg.kelly_multiplier,
            confidence_z=cfg.confidence_z,
            min_edge=cfg.min_edge,
            max_stake_fraction=cfg.max_stake_fraction,
        )

        if (
            cfg.require_positive_worst_case
            and ev.verdict == "bet"
            and not math.isnan(wc)
            and wc * quote.odds <= 1.0
        ):
            ev = BetEvaluation(
                **{
                    **ev.as_dict(),
                    "verdict": "pass",
                    "stake": 0.0,
                    "stake_fraction": 0.0,
                    "reason": "does not survive worst-case devig",
                }
            )

        opportunities.append(
            Opportunity(
                market=market.key,
                book=quote.book,
                outcome=quote.outcome,
                evaluation=ev,
                consensus_prob=bench_prob,
                model_prob=model_probs.get(quote.outcome) if model_probs else None,
                blended_prob=blended,
                worst_case_prob=wc,
                dispersion=bench.dispersion.get(quote.outcome, 0.0),
                n_books_in_benchmark=bench.n_books,
                notes=notes,
            )
        )

    opportunities.sort(key=lambda o: -o.evaluation.ev_per_unit)

    return MarketAnalysis(
        market=market.key,
        consensus=overall,
        fair_probs=fair,
        best=best_prices(market),
        opportunities=opportunities,
        arbitrage=arbitrage(market),
        shopping_gain=line_shopping_gain(market),
        devig_sensitivity=sensitivity,
        warnings=warnings,
    )


def _candidate_name(opp: Opportunity) -> str:
    return f"{opp.market}|{opp.outcome}|{opp.book}"


def analyze_slate(
    markets: Sequence[Market],
    model_probs: Mapping[str, Mapping[str, float]] | None = None,
    config: AnalysisConfig | None = None,
    bankroll: float = 0.0,
) -> dict:
    """Analyse several markets and size the resulting bets jointly.

    Individually-correct Kelly stakes over-lever a slate. The recommended
    stakes come from :func:`claudebet.portfolio.optimize`, which solves for the
    growth-optimal set under a total exposure cap and treats bets sharing a
    market as mutually exclusive.
    """
    cfg = config or AnalysisConfig()
    analyses = [
        analyze(m, (model_probs or {}).get(m.key), cfg, bankroll) for m in markets
    ]

    # Keep only the best price for each (market, outcome) -- betting the same
    # side at two books is one position, not two.
    best_by_selection: dict[tuple[str, str], Opportunity] = {}
    for analysis in analyses:
        for opp in analysis.bets:
            key = (opp.market, opp.outcome)
            current = best_by_selection.get(key)
            if current is None or opp.evaluation.odds > current.evaluation.odds:
                best_by_selection[key] = opp

    # Two outcomes of the same market were each scored against a *different*
    # leave-one-out benchmark, so their probabilities need not be coherent and
    # can sum to more than one. Mutual exclusivity forbids that. Scale the
    # group down proportionally, which is the conservative reading: the
    # incoherence is real uncertainty, and shrinking every leg is the honest
    # way to charge for it. A bet that only cleared the threshold on the back
    # of that incoherence will now fail to, and the optimiser will drop it.
    grouped: dict[str, list[Opportunity]] = {}
    for opp in best_by_selection.values():
        grouped.setdefault(opp.market, []).append(opp)

    probs: dict[str, float] = {}
    incoherent: set[str] = set()
    for market_key, group in grouped.items():
        total = sum(o.evaluation.used_prob for o in group)
        scale = 1.0 / total if total > 1.0 else 1.0
        if scale < 1.0:
            incoherent.add(market_key)
        for opp in group:
            probs[_candidate_name(opp)] = opp.evaluation.used_prob * scale

    candidates = [
        Candidate(
            name=_candidate_name(opp),
            prob=probs[_candidate_name(opp)],
            odds=opp.evaluation.odds,
            group=opp.market,
        )
        for opp in best_by_selection.values()
    ]

    allocation: Allocation | None = None
    if candidates:
        allocation = optimize(
            candidates,
            kelly_multiplier=cfg.kelly_multiplier,
            max_exposure=cfg.max_slate_exposure,
            max_per_bet=cfg.max_stake_fraction,
        )

    recommendations = []
    declined = []
    for opp in best_by_selection.values():
        name = _candidate_name(opp)
        fraction = allocation.stakes.get(name, 0.0) if allocation else 0.0
        notes = list(opp.notes)
        if opp.market in incoherent:
            notes.append(
                "both sides of this market were flagged; their probabilities "
                "were scaled to be mutually consistent before staking"
            )
        row = {
            "market": opp.market,
            "outcome": opp.outcome,
            "book": opp.book,
            "odds": opp.evaluation.odds,
            "american": opp.evaluation.american,
            "fair_prob": opp.blended_prob,
            "edge_pp": opp.evaluation.edge_pp,
            "edge_cents": opp.evaluation.edge_cents,
            "ev_per_unit": opp.evaluation.ev_per_unit,
            "standalone_fraction": opp.evaluation.stake_fraction,
            "slate_fraction": fraction,
            "stake": fraction * bankroll,
            "notes": notes,
        }
        (recommendations if fraction > 1e-9 else declined).append(row)

    recommendations.sort(key=lambda r: -r["ev_per_unit"])

    return {
        "analyses": analyses,
        "recommendations": recommendations,
        "declined": declined,
        "allocation": allocation,
        "total_exposure": allocation.total_exposure if allocation else 0.0,
        "expected_growth": allocation.growth_rate if allocation else 0.0,
        "arbs": [a.arbitrage for a in analyses if a.arbitrage],
    }
