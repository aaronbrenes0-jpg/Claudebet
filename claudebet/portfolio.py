"""Staking a whole slate at once.

Single-bet Kelly is only correct when you have one bet. Bet six games on a
Sunday and the individually-correct stakes add up to an exposure that is
collectively over-levered: the growth-optimal set of stakes on simultaneous
bets is always smaller than the sum of the standalone Kelly stakes, and the
gap widens fast when the bets are correlated or mutually exclusive.

This module maximises expected log wealth over the *joint* outcome space:

    maximise  sum_s  P(s) · log(1 + f · x_s)
    subject to f >= 0 and sum(f) <= max_exposure

where ``s`` ranges over joint scenarios and ``x_s`` is the profit-per-unit
vector in that scenario. Mutually exclusive bets (two sides of the same game,
several outcomes of one market) are declared with a shared ``group`` and are
never allowed to win together, which is what makes hedges and middles come out
correctly instead of being double-counted.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from itertools import product
from typing import Sequence

from .odds import parse_odds

__all__ = ["Candidate", "Allocation", "optimize", "scenarios_for"]

_MAX_ENUMERATED_SCENARIOS = 200_000


@dataclass(frozen=True)
class Candidate:
    """One bet you could place.

    ``group``
        Bets sharing a group are mutually exclusive -- at most one can win.
        Use the market key for the two sides of a total, or the game id to
        stop the optimiser piling into several correlated angles on one game.
    """

    name: str
    prob: float
    odds: float
    group: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "odds", parse_odds(self.odds))
        p = float(self.prob)
        if not 0.0 < p < 1.0:
            raise ValueError(f"{self.name}: prob must be in (0, 1), got {p}")
        object.__setattr__(self, "prob", p)

    @property
    def payoff(self) -> float:
        """Profit per unit staked if it wins."""
        return self.odds - 1.0


@dataclass
class Allocation:
    """The optimiser's answer."""

    stakes: dict[str, float]  # fraction of bankroll per candidate
    growth_rate: float  # expected log-wealth growth per slate
    total_exposure: float
    n_scenarios: int
    iterations: int
    standalone: dict[str, float] = field(default_factory=dict)
    method: str = "exact"  # "exact" or "quadratic+line-search"

    def scaled(self, bankroll: float) -> dict[str, float]:
        return {k: v * bankroll for k, v in self.stakes.items()}

    def shrinkage(self) -> dict[str, float]:
        """How much each stake was cut relative to standalone Kelly. Values
        well below 1 are the point of this module, not a bug."""
        return {
            k: (self.stakes[k] / self.standalone[k]) if self.standalone.get(k) else 0.0
            for k in self.stakes
        }


Block = list[tuple[float, tuple[int, ...]]]


def _blocks(candidates: Sequence[Candidate]) -> list[Block]:
    """Partition the slate into independent blocks of mutually exclusive
    alternatives, each a list of ``(probability, winning indices)``."""
    groups: dict[str, list[int]] = {}
    singles: list[int] = []
    for i, c in enumerate(candidates):
        if c.group is None:
            singles.append(i)
        else:
            groups.setdefault(c.group, []).append(i)

    blocks: list[Block] = []
    for i in singles:
        p = candidates[i].prob
        blocks.append([(p, (i,)), (1.0 - p, ())])
    for group_key, members in groups.items():
        total = sum(candidates[i].prob for i in members)
        if total > 1.0 + 1e-9:
            raise ValueError(
                f"mutually exclusive group {group_key!r} has probabilities summing "
                f"to {total:.4f}; they cannot exceed 1. If these came from separate "
                "leave-one-out benchmarks, rescale them to be coherent first."
            )
        alts: Block = [(candidates[i].prob, (i,)) for i in members]
        residual = max(0.0, 1.0 - total)
        if residual > 1e-12:
            alts.append((residual, ()))
        blocks.append(alts)
    return blocks


def _space_size(blocks: Sequence[Block], ceiling: int) -> int:
    size = 1
    for b in blocks:
        size *= len(b)
        if size > ceiling:
            return size
    return size


def _payoff_vector(
    candidates: Sequence[Candidate], winners: set[int]
) -> tuple[float, ...]:
    return tuple(
        candidates[i].payoff if i in winners else -1.0 for i in range(len(candidates))
    )


def scenarios_for(
    candidates: Sequence[Candidate],
    max_scenarios: int = _MAX_ENUMERATED_SCENARIOS,
) -> list[tuple[float, tuple[float, ...]]]:
    """Enumerate joint outcomes as ``(probability, payoff vector)`` pairs.

    Groups are expanded to their mutually exclusive alternatives (including
    "none of them wins"); ungrouped bets are treated as independent.

    The space doubles with every independent bet, so this raises above
    ``max_scenarios``. :func:`optimize` checks the size first and switches to
    an approximation rather than failing -- call this directly only when you
    want the exact scenario set.
    """
    candidates = list(candidates)
    blocks = _blocks(candidates)
    size = _space_size(blocks, max_scenarios)
    if size > max_scenarios:
        raise ValueError(
            f"joint outcome space of {size}+ scenarios exceeds {max_scenarios}; "
            "use optimize(), which falls back to an approximation at this size"
        )

    out: list[tuple[float, tuple[float, ...]]] = []
    for combo in product(*blocks):
        prob = 1.0
        winners: set[int] = set()
        for p, idx in combo:
            prob *= p
            winners.update(idx)
        if prob <= 0.0:
            continue
        out.append((prob, _payoff_vector(candidates, winners)))
    return out


def _moments(
    candidates: Sequence[Candidate],
) -> tuple[list[float], list[list[float]]]:
    """Mean payoff vector and covariance matrix of the per-unit payoffs.

    Independent bets have zero covariance. Two bets in the same mutually
    exclusive group cannot both win, which makes their covariance strongly
    negative -- that is precisely the structure the optimiser needs to know
    about, and it is available in closed form without enumerating anything.
    """
    n = len(candidates)
    mu = [c.prob * c.payoff - (1.0 - c.prob) for c in candidates]
    cov = [[0.0] * n for _ in range(n)]
    for i, c in enumerate(candidates):
        second = c.prob * c.payoff**2 + (1.0 - c.prob)
        cov[i][i] = second - mu[i] ** 2
    for i, ci in enumerate(candidates):
        for j in range(i + 1, n):
            cj = candidates[j]
            if ci.group is None or ci.group != cj.group:
                continue
            # Exactly one of {i wins, j wins, neither} occurs.
            joint = (
                ci.prob * ci.payoff * -1.0
                + cj.prob * -1.0 * cj.payoff
                + (1.0 - ci.prob - cj.prob) * 1.0
            )
            cov[i][j] = cov[j][i] = joint - mu[i] * mu[j]
    return mu, cov


def _solve_nonneg(cov: list[list[float]], mu: list[float], ridge: float) -> list[float]:
    """Solve ``cov · f = mu`` for ``f >= 0`` by active-set elimination.

    Solves, drops any component that came back negative, and re-solves on the
    survivors. Two or three passes is plenty at the sizes involved here.
    """
    from .models.ratings import solve_ridge

    n = len(mu)
    active = list(range(n))
    for _ in range(6):
        if not active:
            return [0.0] * n
        sub_cov = [[cov[i][j] for j in active] for i in active]
        sub_mu = [mu[i] for i in active]
        sub = solve_ridge(sub_cov, sub_mu, [ridge] * len(active))
        if all(v >= -1e-12 for v in sub):
            out = [0.0] * n
            for idx, v in zip(active, sub):
                out[idx] = max(0.0, v)
            return out
        active = [idx for idx, v in zip(active, sub) if v > 0]
    out = [0.0] * n
    return out


def _sample_payoffs(
    candidates: Sequence[Candidate], blocks: Sequence[Block], samples: int, seed: int
) -> list[tuple[float, ...]]:
    """Draw joint outcomes, each equally likely by construction."""
    rng = random.Random(seed)
    out: list[tuple[float, ...]] = []
    for _ in range(samples):
        winners: set[int] = set()
        for block in blocks:
            draw = rng.random()
            cumulative = 0.0
            for prob, idx in block:
                cumulative += prob
                if draw <= cumulative:
                    winners.update(idx)
                    break
        out.append(_payoff_vector(candidates, winners))
    return out


def _project(f: list[float], cap: float) -> list[float]:
    """Euclidean projection onto {f >= 0, sum(f) <= cap}."""
    f = [max(0.0, x) for x in f]
    total = sum(f)
    if total <= cap:
        return f
    # Project onto the simplex {f >= 0, sum(f) == cap} (Duchi et al.): find the
    # largest prefix of the sorted vector that stays positive after shifting.
    running = 0.0
    theta = 0.0
    for i, val in enumerate(sorted(f, reverse=True), start=1):
        running += val
        candidate = (running - cap) / i
        if val - candidate > 0:
            theta = candidate
    return [max(0.0, x - theta) for x in f]


def _optimize_exact(
    cands: Sequence[Candidate],
    scenarios: Sequence[tuple[float, tuple[float, ...]]],
    iterations: int,
    learning_rate: float,
    tol: float,
) -> tuple[list[float], int]:
    """Projected gradient ascent on the exact expected log wealth."""
    n = len(cands)
    f = [0.01] * n
    step = learning_rate
    used = 0

    def growth(vec: Sequence[float]) -> float:
        total = 0.0
        for prob, payoff in scenarios:
            wealth = 1.0 + sum(v * x for v, x in zip(vec, payoff))
            if wealth <= 1e-12:
                return -math.inf
            total += prob * math.log(wealth)
        return total

    current = growth(f)
    for used in range(1, iterations + 1):
        grad = [0.0] * n
        for prob, payoff in scenarios:
            wealth = 1.0 + sum(v * x for v, x in zip(f, payoff))
            if wealth <= 1e-12:
                wealth = 1e-12
            scale = prob / wealth
            for i in range(n):
                grad[i] += scale * payoff[i]

        # Backtracking line search keeps us inside the log's domain.
        improved = False
        trial_step = step
        for _ in range(40):
            trial = _project([f[i] + trial_step * grad[i] for i in range(n)], 1.0)
            value = growth(trial)
            if value > current + 1e-15:
                converged = all(abs(trial[i] - f[i]) < tol for i in range(n))
                f, current = trial, value
                step = min(trial_step * 1.3, 10.0)
                improved = not converged
                break
            trial_step *= 0.5
        if not improved:
            break
    return f, used


def _optimize_approx(
    cands: Sequence[Candidate], blocks: Sequence[Block], samples: int, seed: int
) -> tuple[list[float], int]:
    """Quadratic direction, then an exact one-dimensional scaling.

    Above a few dozen bets the joint outcome space cannot be enumerated, so we
    split the problem. The *direction* comes from the mean-variance solution
    ``f = Sigma^-1 mu``, which is the second-order expansion of expected log
    wealth and is exact to that order -- including the strong negative
    covariance between mutually exclusive bets. The *magnitude* along that
    direction is then chosen by maximising the true log objective, estimated
    on a seeded sample of joint outcomes.

    This matters because the quadratic form alone is systematically too
    aggressive: it sees the variance penalty but not the log's blow-up near
    total loss. The one-dimensional search restores that, cheaply -- once the
    per-scenario dot products are precomputed, each trial scaling costs one
    pass over the samples rather than a full re-evaluation.
    """
    mu, cov = _moments(cands)
    direction = _solve_nonneg(cov, mu, ridge=1e-9)
    norm = sum(direction)
    if norm <= 0:
        return [0.0] * len(cands), 0
    direction = [v / norm for v in direction]

    payoffs = _sample_payoffs(cands, blocks, samples, seed)
    projected = [sum(d * x for d, x in zip(direction, p)) for p in payoffs]

    worst = min(projected)
    # Keep wealth strictly positive in every sampled scenario.
    lam_max = 0.999 / -worst if worst < 0 else 100.0

    def growth(lam: float) -> float:
        total = 0.0
        for q in projected:
            wealth = 1.0 + lam * q
            if wealth <= 1e-12:
                return -math.inf
            total += math.log(wealth)
        return total / len(projected)

    # Golden-section search on a concave one-dimensional objective.
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    lo, hi = 0.0, lam_max
    c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fc, fd = growth(c), growth(d)
    steps = 0
    for steps in range(1, 81):
        if fc > fd:
            hi, d, fd = d, c, fc
            c = hi - phi * (hi - lo)
            fc = growth(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + phi * (hi - lo)
            fd = growth(d)
        if hi - lo < 1e-9:
            break
    lam = 0.5 * (lo + hi)
    return [lam * v for v in direction], steps


def optimize(
    candidates: Sequence[Candidate],
    *,
    kelly_multiplier: float = 0.25,
    max_exposure: float = 0.10,
    max_per_bet: float = 0.02,
    iterations: int = 600,
    learning_rate: float = 0.5,
    tol: float = 1e-10,
    max_scenarios: int = _MAX_ENUMERATED_SCENARIOS,
    samples: int = 20_000,
    seed: int = 17,
) -> Allocation:
    """Growth-optimal simultaneous stakes, then scaled to fractional Kelly.

    The optimisation runs at full Kelly (that is where the log-optimal maths
    lives) and ``kelly_multiplier`` scales the answer afterwards, which is the
    correct order: scaling the objective instead would change which bets get
    selected, not just their size.

    ``max_exposure`` caps the total fraction of bankroll at risk across the
    slate; ``max_per_bet`` caps any single position.

    Slates small enough to enumerate (up to ``max_scenarios`` joint outcomes,
    roughly 17 independent bets) are solved exactly. Larger ones fall back to
    the approximation described in :func:`_optimize_approx` rather than
    failing, and :attr:`Allocation.method` records which path ran.
    """
    cands = list(candidates)
    if not cands:
        return Allocation({}, 0.0, 0.0, 0, 0, {}, "exact")

    blocks = _blocks(cands)
    size = _space_size(blocks, max_scenarios)

    standalone = {}
    for c in cands:
        b = c.payoff
        standalone[c.name] = max(0.0, (c.prob * b - (1.0 - c.prob)) / b)

    if size <= max_scenarios:
        scenarios = scenarios_for(cands, max_scenarios)
        f, used = _optimize_exact(cands, scenarios, iterations, learning_rate, tol)
        method, n_scenarios = "exact", len(scenarios)

        def growth_of(vec: Sequence[float]) -> float:
            total = 0.0
            for prob, payoff in scenarios:
                wealth = 1.0 + sum(v * x for v, x in zip(vec, payoff))
                if wealth <= 1e-12:
                    return -math.inf
                total += prob * math.log(wealth)
            return total

    else:
        f, used = _optimize_approx(cands, blocks, samples, seed)
        method, n_scenarios = "quadratic+line-search", samples
        sampled = _sample_payoffs(cands, blocks, samples, seed)

        def growth_of(vec: Sequence[float]) -> float:
            total = 0.0
            for payoff in sampled:
                wealth = 1.0 + sum(v * x for v, x in zip(vec, payoff))
                if wealth <= 1e-12:
                    return -math.inf
                total += math.log(wealth)
            return total / len(sampled)

    scaled = [x * float(kelly_multiplier) for x in f]
    scaled = [min(x, float(max_per_bet)) for x in scaled]
    scaled = _project(scaled, float(max_exposure))

    return Allocation(
        stakes={c.name: s for c, s in zip(cands, scaled)},
        growth_rate=growth_of(scaled),
        total_exposure=sum(scaled),
        n_scenarios=n_scenarios,
        iterations=used,
        standalone={k: v * float(kelly_multiplier) for k, v in standalone.items()},
        method=method,
    )
