"""Removing the vig: turning posted prices into fair probabilities.

A book's implied probabilities sum to more than one. Recovering the fair
probabilities means deciding *how* the surplus is distributed across outcomes,
and that choice matters enormously at long prices. On a market posted at
-110/-110 every method here agrees to within a rounding error; on a +2000
longshot the spread between methods can exceed the entire edge you are hunting.

Methods, and when to reach for them:

``multiplicative``
    Divide by the overround. Assumes the vig is proportional to each outcome's
    probability. Fast, standard, and biased: it overstates longshots, because
    books load proportionally *more* margin onto them.
``additive``
    Subtract the surplus equally. The opposite bias -- it treats the margin as
    a flat per-outcome charge, which understates longshots and can produce
    negative probabilities on wide markets.
``power``
    Solve ``sum(q_i ** k) == 1``. A smooth compromise that empirically fits
    observed favourite-longshot bias well on multi-outcome markets.
``shin``
    Models the margin as the book's protection against insider money. Solves
    for ``z``, the implied proportion of informed traders. The best-supported
    method in the literature for two- and three-way markets, and ``z`` is a
    useful signal in its own right.
``odds_ratio``
    Keeps the odds ratio between fair and booked probabilities constant across
    outcomes (Cheung). Behaves similarly to ``shin`` and is cheaper to reason
    about.

Default is ``shin``. If two methods disagree by more than your edge, you do not
have an edge -- :func:`devig_spread` measures exactly that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .odds import decimal_to_prob, prob_to_decimal

__all__ = [
    "DevigResult",
    "devig",
    "devig_probs",
    "devig_spread",
    "worst_case_prob",
    "METHODS",
]

METHODS = ("multiplicative", "additive", "power", "shin", "odds_ratio")

_TOL = 1e-12
_MAX_ITER = 200


@dataclass(frozen=True)
class DevigResult:
    """Fair probabilities recovered from one book's prices on one market."""

    probs: tuple[float, ...]
    method: str
    overround: float
    params: dict[str, float] = field(default_factory=dict)

    @property
    def fair_odds(self) -> tuple[float, ...]:
        return tuple(prob_to_decimal(p) for p in self.probs)

    @property
    def hold(self) -> float:
        return (self.overround - 1.0) / self.overround

    def __getitem__(self, i: int) -> float:
        return self.probs[i]

    def __len__(self) -> int:
        return len(self.probs)


def _booked(odds: Sequence[float]) -> list[float]:
    if len(odds) < 2:
        raise ValueError("a market needs at least two outcomes to devig")
    return [decimal_to_prob(o) for o in odds]


def _bisect(f, lo: float, hi: float) -> float:
    """Root-find a function known to be monotone decreasing on [lo, hi]."""
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        # No sign change in the bracket; return whichever end is closer to zero.
        return lo if abs(flo) < abs(fhi) else hi
    for _ in range(_MAX_ITER):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < _TOL or (hi - lo) < _TOL:
            return mid
        if fmid * flo > 0:
            lo, flo = mid, fmid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _multiplicative(q: list[float]) -> tuple[list[float], dict[str, float]]:
    total = sum(q)
    return [x / total for x in q], {}


def _additive(q: list[float]) -> tuple[list[float], dict[str, float]]:
    n = len(q)
    surplus = (sum(q) - 1.0) / n
    p = [x - surplus for x in q]
    if min(p) <= 0.0:
        # Wide market: the flat charge swamped a longshot. Floor and rescale
        # rather than returning a nonsensical negative probability.
        floor = 1e-6
        p = [max(x, floor) for x in p]
        total = sum(p)
        p = [x / total for x in p]
    return p, {}


def _power(q: list[float]) -> tuple[list[float], dict[str, float]]:
    def excess(k: float) -> float:
        return sum(x**k for x in q) - 1.0

    k = _bisect(excess, 1.0, 100.0)
    p = [x**k for x in q]
    total = sum(p)
    return [x / total for x in p], {"k": k}


def _shin(q: list[float]) -> tuple[list[float], dict[str, float]]:
    total = sum(q)

    def shin_probs(z: float) -> list[float]:
        if z >= 1.0 - 1e-9:
            return [x / total for x in q]
        return [
            (math.sqrt(z * z + 4.0 * (1.0 - z) * x * x / total) - z) / (2.0 * (1.0 - z))
            for x in q
        ]

    def excess(z: float) -> float:
        return sum(shin_probs(z)) - 1.0

    z = _bisect(excess, 0.0, 1.0 - 1e-9)
    p = shin_probs(z)
    s = sum(p)
    return [x / s for x in p], {"z": z}


def _odds_ratio(q: list[float]) -> tuple[list[float], dict[str, float]]:
    booked_odds = [x / (1.0 - x) for x in q]

    def probs_at(c: float) -> list[float]:
        return [c * o / (1.0 + c * o) for o in booked_odds]

    def excess(c: float) -> float:
        return sum(probs_at(c)) - 1.0

    c = _bisect(excess, 1e-9, 1.0)
    p = probs_at(c)
    s = sum(p)
    return [x / s for x in p], {"c": c}


_IMPLS = {
    "multiplicative": _multiplicative,
    "additive": _additive,
    "power": _power,
    "shin": _shin,
    "odds_ratio": _odds_ratio,
}


def devig(odds: Sequence[float], method: str = "shin") -> DevigResult:
    """Strip the margin out of one book's prices on a single market.

    ``odds`` must be decimal prices covering every outcome of the market. A
    partial market (say, one side of a total) cannot be devigged -- you need
    both sides to know how much margin there is.
    """
    if method not in _IMPLS:
        raise ValueError(f"unknown devig method {method!r}; choose from {METHODS}")
    q = _booked(odds)
    total = sum(q)
    if total <= 1.0:
        # Zero or negative hold: either a promo, a stale screen, or a genuine
        # arb. Normalising is still the honest fair estimate.
        p = [x / total for x in q]
        return DevigResult(tuple(p), method, total, {"note_no_vig": 1.0})
    p, params = _IMPLS[method](q)
    return DevigResult(tuple(p), method, total, params)


def devig_probs(odds: Sequence[float], method: str = "shin") -> tuple[float, ...]:
    return devig(odds, method).probs


def devig_spread(odds: Sequence[float], methods: Sequence[str] = METHODS) -> dict:
    """How much the choice of devig method moves each outcome's fair price.

    Returns per-outcome min/max/spread across methods plus the max spread over
    the market. Treat ``max_spread`` as a floor on your model error: if your
    claimed edge is 1.5% and the methods disagree by 2%, the edge is noise.
    """
    results = {m: devig(odds, m).probs for m in methods}
    n = len(odds)
    per_outcome = []
    for i in range(n):
        vals = [results[m][i] for m in methods]
        lo, hi = min(vals), max(vals)
        per_outcome.append(
            {
                "min": lo,
                "max": hi,
                "spread": hi - lo,
                "by_method": {m: results[m][i] for m in methods},
            }
        )
    return {
        "outcomes": per_outcome,
        "max_spread": max(o["spread"] for o in per_outcome),
        "methods": list(methods),
    }


def worst_case_prob(odds: Sequence[float], index: int) -> float:
    """The most pessimistic fair probability for one outcome.

    Assumes every other outcome is priced with no margin at all, so the entire
    overround sits on the outcome you care about. If a bet still shows positive
    EV under this assumption, the edge does not depend on your devig choice.
    """
    q = _booked(odds)
    rest = sum(x for i, x in enumerate(q) if i != index)
    return max(1e-9, min(1.0 - 1e-9, 1.0 - rest))
