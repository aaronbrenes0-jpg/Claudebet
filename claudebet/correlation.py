"""Parlays, same-game parlays, and why they are usually a tax.

A parlay's fair probability is the product of the leg probabilities *only when
the legs are independent*. Same-game legs almost never are: a quarterback going
over his passing yards and his team winning are positively correlated; two
opposing running backs going over their rushing totals are negatively
correlated (the game only has so many plays).

Two things follow, and this module measures both:

* Multiplying independent legs together compounds the vig. A three-leg parlay
  built from three 4.5%-hold markets carries roughly 13% hold. That is the
  default state of a parlay and it is why they are a losing product.
* When a book prices a same-game parlay as if the legs were independent but
  they are *positively* correlated, the true probability exceeds the priced
  one, and the correlation is the edge. Books know this and apply their own
  correlation adjustment; the question is whether theirs is bigger than the
  real one.

Dependence is modelled with a Gaussian copula: each leg's fair probability is
mapped to a standard normal threshold, and the joint probability is the
probability that all thresholds are cleared under a correlated normal. That
keeps every leg's marginal probability exactly right while letting the joint
move, which is the property you need.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from .odds import decimal_to_prob, parse_odds, prob_to_decimal

__all__ = [
    "norm_cdf",
    "norm_ppf",
    "bivariate_normal_cdf",
    "ParlayLeg",
    "ParlayResult",
    "parlay",
    "independent_parlay_prob",
]


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# Acklam's rational approximation, refined by one Halley step; accurate to
# roughly 1e-15 across the range, which is far more than we need.
_A = [
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
]
_B = [
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
]
_C = [
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
]
_D = [
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
]


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf needs p in (0, 1), got {p}")
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1
        )
    elif p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(
            ((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]
        ) / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    else:
        q = p - 0.5
        r = q * q
        x = (
            ((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]
        ) * q / (
            ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1
        )
    # One Halley refinement.
    e = norm_cdf(x) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def bivariate_normal_cdf(h: float, k: float, rho: float) -> float:
    """P(X <= h, Y <= k) for standard bivariate normal with correlation rho.

    Integrates the density's derivative with respect to rho by Simpson's rule,
    using the identity ``Phi2(h,k,r) = Phi(h)Phi(k) + integral_0^r phi2 dr'``.
    """
    if not -1.0 < rho < 1.0:
        rho = max(-0.999999, min(0.999999, rho))
    if abs(rho) < 1e-12:
        return norm_cdf(h) * norm_cdf(k)

    n = 200  # even
    step = rho / n

    def density(r: float) -> float:
        one_minus = 1.0 - r * r
        return (
            1.0
            / (2.0 * math.pi * math.sqrt(one_minus))
            * math.exp(-(h * h - 2.0 * r * h * k + k * k) / (2.0 * one_minus))
        )

    total = density(0.0) + density(rho)
    for i in range(1, n):
        total += (4.0 if i % 2 else 2.0) * density(i * step)
    integral = total * step / 3.0
    return min(1.0, max(0.0, norm_cdf(h) * norm_cdf(k) + integral))


@dataclass(frozen=True)
class ParlayLeg:
    name: str
    prob: float  # fair probability, already devigged
    odds: float  # the price the book is offering on this leg alone

    def __post_init__(self) -> None:
        object.__setattr__(self, "odds", parse_odds(self.odds))


@dataclass(frozen=True)
class ParlayResult:
    fair_prob: float
    independent_prob: float
    offered_odds: float
    fair_odds: float
    ev_per_unit: float
    hold: float
    correlation_gain: float  # fair - independent, in probability points
    n_legs: int
    method: str

    @property
    def is_positive_ev(self) -> bool:
        return self.ev_per_unit > 0


def independent_parlay_prob(legs: Sequence[ParlayLeg]) -> float:
    p = 1.0
    for leg in legs:
        p *= leg.prob
    return p


def _cholesky(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = matrix[i][i] - s
                # Nudge a non-PSD correlation matrix onto the boundary rather
                # than failing; user-supplied correlations are rarely coherent.
                L[i][j] = math.sqrt(max(val, 1e-12))
            else:
                L[i][j] = (matrix[i][j] - s) / L[j][j]
    return L


def parlay(
    legs: Sequence[ParlayLeg],
    offered_odds: float | None = None,
    correlations: Mapping[tuple[str, str], float] | None = None,
    samples: int = 200_000,
    seed: int = 12345,
) -> ParlayResult:
    """Fair value of a parlay, with correlation between legs.

    ``correlations``
        ``{(leg_a, leg_b): rho}`` on the latent normal scale, in (-1, 1).
        Order does not matter. Omitted pairs are treated as independent. As a
        rough guide for same-game legs: a team's win and its star player going
        over a counting stat sit around +0.2 to +0.4; a game total over and
        both teams' player overs around +0.3; opposing players competing for
        the same touches around -0.2.
    ``offered_odds``
        The book's parlay price. Defaults to the independent product of the
        legs' own prices, which is what most books post before their own
        correlation haircut.

    Two legs are computed exactly from the bivariate normal; three or more use
    a seeded Monte Carlo over the Gaussian copula, so results are reproducible.
    """
    if len(legs) < 2:
        raise ValueError("a parlay needs at least two legs")
    names = [leg.name for leg in legs]
    if len(set(names)) != len(names):
        raise ValueError("leg names must be unique")

    n = len(legs)
    rho = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    if correlations:
        index = {name: i for i, name in enumerate(names)}
        for (a, b), value in correlations.items():
            if a not in index or b not in index:
                raise ValueError(f"correlation references unknown leg: {(a, b)}")
            i, j = index[a], index[b]
            if i == j:
                continue
            v = max(-0.999, min(0.999, float(value)))
            rho[i][j] = rho[j][i] = v

    # Threshold form: leg i hits when Z_i > z_i, i.e. Z_i <= -z_i fails.
    thresholds = [norm_ppf(1.0 - leg.prob) for leg in legs]

    if n == 2 and correlations:
        # P(both hit) = P(Z1 > t1, Z2 > t2) = Phi2(-t1, -t2, rho) by symmetry.
        fair = bivariate_normal_cdf(-thresholds[0], -thresholds[1], rho[0][1])
        method = "bivariate-exact"
    elif not correlations:
        fair = independent_parlay_prob(legs)
        method = "independent"
    else:
        L = _cholesky(rho)
        rng = random.Random(seed)
        hits = 0
        for _ in range(samples):
            z = [rng.gauss(0.0, 1.0) for _ in range(n)]
            ok = True
            for i in range(n):
                corr_z = sum(L[i][k] * z[k] for k in range(i + 1))
                if corr_z <= thresholds[i]:
                    ok = False
                    break
            if ok:
                hits += 1
        fair = hits / samples
        method = f"gaussian-copula-mc({samples})"

    fair = min(max(fair, 1e-12), 1.0 - 1e-12)
    independent = independent_parlay_prob(legs)

    if offered_odds is None:
        price = 1.0
        for leg in legs:
            price *= leg.odds
    else:
        price = parse_odds(offered_odds)

    ev = fair * price - 1.0
    booked = decimal_to_prob(price)
    return ParlayResult(
        fair_prob=fair,
        independent_prob=independent,
        offered_odds=price,
        fair_odds=prob_to_decimal(fair),
        ev_per_unit=ev,
        hold=(booked - fair) / booked if booked > 0 else 0.0,
        correlation_gain=fair - independent,
        n_legs=n,
        method=method,
    )
