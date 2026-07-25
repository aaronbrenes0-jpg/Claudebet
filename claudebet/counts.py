"""Distributions for counting things: goals, corners, cards, shots.

Every "over/under" market is a question about a count, and the answer depends
on which distribution you assume. Goals are close to Poisson. Corners and cards
are *not* -- they are overdispersed, meaning their variance exceeds their mean,
because tempo, referee and game state push whole matches high or low together.

Assuming Poisson for corners is the single most common error in this corner of
betting. It understates both tails, so it makes the extreme lines look like
value when they are not. A negative binomial with the dispersion estimated from
your own data fixes it, and :func:`fit_counts` picks between the two by looking
at the sample rather than by assumption.

Typical variance-to-mean ratios, for sanity-checking your data:

===========  ==================
Goals        1.0 - 1.1  (Poisson is fine)
Corners      1.2 - 1.5  (overdispersed)
Cards        1.3 - 1.8  (overdispersed)
===========  ==================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "CountModel",
    "Poisson",
    "NegativeBinomial",
    "Discrete",
    "fit_counts",
    "convolve",
    "dispersion_ratio",
]

_MAX_SUPPORT = 60


class CountModel:
    """A distribution over non-negative integers, priced for betting markets."""

    mean: float
    variance: float

    def pmf(self, k: int) -> float:
        raise NotImplementedError

    def cdf(self, k: int) -> float:
        """P(X <= k)."""
        if k < 0:
            return 0.0
        return sum(self.pmf(i) for i in range(int(k) + 1))

    def sf(self, k: int) -> float:
        """P(X > k)."""
        return max(0.0, 1.0 - self.cdf(k))

    # -- market pricing -------------------------------------------------

    def over(self, line: float) -> float:
        return self.sf(math.floor(line))

    def push(self, line: float) -> float:
        """Whole-number lines can push; half lines cannot."""
        return self.pmf(int(line)) if float(line).is_integer() else 0.0

    def under(self, line: float) -> float:
        return max(0.0, 1.0 - self.over(line) - self.push(line))

    def total_market(self, line: float) -> dict[str, float]:
        return {
            "over": self.over(line),
            "under": self.under(line),
            "push": self.push(line),
        }

    def at_least(self, k: int) -> float:
        return self.sf(k - 1)

    def support(self, limit: int = _MAX_SUPPORT) -> list[float]:
        """Materialise the pmf as a list, renormalised over the truncation."""
        values = [self.pmf(k) for k in range(limit + 1)]
        total = sum(values)
        return [v / total for v in values] if total > 0 else values


@dataclass
class Poisson(CountModel):
    mean: float

    def __post_init__(self) -> None:
        self.mean = max(1e-9, float(self.mean))
        self.variance = self.mean

    def pmf(self, k: int) -> float:
        if k < 0:
            return 0.0
        k = int(k)
        if k > 170:
            return 0.0
        return math.exp(k * math.log(self.mean) - self.mean - math.lgamma(k + 1))


@dataclass
class NegativeBinomial(CountModel):
    """Parameterised by mean and variance, which is how you actually measure it.

    ``size`` (the usual ``r``) is derived as ``mean^2 / (variance - mean)``.
    As the variance approaches the mean, ``size`` grows without bound and the
    distribution converges to Poisson -- so this degrades gracefully rather
    than blowing up on well-behaved data.
    """

    mean: float
    variance: float

    def __post_init__(self) -> None:
        self.mean = max(1e-9, float(self.mean))
        self.variance = max(float(self.variance), self.mean * (1.0 + 1e-9))
        self.size = self.mean**2 / (self.variance - self.mean)
        self.prob = self.size / (self.size + self.mean)

    def pmf(self, k: int) -> float:
        if k < 0:
            return 0.0
        k = int(k)
        r, p = self.size, self.prob
        return math.exp(
            math.lgamma(k + r)
            - math.lgamma(r)
            - math.lgamma(k + 1)
            + r * math.log(p)
            + k * math.log(1.0 - p)
        )


@dataclass
class Discrete(CountModel):
    """An explicit pmf, used for sums of independent counts."""

    probs: Sequence[float]

    def __post_init__(self) -> None:
        total = sum(self.probs)
        self.probs = [p / total for p in self.probs] if total > 0 else list(self.probs)
        self.mean = sum(k * p for k, p in enumerate(self.probs))
        self.variance = sum((k - self.mean) ** 2 * p for k, p in enumerate(self.probs))

    def pmf(self, k: int) -> float:
        k = int(k)
        return self.probs[k] if 0 <= k < len(self.probs) else 0.0


def dispersion_ratio(samples: Sequence[float]) -> float:
    """Variance divided by mean. Above ~1.15 means Poisson is the wrong model."""
    values = [float(s) for s in samples]
    if len(values) < 2:
        return 1.0
    mean = sum(values) / len(values)
    if mean <= 0:
        return 1.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return var / mean


def fit_counts(
    samples: Sequence[float], overdispersion_threshold: float = 1.15
) -> CountModel:
    """Fit whichever of Poisson or negative binomial the data supports.

    Small samples routinely show a variance-to-mean ratio above one by chance,
    so the threshold defaults slightly above 1.0 rather than at it -- fitting a
    negative binomial to noise buys nothing and costs tail accuracy.
    """
    values = [float(s) for s in samples]
    if not values:
        raise ValueError("no samples to fit")
    mean = sum(values) / len(values)
    if len(values) < 2 or mean <= 0:
        return Poisson(max(mean, 1e-9))
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    if var / mean <= overdispersion_threshold:
        return Poisson(mean)
    return NegativeBinomial(mean, var)


def convolve(a: CountModel, b: CountModel, limit: int = _MAX_SUPPORT) -> Discrete:
    """Distribution of the sum of two independent counts.

    Used for match totals built from the two teams separately -- total corners
    from home corners plus away corners, say. Independence is an approximation:
    a fast, open game lifts both. If you have match-level totals, model them
    directly instead of convolving; :func:`fit_counts` on the totals column
    will capture that shared variation as overdispersion.
    """
    pa, pb = a.support(limit), b.support(limit)
    out = [0.0] * (limit + 1)
    for i, x in enumerate(pa):
        if x <= 0:
            continue
        for j, y in enumerate(pb):
            if i + j > limit:
                break
            out[i + j] += x * y
    return Discrete(out)
