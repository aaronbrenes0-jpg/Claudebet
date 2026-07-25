"""Is the model actually any good, and are its numbers real probabilities?

Two different questions, and both matter.

*Calibration* asks whether things you call 70% happen 70% of the time. A model
can be badly calibrated and still profitable, but you cannot size bets with it
-- Kelly consumes probabilities, so miscalibration goes straight into your
stake. :class:`PlattCalibrator` and :class:`IsotonicCalibrator` fix it after
the fact, which is nearly always cheaper than retraining.

*Skill* asks whether the model beats the market on the same games. This is the
only test that matters for whether you should be betting at all, and the
reference has to be the market's devigged probability, never a coin flip. A
model with a beautiful reliability diagram and a skill score of -0.01 against
the closing line is a model that has learned to agree with the market slightly
worse than the market does.

Sample sizes are brutal here: distinguishing a 2% edge from zero at the 95%
level takes on the order of a thousand bets. Treat anything under a few hundred
graded predictions as unmeasured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .blend import expit, logit

__all__ = [
    "brier_score",
    "log_loss",
    "skill_score",
    "reliability_table",
    "expected_calibration_error",
    "PlattCalibrator",
    "IsotonicCalibrator",
    "report",
]

_EPS = 1e-12


def _check(probs: Sequence[float], outcomes: Sequence[int]) -> None:
    if len(probs) != len(outcomes):
        raise ValueError("probs and outcomes must be the same length")
    if not probs:
        raise ValueError("no predictions supplied")
    for o in outcomes:
        if o not in (0, 1, True, False):
            raise ValueError(f"outcomes must be 0/1, got {o!r}")


def brier_score(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mean squared error of the probabilities. Lower is better."""
    _check(probs, outcomes)
    return sum((p - float(o)) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def log_loss(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mean negative log likelihood. Punishes confident mistakes harshly --
    which is correct, because that is what Kelly does to your bankroll."""
    _check(probs, outcomes)
    total = 0.0
    for p, o in zip(probs, outcomes):
        p = min(max(p, _EPS), 1 - _EPS)
        total += -(math.log(p) if o else math.log(1 - p))
    return total / len(probs)


def skill_score(
    probs: Sequence[float],
    outcomes: Sequence[int],
    reference: Sequence[float],
) -> float:
    """``1 - BS_model / BS_reference``. Positive means the model beat the
    reference. Pass the market's devigged closing probability as
    ``reference`` -- that is the bar, and clearing it is hard."""
    bs_model = brier_score(probs, outcomes)
    bs_ref = brier_score(reference, outcomes)
    if bs_ref <= 0:
        return 0.0
    return 1.0 - bs_model / bs_ref


@dataclass(frozen=True)
class Bin:
    lo: float
    hi: float
    n: int
    mean_pred: float
    observed: float

    @property
    def error(self) -> float:
        return self.observed - self.mean_pred


def reliability_table(
    probs: Sequence[float], outcomes: Sequence[int], bins: int = 10
) -> list[Bin]:
    """Group predictions into probability buckets and compare predicted to
    observed. Empty buckets are dropped."""
    _check(probs, outcomes)
    if bins < 2:
        raise ValueError("need at least two bins")
    edges = [i / bins for i in range(bins + 1)]
    table: list[Bin] = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        members = [
            (p, float(o))
            for p, o in zip(probs, outcomes)
            if (lo <= p < hi) or (i == bins - 1 and p == hi)
        ]
        if not members:
            continue
        n = len(members)
        table.append(
            Bin(
                lo=lo,
                hi=hi,
                n=n,
                mean_pred=sum(p for p, _ in members) / n,
                observed=sum(o for _, o in members) / n,
            )
        )
    return table


def expected_calibration_error(
    probs: Sequence[float], outcomes: Sequence[int], bins: int = 10
) -> float:
    """Sample-weighted mean absolute gap between predicted and observed."""
    table = reliability_table(probs, outcomes, bins)
    total = sum(b.n for b in table)
    if total == 0:
        return 0.0
    return sum(b.n * abs(b.error) for b in table) / total


class PlattCalibrator:
    """Logistic recalibration: ``p' = sigmoid(a · logit(p) + b)``.

    Two parameters, so it is stable on a few hundred observations. Fixes
    systematic over- or under-confidence (``a``) and a constant bias (``b``)
    but cannot fix a model that is wrong in a non-monotone way.
    """

    def __init__(self) -> None:
        self.a: float = 1.0
        self.b: float = 0.0
        self.n: int = 0

    def fit(
        self, probs: Sequence[float], outcomes: Sequence[int], iterations: int = 100
    ) -> "PlattCalibrator":
        _check(probs, outcomes)
        x = [logit(p) for p in probs]
        y = [float(bool(o)) for o in outcomes]
        # Smoothed targets (Platt's own correction) keep the fit finite when a
        # bucket is perfectly separated.
        n_pos = sum(y)
        n_neg = len(y) - n_pos
        hi = (n_pos + 1.0) / (n_pos + 2.0)
        lo = 1.0 / (n_neg + 2.0)
        t = [hi if v else lo for v in y]

        a, b = 1.0, 0.0
        for _ in range(iterations):
            g_a = g_b = h_aa = h_ab = h_bb = 0.0
            for xi, ti in zip(x, t):
                p = expit(a * xi + b)
                d = p - ti
                w = max(p * (1 - p), 1e-12)
                g_a += d * xi
                g_b += d
                h_aa += w * xi * xi
                h_ab += w * xi
                h_bb += w
            det = h_aa * h_bb - h_ab * h_ab
            if abs(det) < 1e-14:
                break
            step_a = (h_bb * g_a - h_ab * g_b) / det
            step_b = (h_aa * g_b - h_ab * g_a) / det
            a -= step_a
            b -= step_b
            if abs(step_a) < 1e-10 and abs(step_b) < 1e-10:
                break
        self.a, self.b, self.n = a, b, len(x)
        return self

    def transform(self, p: float) -> float:
        return expit(self.a * logit(p) + self.b)

    def __call__(self, p: float) -> float:
        return self.transform(p)

    def describe(self) -> str:
        if self.a > 1.05:
            shape = "model is under-confident; recalibration sharpens it"
        elif self.a < 0.95:
            shape = "model is over-confident; recalibration flattens it"
        else:
            shape = "model confidence is about right"
        bias = "no systematic bias" if abs(self.b) < 0.05 else (
            f"systematic bias of {self.b:+.3f} in log-odds"
        )
        return f"a={self.a:.3f}, b={self.b:+.3f} (n={self.n}): {shape}; {bias}"


class IsotonicCalibrator:
    """Non-parametric monotone recalibration by pool-adjacent-violators.

    More flexible than Platt and it will happily overfit a small sample --
    below about 500 observations prefer Platt. Predictions between fitted
    points are linearly interpolated.
    """

    def __init__(self) -> None:
        self.x: list[float] = []
        self.y: list[float] = []

    def fit(
        self, probs: Sequence[float], outcomes: Sequence[int]
    ) -> "IsotonicCalibrator":
        _check(probs, outcomes)
        pairs = sorted(zip(probs, (float(bool(o)) for o in outcomes)))
        values = [p[1] for p in pairs]
        weights = [1.0] * len(pairs)
        xs = [p[0] for p in pairs]

        # PAVA: merge adjacent blocks until the sequence is non-decreasing.
        block_val: list[float] = []
        block_w: list[float] = []
        block_x: list[float] = []
        for xi, vi, wi in zip(xs, values, weights):
            block_val.append(vi)
            block_w.append(wi)
            block_x.append(xi)
            while len(block_val) > 1 and block_val[-2] > block_val[-1]:
                w = block_w[-2] + block_w[-1]
                v = (block_val[-2] * block_w[-2] + block_val[-1] * block_w[-1]) / w
                x_merged = (block_x[-2] * block_w[-2] + block_x[-1] * block_w[-1]) / w
                block_val[-2:] = [v]
                block_w[-2:] = [w]
                block_x[-2:] = [x_merged]
        self.x, self.y = block_x, block_val
        return self

    def transform(self, p: float) -> float:
        if not self.x:
            return p
        if p <= self.x[0]:
            return self.y[0]
        if p >= self.x[-1]:
            return self.y[-1]
        lo, hi = 0, len(self.x) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.x[mid] <= p:
                lo = mid
            else:
                hi = mid
        span = self.x[hi] - self.x[lo]
        if span <= 0:
            return self.y[lo]
        t = (p - self.x[lo]) / span
        return self.y[lo] + t * (self.y[hi] - self.y[lo])

    def __call__(self, p: float) -> float:
        return self.transform(p)


def report(
    probs: Sequence[float],
    outcomes: Sequence[int],
    reference: Sequence[float] | None = None,
    bins: int = 10,
) -> dict:
    """Everything worth knowing about a set of graded predictions."""
    out = {
        "n": len(probs),
        "brier": brier_score(probs, outcomes),
        "log_loss": log_loss(probs, outcomes),
        "ece": expected_calibration_error(probs, outcomes, bins),
        "base_rate": sum(float(bool(o)) for o in outcomes) / len(outcomes),
        "mean_prediction": sum(probs) / len(probs),
        "reliability": [
            {
                "range": f"{b.lo:.0%}-{b.hi:.0%}",
                "n": b.n,
                "predicted": b.mean_pred,
                "observed": b.observed,
                "error": b.error,
            }
            for b in reliability_table(probs, outcomes, bins)
        ],
    }
    if reference is not None:
        out["reference_brier"] = brier_score(reference, outcomes)
        out["skill_vs_reference"] = skill_score(probs, outcomes, reference)
    if len(probs) < 200:
        out["warning"] = (
            f"only {len(probs)} predictions: too few to distinguish skill from "
            "noise. Treat every number here as provisional."
        )
    return out
