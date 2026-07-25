"""Combining a model's opinion with the market's.

The market is not an opponent to be beaten from scratch -- it is a very strong
prior assembled from far more information than any private model has. The
useful question is never "what does my model say" but "how far, and in which
direction, should my model be allowed to move the market's number".

Log-linear pooling is the tool: ``p ∝ p_model^w · p_market^(1-w)``, renormalised
over the market's outcomes. In a two-way market this is exactly averaging the
log-odds, so a model that is 60% where the market is 50% moves the blend by a
fixed number of log-odds regardless of where on the probability scale it sits
-- which is the behaviour you want. Straight arithmetic averaging does not have
that property and systematically over-weights the model at long prices.

The weight itself should not be a constant you like the look of. Use
:func:`weight_from_evidence`, which lets three things decide it: how much
history the model has been validated on, how well calibrated it proved to be
out of sample, and how much the books currently disagree with each other. A
model with 80 graded bets and no calibration record earns a weight near 0.1,
not 0.5.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

__all__ = [
    "logit",
    "expit",
    "log_linear_pool",
    "blend",
    "weight_from_evidence",
    "shrink_toward",
]

_EPS = 1e-9


def logit(p: float) -> float:
    p = min(max(float(p), _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def expit(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def log_linear_pool(
    distributions: Sequence[Mapping[str, float]],
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """Weighted geometric pooling of several distributions over the same
    outcomes, renormalised to sum to one."""
    if not distributions:
        raise ValueError("nothing to pool")
    outcomes = list(distributions[0])
    for d in distributions[1:]:
        if set(d) != set(outcomes):
            raise ValueError("all distributions must cover the same outcomes")
    if weights is None:
        weights = [1.0] * len(distributions)
    if len(weights) != len(distributions):
        raise ValueError("weights and distributions differ in length")
    total_w = float(sum(weights))
    if total_w <= 0:
        raise ValueError("weights must sum to a positive number")

    pooled: dict[str, float] = {}
    for o in outcomes:
        acc = sum(
            w * math.log(max(float(d[o]), _EPS)) for d, w in zip(distributions, weights)
        )
        pooled[o] = math.exp(acc / total_w)
    norm = sum(pooled.values())
    return {o: p / norm for o, p in pooled.items()}


def blend(
    model: Mapping[str, float],
    market: Mapping[str, float],
    model_weight: float,
) -> dict[str, float]:
    """Blend a model distribution toward the market's.

    ``model_weight`` of 0 returns the market untouched, 1 returns the model
    untouched. Anything above ~0.4 is a strong claim; be sure you have the
    out-of-sample record to justify it.
    """
    w = float(model_weight)
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"model_weight must be in [0, 1], got {w}")
    return log_linear_pool([dict(model), dict(market)], [w, 1.0 - w])


def weight_from_evidence(
    n_graded: int,
    calibration_score: float | None = None,
    market_confidence: float = 1.0,
    max_weight: float = 0.45,
    half_sample: int = 400,
) -> float:
    """Derive a defensible model weight instead of picking one by feel.

    ``n_graded``
        Out-of-sample predictions the model has been scored on. The weight
        follows ``n / (n + half_sample)``, so 400 graded bets buys half of
        ``max_weight`` and it takes thousands to approach the cap.
    ``calibration_score``
        Skill score against the market on the same sample: ``1 - BS_model /
        BS_market``, as returned by
        :func:`claudebet.calibration.skill_score`. Zero or negative means the
        model added nothing to the market and the weight collapses to zero.
    ``market_confidence``
        From :meth:`claudebet.market.Consensus.confidence`. A thin or
        disagreeing market is a weaker prior, so the model gets more room.
    """
    if n_graded < 0:
        raise ValueError("n_graded cannot be negative")
    sample = n_graded / (n_graded + float(half_sample))

    if calibration_score is None:
        skill = 0.35  # unproven: allow a token weight, no more
    elif calibration_score <= 0:
        return 0.0
    else:
        skill = min(1.0, calibration_score / 0.05)  # 5% skill saturates

    room = 1.0 + 0.6 * (1.0 - max(0.0, min(1.0, market_confidence)))
    return max(0.0, min(max_weight, max_weight * sample * skill * room))


def shrink_toward(
    p: float, anchor: float, strength: float, in_logit: bool = True
) -> float:
    """Pull an estimate toward an anchor by ``strength`` in [0, 1].

    Shrinking in log-odds (the default) keeps the move proportional at the
    tails, so a 2% estimate shrunk toward 5% does not collapse to zero.
    """
    s = max(0.0, min(1.0, float(strength)))
    if in_logit:
        return expit((1.0 - s) * logit(p) + s * logit(anchor))
    return (1.0 - s) * float(p) + s * float(anchor)
