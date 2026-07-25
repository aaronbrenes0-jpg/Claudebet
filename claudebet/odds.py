"""Odds representation and conversion.

Everything in the rest of the package speaks *decimal* odds internally: the
total return per unit staked, including the stake. American, fractional and
implied-probability inputs are converted at the edge.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Iterable, Sequence

__all__ = [
    "american_to_decimal",
    "decimal_to_american",
    "fractional_to_decimal",
    "decimal_to_fractional",
    "prob_to_decimal",
    "decimal_to_prob",
    "prob_to_american",
    "american_to_prob",
    "parse_odds",
    "profit",
    "payout",
    "overround",
    "hold",
    "breakeven_prob",
]

# Below this, a decimal price is not a price at all (you cannot return less
# than the stake) and almost always means the caller passed a probability.
MIN_DECIMAL = 1.0000001


def _check_decimal(d: float) -> float:
    d = float(d)
    if not math.isfinite(d) or d <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {d!r}")
    return d


def _check_prob(p: float) -> float:
    p = float(p)
    if not math.isfinite(p) or not (0.0 < p < 1.0):
        raise ValueError(f"probability must be strictly between 0 and 1, got {p!r}")
    return p


def american_to_decimal(american: float) -> float:
    """+150 -> 2.5, -150 -> 1.6667."""
    a = float(american)
    if a == 0 or not math.isfinite(a):
        raise ValueError(f"invalid american odds: {american!r}")
    if abs(a) < 100:
        raise ValueError(
            f"american odds of {american!r} are inside the +/-100 dead zone; "
            "did you mean decimal odds?"
        )
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / -a)


def decimal_to_american(decimal: float) -> float:
    d = _check_decimal(decimal)
    return (d - 1.0) * 100.0 if d >= 2.0 else -100.0 / (d - 1.0)


def fractional_to_decimal(frac: str | Fraction) -> float:
    """'5/2' -> 3.5."""
    if isinstance(frac, Fraction):
        return 1.0 + float(frac)
    text = str(frac).strip()
    if "/" not in text:
        raise ValueError(f"not fractional odds: {frac!r}")
    num, _, den = text.partition("/")
    try:
        n, d = float(num), float(den)
    except ValueError as exc:
        raise ValueError(f"not fractional odds: {frac!r}") from exc
    if d <= 0 or n <= 0:
        raise ValueError(f"fractional odds must be positive: {frac!r}")
    return 1.0 + n / d


def decimal_to_fractional(decimal: float, max_denominator: int = 100) -> str:
    d = _check_decimal(decimal)
    f = Fraction(d - 1.0).limit_denominator(max_denominator)
    return f"{f.numerator}/{f.denominator}"


def prob_to_decimal(p: float) -> float:
    return 1.0 / _check_prob(p)


def decimal_to_prob(decimal: float) -> float:
    """Implied probability *including* the vig. This is a booked probability,
    not a fair one -- run it through :mod:`claudebet.devig` first."""
    return 1.0 / _check_decimal(decimal)


def prob_to_american(p: float) -> float:
    return decimal_to_american(prob_to_decimal(p))


def american_to_prob(american: float) -> float:
    return decimal_to_prob(american_to_decimal(american))


_AMERICAN_RE = re.compile(r"^[+-]\d+(\.\d+)?$")


def parse_odds(value: float | int | str) -> float:
    """Best-effort parse of any common odds notation into decimal odds.

    Rules, in order:
      * ``"5/2"`` or ``"evens"``/``"ev"``      -> fractional
      * ``"+150"`` / ``"-150"`` (explicit sign) -> American
      * ``"85%"`` or ``0 < x < 1``              -> probability
      * ``|x| >= 100``                          -> American
      * otherwise                               -> already decimal

    The explicit sign is what disambiguates American from decimal, so keep it
    on when your source has it.
    """
    if isinstance(value, str):
        text = value.strip().lower().replace(" ", "")
        if not text:
            raise ValueError("empty odds string")
        if text in ("evens", "even", "ev", "evs"):
            return 2.0
        if "/" in text:
            return fractional_to_decimal(text)
        if text.endswith("%"):
            return prob_to_decimal(float(text[:-1]) / 100.0)
        if _AMERICAN_RE.match(text):
            return american_to_decimal(float(text))
        value = float(text)

    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"invalid odds: {value!r}")
    if 0.0 < x < 1.0:
        return prob_to_decimal(x)
    if abs(x) >= 100.0:
        return american_to_decimal(x)
    if x <= 1.0:
        raise ValueError(
            f"ambiguous or invalid odds {value!r}: decimal odds must exceed 1.0"
        )
    return x


def profit(decimal: float, stake: float = 1.0) -> float:
    """Net win on a successful bet (excludes the returned stake)."""
    return (_check_decimal(decimal) - 1.0) * stake


def payout(decimal: float, stake: float = 1.0) -> float:
    """Gross return on a successful bet (includes the returned stake)."""
    return _check_decimal(decimal) * stake


def overround(odds: Iterable[float]) -> float:
    """Sum of booked probabilities. 1.05 means a 5% overround."""
    total = sum(decimal_to_prob(o) for o in odds)
    if total <= 0:
        raise ValueError("no odds supplied")
    return total


def hold(odds: Iterable[float]) -> float:
    """The book's theoretical margin as a fraction of handle.

    A two-way -110/-110 market holds ~4.5%, not the 4.76% overround: hold is
    ``(sum - 1) / sum``, which is the share of *money bet* the book keeps if
    action is balanced. That is the number to compare across markets with
    different numbers of outcomes.
    """
    total = overround(odds)
    return (total - 1.0) / total


def breakeven_prob(decimal: float) -> float:
    """Win rate needed to break even at this price. Same as implied prob."""
    return decimal_to_prob(decimal)


def normalize(probs: Sequence[float]) -> list[float]:
    total = float(sum(probs))
    if total <= 0:
        raise ValueError("probabilities must sum to a positive number")
    return [float(p) / total for p in probs]
