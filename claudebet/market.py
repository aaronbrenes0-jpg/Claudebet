"""Market structure: consensus fair prices, line shopping, and movement.

The central object is a :class:`Market` -- every book's price on every outcome
of one betting market at one moment. From it we build a *consensus fair
probability*: each book's prices are devigged separately, then pooled in
log-odds space with weights for how sharp the book is, how large its limits
are, and how fresh the quote is.

One rule this module enforces that most naive implementations get wrong: when
you evaluate a price at book X, the consensus you compare it against must
exclude book X. Otherwise the outlier you are trying to detect is sitting
inside your own benchmark, dragging it toward the number you are testing and
shrinking the edge you would have measured. See ``exclude`` on
:func:`consensus` and note that :func:`evaluate_market` applies it for you.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping, Sequence

from .devig import devig
from .odds import decimal_to_prob, parse_odds, prob_to_decimal

__all__ = [
    "Quote",
    "Market",
    "Consensus",
    "consensus",
    "best_prices",
    "arbitrage",
    "line_shopping_gain",
    "movement",
    "SHARPNESS",
    "DEFAULT_SHARPNESS",
]

# How much weight a book's opinion carries, on a 0-1 scale, before limit and
# recency adjustments. These are priors from how each book operates -- who
# takes size, who moves first, who copies -- not fitted constants. Override
# them with your own if your data says otherwise; the pooling is what matters.
SHARPNESS: dict[str, float] = {
    "pinnacle": 1.00,
    "circa": 0.95,
    "bookmaker": 0.90,
    "betcris": 0.85,
    "betfair": 0.85,  # exchange, after commission
    "smarkets": 0.80,
    "matchbook": 0.78,
    "betonline": 0.65,
    "heritage": 0.60,
    "bovada": 0.45,
    "fanduel": 0.45,
    "draftkings": 0.45,
    "betmgm": 0.40,
    "caesars": 0.40,
    "pointsbet": 0.35,
    "espnbet": 0.30,
    "fanatics": 0.30,
    "betrivers": 0.30,
    "unibet": 0.30,
    "hardrock": 0.25,
}
DEFAULT_SHARPNESS = 0.35

# A quote this old is worth nothing; weight decays exponentially toward it.
_STALENESS_HALFLIFE = timedelta(minutes=20)


def _norm_book(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("_", "")


@dataclass(frozen=True)
class Quote:
    """One book's price on one outcome."""

    book: str
    outcome: str
    odds: float  # decimal
    limit: float | None = None  # max stake accepted, in currency units
    timestamp: datetime | None = None
    points: float | None = None  # the handicap/total this price refers to

    def __post_init__(self) -> None:
        object.__setattr__(self, "odds", parse_odds(self.odds))

    @property
    def implied(self) -> float:
        return decimal_to_prob(self.odds)


@dataclass
class Market:
    """Every quote on one market at one point in time."""

    key: str
    outcomes: tuple[str, ...]
    quotes: list[Quote] = field(default_factory=list)
    asof: datetime | None = None
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_book_prices(
        cls,
        key: str,
        prices: Mapping[str, Mapping[str, float]],
        outcomes: Sequence[str] | None = None,
        asof: datetime | None = None,
    ) -> "Market":
        """Build from ``{book: {outcome: odds}}`` -- the usual shape of an API
        response or a hand-typed screen scrape."""
        if outcomes is None:
            seen: list[str] = []
            for book_prices in prices.values():
                for outcome in book_prices:
                    if outcome not in seen:
                        seen.append(outcome)
            outcomes = tuple(seen)
        quotes = [
            Quote(book=book, outcome=outcome, odds=odds)
            for book, book_prices in prices.items()
            for outcome, odds in book_prices.items()
        ]
        return cls(key=key, outcomes=tuple(outcomes), quotes=quotes, asof=asof)

    def books(self) -> list[str]:
        out: list[str] = []
        for q in self.quotes:
            if q.book not in out:
                out.append(q.book)
        return out

    def by_book(self, complete_only: bool = True) -> dict[str, dict[str, float]]:
        """``{book: {outcome: decimal odds}}``.

        With ``complete_only`` (the default) a book appears only if it quotes
        every outcome, because a partial market cannot be devigged.
        """
        table: dict[str, dict[str, float]] = {}
        for q in self.quotes:
            table.setdefault(q.book, {})[q.outcome] = q.odds
        if complete_only:
            table = {
                b: p for b, p in table.items() if all(o in p for o in self.outcomes)
            }
        return table

    def quote(self, book: str, outcome: str) -> Quote | None:
        for q in self.quotes:
            if q.book == book and q.outcome == outcome:
                return q
        return None


@dataclass(frozen=True)
class Consensus:
    """Pooled fair probabilities across books."""

    probs: dict[str, float]
    weights: dict[str, float]
    per_book: dict[str, dict[str, float]]
    dispersion: dict[str, float]
    mean_hold: float
    n_books: int
    method: str
    excluded: tuple[str, ...] = ()

    @property
    def fair_odds(self) -> dict[str, float]:
        return {o: prob_to_decimal(p) for o, p in self.probs.items()}

    @property
    def effective_books(self) -> float:
        """Kish effective sample size of the weights. Three books that are all
        copying one origin behave like fewer than three independent opinions;
        this at least catches the case where one book carries all the weight."""
        total = sum(self.weights.values())
        if total <= 0:
            return 0.0
        sq = sum(w * w for w in self.weights.values())
        return (total * total) / sq if sq > 0 else 0.0

    def confidence(self) -> float:
        """0-1 summary of how much to trust this consensus.

        Falls with few effective books and with disagreement between them.
        Feed it into the model/market blend weight rather than reading it as a
        probability of anything.
        """
        breadth = 1.0 - math.exp(-self.effective_books / 2.5)
        spread = max(self.dispersion.values()) if self.dispersion else 0.0
        agreement = math.exp(-spread / 0.15)
        return max(0.0, min(1.0, breadth * agreement))


def _weight_for(
    book: str,
    prices: Mapping[str, float],
    quotes: Sequence[Quote],
    asof: datetime | None,
    sharpness: Mapping[str, float],
) -> float:
    base = sharpness.get(_norm_book(book), DEFAULT_SHARPNESS)

    # Limits: a book taking 50k is telling you far more than one taking 200.
    limits = [q.limit for q in quotes if q.limit is not None]
    if limits:
        limit_factor = min(1.5, 0.6 + 0.4 * math.log10(max(10.0, min(limits)) / 100.0))
        base *= max(0.5, limit_factor)

    # Freshness.
    stamps = [q.timestamp for q in quotes if q.timestamp is not None]
    if stamps and asof is not None:
        oldest = min(stamps)
        if oldest.tzinfo is None:
            oldest = oldest.replace(tzinfo=timezone.utc)
        ref = asof if asof.tzinfo else asof.replace(tzinfo=timezone.utc)
        age = (ref - oldest).total_seconds()
        if age > 0:
            base *= 0.5 ** (age / _STALENESS_HALFLIFE.total_seconds())

    # A book posting a huge margin is not offering an opinion, it is offering a
    # tax. Down-weight it relative to a tight one.
    total = sum(decimal_to_prob(o) for o in prices.values())
    excess = max(0.0, total - 1.0)
    base *= math.exp(-6.0 * excess)

    return max(0.0, base)


def consensus(
    market: Market,
    method: str = "shin",
    sharpness: Mapping[str, float] | None = None,
    exclude: str | Iterable[str] | None = None,
    weights: Mapping[str, float] | None = None,
) -> Consensus:
    """Pool every book's devigged view into one fair probability per outcome.

    Pooling happens in log space (a weighted geometric mean, renormalised),
    which is the right average for probabilities: it is invariant to how you
    label the outcomes and cannot be dragged past the extremes by one book.
    """
    sharp = dict(SHARPNESS)
    if sharpness:
        sharp.update({_norm_book(k): v for k, v in sharpness.items()})

    if exclude is None:
        excluded: tuple[str, ...] = ()
    elif isinstance(exclude, str):
        excluded = (exclude,)
    else:
        excluded = tuple(exclude)
    excluded_norm = {_norm_book(b) for b in excluded}

    table = market.by_book(complete_only=True)
    table = {b: p for b, p in table.items() if _norm_book(b) not in excluded_norm}
    if not table:
        raise ValueError(
            f"no book quotes every outcome of {market.key!r} "
            f"after excluding {excluded or '()'}"
        )

    per_book: dict[str, dict[str, float]] = {}
    book_weights: dict[str, float] = {}
    holds: list[float] = []

    for book, prices in table.items():
        ordered = [prices[o] for o in market.outcomes]
        result = devig(ordered, method=method)
        per_book[book] = dict(zip(market.outcomes, result.probs))
        holds.append(result.hold)
        if weights is not None:
            w = float(weights.get(book, 0.0))
        else:
            book_quotes = [q for q in market.quotes if q.book == book]
            w = _weight_for(book, prices, book_quotes, market.asof, sharp)
        book_weights[book] = w

    total_weight = sum(book_weights.values())
    if total_weight <= 0:
        book_weights = {b: 1.0 for b in per_book}
        total_weight = float(len(book_weights))

    # Weighted geometric mean in probability space == arithmetic mean of logs.
    pooled: dict[str, float] = {}
    for outcome in market.outcomes:
        acc = 0.0
        for book, probs in per_book.items():
            acc += book_weights[book] * math.log(max(probs[outcome], 1e-12))
        pooled[outcome] = math.exp(acc / total_weight)
    norm = sum(pooled.values())
    pooled = {o: p / norm for o, p in pooled.items()}

    # Dispersion: weighted stdev of each outcome's log-odds across books. This
    # is the market's own disagreement, and it is the honest error bar on the
    # consensus.
    dispersion: dict[str, float] = {}
    for outcome in market.outcomes:
        logits = []
        ws = []
        for book, probs in per_book.items():
            p = min(max(probs[outcome], 1e-9), 1 - 1e-9)
            logits.append(math.log(p / (1 - p)))
            ws.append(book_weights[book])
        wsum = sum(ws)
        if wsum <= 0 or len(logits) < 2:
            dispersion[outcome] = 0.0
            continue
        mean = sum(w * x for w, x in zip(ws, logits)) / wsum
        var = sum(w * (x - mean) ** 2 for w, x in zip(ws, logits)) / wsum
        dispersion[outcome] = math.sqrt(var)

    return Consensus(
        probs=pooled,
        weights=book_weights,
        per_book=per_book,
        dispersion=dispersion,
        mean_hold=sum(holds) / len(holds),
        n_books=len(per_book),
        method=method,
        excluded=excluded,
    )


def best_prices(market: Market) -> dict[str, tuple[float, str]]:
    """Best available decimal price per outcome, with the book offering it."""
    best: dict[str, tuple[float, str]] = {}
    for q in market.quotes:
        current = best.get(q.outcome)
        if current is None or q.odds > current[0]:
            best[q.outcome] = (q.odds, q.book)
    return best


def arbitrage(market: Market) -> dict | None:
    """Detect a risk-free book across the best available prices.

    Returns ``None`` unless the best prices on every outcome sum to under one
    booked probability. Stakes are normalised to a total outlay of 1.
    """
    best = best_prices(market)
    if not all(o in best for o in market.outcomes):
        return None
    total = sum(decimal_to_prob(best[o][0]) for o in market.outcomes)
    if total >= 1.0:
        return None
    stakes = {o: decimal_to_prob(best[o][0]) / total for o in market.outcomes}
    return {
        "booked_total": total,
        "roi": (1.0 / total) - 1.0,
        "stakes": stakes,
        "legs": {o: {"odds": best[o][0], "book": best[o][1]} for o in market.outcomes},
    }


def line_shopping_gain(market: Market) -> dict[str, float]:
    """Edge in probability points gained by taking the best price rather than
    the average one. This is free money and it compounds; most bettors leave
    more here than their model ever generates."""
    best = best_prices(market)
    table = market.by_book(complete_only=False)
    gain: dict[str, float] = {}
    for outcome in market.outcomes:
        offers = [p[outcome] for p in table.values() if outcome in p]
        if not offers or outcome not in best:
            continue
        avg_implied = sum(decimal_to_prob(o) for o in offers) / len(offers)
        gain[outcome] = avg_implied - decimal_to_prob(best[outcome][0])
    return gain


def movement(
    snapshots: Sequence[Market],
    method: str = "shin",
    steam_threshold: float = 0.02,
    steam_window: timedelta = timedelta(minutes=10),
) -> dict:
    """Track the fair price through time across an ordered list of snapshots.

    Reports total drift per outcome, the closing consensus, and any *steam*:
    a move of at least ``steam_threshold`` in fair probability inside
    ``steam_window``. Steam is a sharp-money footprint -- it says the number
    moved fast and in one direction, which is a reason to take the price now
    or not at all, depending on which way you are leaning.
    """
    if not snapshots:
        raise ValueError("no snapshots supplied")
    ordered = sorted(
        snapshots, key=lambda m: m.asof or datetime.min.replace(tzinfo=timezone.utc)
    )
    series: list[dict] = []
    for snap in ordered:
        try:
            c = consensus(snap, method=method)
        except ValueError:
            continue
        series.append({"asof": snap.asof, "probs": c.probs, "n_books": c.n_books})
    if not series:
        raise ValueError("no snapshot had a complete book")

    first, last = series[0], series[-1]
    outcomes = list(last["probs"])
    drift = {o: last["probs"][o] - first["probs"].get(o, float("nan")) for o in outcomes}

    steam: list[dict] = []
    for i, point in enumerate(series):
        if point["asof"] is None:
            continue
        for j in range(i + 1, len(series)):
            other = series[j]
            if other["asof"] is None:
                continue
            if other["asof"] - point["asof"] > steam_window:
                break
            for o in outcomes:
                if o not in point["probs"] or o not in other["probs"]:
                    continue
                delta = other["probs"][o] - point["probs"][o]
                if abs(delta) >= steam_threshold:
                    steam.append(
                        {
                            "outcome": o,
                            "from": point["asof"],
                            "to": other["asof"],
                            "delta": delta,
                            "direction": "shortening" if delta > 0 else "drifting",
                        }
                    )
    return {
        "series": series,
        "open": first["probs"],
        "close": last["probs"],
        "drift": drift,
        "steam": steam,
        "n_snapshots": len(series),
    }
