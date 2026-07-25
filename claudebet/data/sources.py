"""Getting prices in.

Three routes, in increasing order of effort:

* :func:`load_json` -- a small hand-written or scripted file. Good enough to
  analyse a single market you are staring at right now.
* :func:`load_csv` -- one row per quote. Good for a slate you paste out of a
  spreadsheet.
* :class:`OddsApiClient` -- live prices from the-odds-api.com, which is the
  cheapest commercial feed that covers enough books to build a real consensus.
  Note that it does *not* carry Pinnacle in all regions, and a consensus with
  no sharp book in it is a consensus of followers.

The JSON format is deliberately trivial::

    {
      "key": "NFL 2026-01-04 KC@BUF moneyline",
      "outcomes": ["KC", "BUF"],
      "prices": {
        "pinnacle":   {"KC": 2.05, "BUF": 1.87},
        "draftkings": {"KC": 2.15, "BUF": 1.80}
      }
    }

Odds may be decimal, American with an explicit sign (``"+115"``), or fractional
(``"5/2"``) -- see :func:`claudebet.odds.parse_odds`.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..market import Market, Quote

__all__ = [
    "load_json",
    "load_csv",
    "from_the_odds_api",
    "OddsApiClient",
    "write_template",
]

_MARKET_NAMES = {
    "h2h": "moneyline",
    "spreads": "spread",
    "totals": "total",
    "outrights": "outright",
}


def _parse_time(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def load_json(path: str | Path) -> list[Market]:
    """Load one market or a list of markets from a JSON file."""
    data = json.loads(Path(path).read_text())
    blobs = data if isinstance(data, list) else [data]
    markets: list[Market] = []
    for blob in blobs:
        if "bookmakers" in blob:  # the-odds-api event passed through directly
            markets.extend(from_the_odds_api([blob]))
            continue
        prices = blob["prices"]
        outcomes = blob.get("outcomes")
        market = Market.from_book_prices(
            key=blob.get("key", "market"),
            prices=prices,
            outcomes=outcomes,
            asof=_parse_time(blob.get("asof")) or datetime.now(timezone.utc),
        )
        limits = blob.get("limits") or {}
        if limits:
            market.quotes = [
                Quote(
                    book=q.book,
                    outcome=q.outcome,
                    odds=q.odds,
                    limit=limits.get(q.book),
                    timestamp=q.timestamp,
                    points=q.points,
                )
                for q in market.quotes
            ]
        market.meta = blob.get("meta", {})
        markets.append(market)
    return markets


def load_csv(path: str | Path, market_column: str = "market") -> list[Market]:
    """Load quotes from a CSV with one row per (book, outcome) price.

    Required columns: ``book``, ``outcome``, ``odds``. Optional: ``market``
    (defaults to a single unnamed market), ``limit``, ``timestamp``, ``points``.
    """
    rows = list(csv.DictReader(Path(path).open(newline="")))
    if not rows:
        raise ValueError(f"{path} has no rows")
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row.get(market_column) or "market", []).append(row)

    markets: list[Market] = []
    for key, group in grouped.items():
        outcomes: list[str] = []
        quotes: list[Quote] = []
        for row in group:
            outcome = row["outcome"].strip()
            if outcome not in outcomes:
                outcomes.append(outcome)
            quotes.append(
                Quote(
                    book=row["book"].strip(),
                    outcome=outcome,
                    odds=row["odds"],
                    limit=float(row["limit"]) if row.get("limit") else None,
                    timestamp=_parse_time(row.get("timestamp")),
                    points=float(row["points"]) if row.get("points") else None,
                )
            )
        markets.append(
            Market(
                key=key,
                outcomes=tuple(outcomes),
                quotes=quotes,
                asof=datetime.now(timezone.utc),
            )
        )
    return markets


def from_the_odds_api(
    events: Iterable[dict], market_keys: Sequence[str] | None = None
) -> list[Market]:
    """Convert the-odds-api ``/odds`` payloads into markets.

    Spread and total markets are split by line, because a -3.5 and a -2.5 are
    different bets and pooling them would be nonsense. Books offering a
    different number are simply absent from that market's consensus, which is
    the correct treatment -- comparing across lines requires a model, not an
    average.
    """
    markets: list[Market] = []
    for event in events:
        home = event.get("home_team", "home")
        away = event.get("away_team", "away")
        start = event.get("commence_time", "")
        label = f"{away} @ {home} {start}".strip()

        # (market key, line) -> {book: {outcome: (odds, last_update)}}
        buckets: dict[tuple[str, float | None], dict[str, dict[str, tuple[float, Any]]]] = {}
        for bookmaker in event.get("bookmakers", []):
            book = bookmaker.get("key") or bookmaker.get("title") or "unknown"
            for market in bookmaker.get("markets", []):
                mkey = market.get("key", "h2h")
                if market_keys and mkey not in market_keys:
                    continue
                for outcome in market.get("outcomes", []):
                    point = outcome.get("point")
                    # Group a spread by the favourite's line so both sides land
                    # in the same bucket.
                    line = abs(float(point)) if point is not None else None
                    name = outcome["name"]
                    if point is not None:
                        name = f"{name} {float(point):+g}"
                    bucket = buckets.setdefault((mkey, line), {}).setdefault(book, {})
                    bucket[name] = (
                        float(outcome["price"]),
                        market.get("last_update") or bookmaker.get("last_update"),
                    )

        for (mkey, line), by_book in buckets.items():
            pretty = _MARKET_NAMES.get(mkey, mkey)
            key = f"{label} | {pretty}" + (f" {line:g}" if line is not None else "")
            outcomes: list[str] = []
            quotes: list[Quote] = []
            for book, prices in by_book.items():
                for name, (price, updated) in prices.items():
                    if name not in outcomes:
                        outcomes.append(name)
                    quotes.append(
                        Quote(
                            book=book,
                            outcome=name,
                            odds=price,
                            timestamp=_parse_time(updated),
                            points=line,
                        )
                    )
            markets.append(
                Market(
                    key=key,
                    outcomes=tuple(outcomes),
                    quotes=quotes,
                    asof=datetime.now(timezone.utc),
                    meta={"home": home, "away": away, "commence_time": start},
                )
            )
    return markets


class OddsApiClient:
    """Thin client for the-odds-api.com.

    Requires ``requests`` (``pip install claudebet[http]``) and an API key,
    read from the ``ODDS_API_KEY`` environment variable if not passed.
    Every call costs quota, so fetch a whole sport at once rather than
    per-game, and cache what you get.
    """

    BASE = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str | None = None, timeout: float = 15.0) -> None:
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "no API key: pass api_key= or set the ODDS_API_KEY environment variable"
            )
        self.timeout = timeout
        self.last_quota: dict[str, str] = {}

    def _get(self, path: str, params: dict) -> Any:
        try:
            import requests  # noqa: PLC0415 -- optional dependency
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "OddsApiClient needs requests: pip install 'claudebet[http]'"
            ) from exc
        params = {**params, "apiKey": self.api_key}
        response = requests.get(f"{self.BASE}{path}", params=params, timeout=self.timeout)
        response.raise_for_status()
        self.last_quota = {
            "remaining": response.headers.get("x-requests-remaining", "?"),
            "used": response.headers.get("x-requests-used", "?"),
        }
        return response.json()

    def sports(self) -> list[dict]:
        return self._get("/sports", {})

    def odds(
        self,
        sport: str,
        regions: str = "us,eu",
        markets: str = "h2h",
        odds_format: str = "decimal",
    ) -> list[Market]:
        payload = self._get(
            f"/sports/{sport}/odds",
            {"regions": regions, "markets": markets, "oddsFormat": odds_format},
        )
        return from_the_odds_api(payload, market_keys=markets.split(","))


_TEMPLATE = {
    "key": "NFL 2026-01-04 KC@BUF moneyline",
    "outcomes": ["KC", "BUF"],
    "prices": {
        "pinnacle": {"KC": 2.05, "BUF": 1.87},
        "circa": {"KC": 2.06, "BUF": 1.85},
        "draftkings": {"KC": 2.15, "BUF": 1.78},
        "fanduel": {"KC": 2.10, "BUF": 1.80},
    },
    "limits": {"pinnacle": 25000, "circa": 15000, "draftkings": 2500, "fanduel": 2000},
}


def write_template(path: str | Path) -> Path:
    """Drop a filled-in example file to edit."""
    target = Path(path)
    target.write_text(json.dumps(_TEMPLATE, indent=2) + "\n")
    return target
