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
from typing import TYPE_CHECKING, Any, Iterable, Sequence

if TYPE_CHECKING:  # avoids a circular import at runtime
    from ..form import MatchLog

from ..market import Market, Quote

__all__ = [
    "load_json",
    "load_csv",
    "load_match_log",
    "from_the_odds_api",
    "OddsApiClient",
    "write_template",
    "write_match_log_template",
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


def load_match_log(path: str | Path) -> "MatchLog":
    """Load a results file into a :class:`claudebet.form.MatchLog`.

    One row per completed match. Required columns: ``date``, ``home``, ``away``.
    Every other column named ``home_<stat>`` with a matching ``away_<stat>``
    becomes a tracked statistic, so you decide what to feed the model just by
    what you put in the file::

        date,home,away,home_goals,away_goals,home_corners,away_corners,home_cards,away_cards
        2026-02-01,Alianza,Cristal,2,1,7,4,2,3

    ``goals`` is the only name treated specially -- results and every
    goal-derived market come from it. Add ``home_shots``/``away_shots``,
    ``home_offsides``/``away_offsides``, or anything else and it gets modelled
    and priced the same way. Unpaired columns are ignored rather than guessed
    at. Optional columns: ``competition``, ``neutral``.
    """
    from ..form import MatchLog, MatchRecord

    rows = list(csv.DictReader(Path(path).open(newline="")))
    if not rows:
        raise ValueError(f"{path} has no rows")

    fields = list(rows[0])
    stats = [
        name[len("home_") :]
        for name in fields
        if name.startswith("home_") and f"away_{name[len('home_'):]}" in fields
    ]
    if not stats:
        raise ValueError(
            f"{path} has no paired home_*/away_* columns; expected at least "
            "home_goals and away_goals"
        )

    def number(value: str | None) -> float | None:
        if value is None or str(value).strip() == "":
            return None
        try:
            return float(value)
        except ValueError:
            return None

    records = []
    for row in rows:
        home_stats, away_stats = {}, {}
        for stat in stats:
            h, a = number(row.get(f"home_{stat}")), number(row.get(f"away_{stat}"))
            if h is None or a is None:
                continue  # a missing half makes the pair unusable for this match
            home_stats[stat] = h
            away_stats[stat] = a
        if not home_stats:
            continue
        neutral = str(row.get("neutral", "")).strip().lower() in ("1", "true", "yes", "y")
        records.append(
            MatchRecord(
                home=(row.get("home") or "").strip(),
                away=(row.get("away") or "").strip(),
                home_stats=home_stats,
                away_stats=away_stats,
                when=_parse_time(row.get("date") or row.get("when")),
                competition=(row.get("competition") or "").strip(),
                neutral=neutral,
            )
        )
    if not records:
        raise ValueError(f"{path} produced no usable matches")
    return MatchLog(records)


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


EXAMPLE_DATA_MARKER = "EXAMPLE-DATA-DO-NOT-BET"

_MATCH_LOG_HEADER = (
    "date,home,away,home_goals,away_goals,home_corners,away_corners,"
    "home_cards,away_cards,home_shots,away_shots,competition\n"
)
_MATCH_LOG_ROWS = [
    "2025-08-03,Melgar,Boys,4,3,12,12,2,4,12,13,EXAMPLE-DATA-DO-NOT-BET",
    "2025-08-03,Melgar,Cienciano,1,1,4,4,6,1,12,10,EXAMPLE-DATA-DO-NOT-BET",
    "2025-08-03,Alianza Lima,Universitario,2,1,7,7,3,1,16,4,EXAMPLE-DATA-DO-NOT-BET",
    "2025-08-07,Boys,Cienciano,1,3,3,5,3,1,6,6,EXAMPLE-DATA-DO-NOT-BET",
    "2025-08-07,Cienciano,Melgar,0,1,10,7,2,1,12,5,EXAMPLE-DATA-DO-NOT-BET",
    "2025-08-07,Alianza Lima,Cienciano,3,0,5,0,5,0,14,8,EXAMPLE-DATA-DO-NOT-BET",
    "2025-08-11,Alianza Lima,Cienciano,2,1,7,1,3,0,16,9,EXAMPLE-DATA-DO-NOT-BET",
    "2025-08-11,Universitario,Alianza Lima,1,1,12,2,0,4,9,15,EXAMPLE-DATA-DO-NOT-BET",
    "2025-08-11,Alianza Lima,Boys,4,0,9,5,3,1,16,6,EXAMPLE-DATA-DO-NOT-BET",
    "2025-08-15,Alianza Lima,Boys,1,1,8,9,4,0,13,7,EXAMPLE-DATA-DO-NOT-BET",
    "2025-08-15,Universitario,Melgar,1,2,12,3,7,2,15,12,EXAMPLE-DATA-DO-NOT-BET",
    "2025-08-15,Boys,Universitario,2,0,14,7,0,2,12,13,EXAMPLE-DATA-DO-NOT-BET",
    "2025-08-19,Cienciano,Universitario,2,0,5,2,1,1,10,11,EXAMPLE-DATA-DO-NOT-BET",
    "2025-08-19,Melgar,Boys,3,0,4,7,3,1,7,12,EXAMPLE-DATA-DO-NOT-BET",
    "2025-08-19,Cienciano,Universitario,0,2,8,2,1,2,15,9,EXAMPLE-DATA-DO-NOT-BET",
    "2025-08-23,Cienciano,Boys,2,1,6,6,4,0,12,9,EXAMPLE-DATA-DO-NOT-BET",
    "2025-08-23,Sporting Cristal,Boys,5,1,3,3,3,0,13,12,EXAMPLE-DATA-DO-NOT-BET",
    "2025-08-23,Universitario,Boys,4,1,5,7,3,2,16,8,EXAMPLE-DATA-DO-NOT-BET",
    "2025-08-27,Melgar,Universitario,1,1,7,6,3,1,11,11,EXAMPLE-DATA-DO-NOT-BET",
    "2025-08-27,Cienciano,Sporting Cristal,2,1,2,3,1,1,8,14,EXAMPLE-DATA-DO-NOT-BET",
    "2025-08-27,Universitario,Alianza Lima,0,3,3,1,1,3,16,7,EXAMPLE-DATA-DO-NOT-BET",
    "2025-08-31,Alianza Lima,Melgar,0,1,6,6,1,2,16,9,EXAMPLE-DATA-DO-NOT-BET",
    "2025-08-31,Boys,Sporting Cristal,1,0,2,5,2,2,7,11,EXAMPLE-DATA-DO-NOT-BET",
    "2025-08-31,Melgar,Sporting Cristal,0,0,4,2,3,3,9,10,EXAMPLE-DATA-DO-NOT-BET",
    "2025-09-04,Cienciano,Melgar,1,1,7,7,2,2,8,9,EXAMPLE-DATA-DO-NOT-BET",
    "2025-09-04,Alianza Lima,Sporting Cristal,1,1,6,4,2,5,16,16,EXAMPLE-DATA-DO-NOT-BET",
    "2025-09-04,Boys,Alianza Lima,0,2,8,4,3,2,4,4,EXAMPLE-DATA-DO-NOT-BET",
    "2025-09-08,Melgar,Universitario,3,2,5,2,0,3,9,5,EXAMPLE-DATA-DO-NOT-BET",
    "2025-09-08,Universitario,Sporting Cristal,3,1,5,3,5,2,12,11,EXAMPLE-DATA-DO-NOT-BET",
    "2025-09-08,Sporting Cristal,Cienciano,2,0,6,3,5,3,16,13,EXAMPLE-DATA-DO-NOT-BET",
    "2025-09-12,Universitario,Melgar,2,0,7,5,1,3,12,9,EXAMPLE-DATA-DO-NOT-BET",
    "2025-09-12,Universitario,Cienciano,5,2,7,4,0,1,5,4,EXAMPLE-DATA-DO-NOT-BET",
    "2025-09-12,Boys,Melgar,1,0,3,7,0,3,12,9,EXAMPLE-DATA-DO-NOT-BET",
    "2025-09-16,Melgar,Alianza Lima,3,2,8,7,4,2,13,15,EXAMPLE-DATA-DO-NOT-BET",
    "2025-09-16,Boys,Alianza Lima,2,3,1,4,1,2,7,15,EXAMPLE-DATA-DO-NOT-BET",
    "2025-09-16,Universitario,Cienciano,3,0,10,5,0,4,16,9,EXAMPLE-DATA-DO-NOT-BET",
    "2025-09-20,Cienciano,Boys,1,1,5,8,2,1,2,5,EXAMPLE-DATA-DO-NOT-BET",
    "2025-09-20,Melgar,Cienciano,1,1,5,4,2,1,12,8,EXAMPLE-DATA-DO-NOT-BET",
    "2025-09-20,Boys,Cienciano,2,1,7,2,3,1,14,5,EXAMPLE-DATA-DO-NOT-BET",
    "2025-09-24,Sporting Cristal,Alianza Lima,1,1,7,5,4,0,16,16,EXAMPLE-DATA-DO-NOT-BET",
    "2025-09-24,Cienciano,Sporting Cristal,3,1,9,6,3,3,16,11,EXAMPLE-DATA-DO-NOT-BET",
    "2025-09-24,Alianza Lima,Sporting Cristal,0,0,7,6,1,5,15,14,EXAMPLE-DATA-DO-NOT-BET",
    "2025-09-28,Melgar,Alianza Lima,0,1,10,3,0,3,16,13,EXAMPLE-DATA-DO-NOT-BET",
    "2025-09-28,Boys,Melgar,0,3,3,4,3,0,6,10,EXAMPLE-DATA-DO-NOT-BET",
    "2025-09-28,Sporting Cristal,Melgar,1,1,0,7,3,3,15,9,EXAMPLE-DATA-DO-NOT-BET",
    "2025-10-02,Alianza Lima,Universitario,1,1,8,4,1,0,16,5,EXAMPLE-DATA-DO-NOT-BET",
    "2025-10-02,Melgar,Sporting Cristal,3,0,7,6,6,0,15,15,EXAMPLE-DATA-DO-NOT-BET",
    "2025-10-02,Cienciano,Alianza Lima,1,1,5,3,3,2,9,14,EXAMPLE-DATA-DO-NOT-BET",
    "2025-10-06,Sporting Cristal,Universitario,1,0,4,3,3,0,16,15,EXAMPLE-DATA-DO-NOT-BET",
    "2025-10-06,Boys,Sporting Cristal,0,2,11,2,2,0,5,11,EXAMPLE-DATA-DO-NOT-BET",
    "2025-10-06,Sporting Cristal,Alianza Lima,2,1,5,2,3,2,16,15,EXAMPLE-DATA-DO-NOT-BET",
    "2025-10-10,Sporting Cristal,Boys,0,1,6,8,1,1,11,14,EXAMPLE-DATA-DO-NOT-BET",
    "2025-10-10,Sporting Cristal,Cienciano,2,0,4,3,0,7,13,8,EXAMPLE-DATA-DO-NOT-BET",
    "2025-10-10,Universitario,Boys,4,0,5,2,2,1,7,13,EXAMPLE-DATA-DO-NOT-BET",
    "2025-10-14,Sporting Cristal,Universitario,2,0,5,6,1,1,16,15,EXAMPLE-DATA-DO-NOT-BET",
    "2025-10-14,Universitario,Sporting Cristal,2,4,3,4,1,1,10,9,EXAMPLE-DATA-DO-NOT-BET",
    "2025-10-14,Boys,Universitario,3,0,7,7,3,1,13,14,EXAMPLE-DATA-DO-NOT-BET",
    "2025-10-18,Alianza Lima,Melgar,0,1,7,3,1,5,12,7,EXAMPLE-DATA-DO-NOT-BET",
    "2025-10-18,Cienciano,Alianza Lima,1,1,8,3,0,2,8,16,EXAMPLE-DATA-DO-NOT-BET",
    "2025-10-18,Sporting Cristal,Melgar,3,3,3,3,0,2,14,10,EXAMPLE-DATA-DO-NOT-BET",
]


def write_match_log_template(path: str | Path) -> Path:
    """Drop an example results file showing the expected column layout."""
    target = Path(path)
    target.write_text(_MATCH_LOG_HEADER + "\n".join(_MATCH_LOG_ROWS) + "\n")
    return target
