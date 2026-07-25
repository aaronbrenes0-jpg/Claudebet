"""Downloading real results so you do not have to type them in.

Source is football-data.co.uk, which publishes free CSVs going back decades.
No signup, no API key, no rate limit worth worrying about. It comes in two
shapes and the difference matters a great deal:

**Main leagues** -- the big European divisions -- carry per-match shots, shots
on target, corners, fouls and cards. Everything this package models.

**Extra leagues** -- MLS, Argentina, Brazil, Mexico, Japan and others -- carry
goals and closing odds only. You can still price match results, goal totals and
both-teams-to-score, but corner and card markets are simply not available,
because the underlying numbers do not exist in the feed. :func:`fetch` says so
rather than quietly writing a file that cannot answer the question you bought
it for.

Some competitions are not covered at all. Peru's Liga 1 is one of them, which
matters if that is what your book leads with -- see :data:`NOT_COVERED`.
"""

from __future__ import annotations

import csv
import io
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "LEAGUES",
    "NOT_COVERED",
    "League",
    "fetch",
    "write_results_csv",
    "normalise_season",
    "parse_main_csv",
    "parse_extra_csv",
]

_MAIN_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
_EXTRA_URL = "https://www.football-data.co.uk/new/{code}.csv"


@dataclass(frozen=True)
class League:
    code: str
    name: str
    detailed: bool
    """True when the feed carries corners, shots and cards as well as goals."""


LEAGUES: dict[str, League] = {
    league.code: league
    for league in [
        # Full statistics: goals, shots, shots on target, corners, fouls, cards.
        League("E0", "England — Premier League", True),
        League("E1", "England — Championship", True),
        League("E2", "England — League One", True),
        League("E3", "England — League Two", True),
        League("EC", "England — National League", True),
        League("SC0", "Scotland — Premiership", True),
        League("SC1", "Scotland — Championship", True),
        League("D1", "Germany — Bundesliga", True),
        League("D2", "Germany — 2. Bundesliga", True),
        League("I1", "Italy — Serie A", True),
        League("I2", "Italy — Serie B", True),
        League("SP1", "Spain — La Liga", True),
        League("SP2", "Spain — La Liga 2", True),
        League("F1", "France — Ligue 1", True),
        League("F2", "France — Ligue 2", True),
        League("N1", "Netherlands — Eredivisie", True),
        League("B1", "Belgium — Pro League", True),
        League("P1", "Portugal — Primeira Liga", True),
        League("T1", "Turkey — Super Lig", True),
        League("G1", "Greece — Super League", True),
        # Goals only -- no corners, shots or cards in the feed.
        League("USA", "USA — MLS", False),
        League("ARG", "Argentina — Primera", False),
        League("BRA", "Brazil — Serie A", False),
        League("MEX", "Mexico — Liga MX", False),
        League("JPN", "Japan — J1 League", False),
        League("CHN", "China — Super League", False),
        League("DNK", "Denmark — Superliga", False),
        League("NOR", "Norway — Eliteserien", False),
        League("SWE", "Sweden — Allsvenskan", False),
        League("FIN", "Finland — Veikkausliiga", False),
        League("IRL", "Ireland — Premier Division", False),
        League("POL", "Poland — Ekstraklasa", False),
        League("ROU", "Romania — Liga I", False),
        League("RUS", "Russia — Premier League", False),
        League("AUT", "Austria — Bundesliga", False),
        League("SWZ", "Switzerland — Super League", False),
    ]
}

NOT_COVERED = (
    "Peru, Colombia, Chile, Ecuador and most other South American leagues "
    "outside Argentina and Brazil are not published by this source. For those "
    "you need a paid feed (api-football.com has them on a small free tier) or "
    "to enter results by hand."
)

# football-data.co.uk column -> the name this package uses. Only paired
# statistics are worth importing; anything without both sides is dropped.
_MAIN_STATS = {
    "goals": ("FTHG", "FTAG"),
    "shots": ("HS", "AS"),
    "shots_on_target": ("HST", "AST"),
    "corners": ("HC", "AC"),
    "fouls": ("HF", "AF"),
    "yellows": ("HY", "AY"),
    "reds": ("HR", "AR"),
}


def normalise_season(season: str) -> str:
    """Accept 2425, 2024-25, 24/25 or 2024 and return the feed's own code."""
    text = str(season).strip().replace("/", "-").replace(" ", "")
    if text.isdigit() and len(text) == 4 and not text.startswith("20"):
        return text  # already 2425
    if "-" in text:
        start, _, end = text.partition("-")
        return f"{int(start) % 100:02d}{int(end) % 100:02d}"
    if text.isdigit() and len(text) == 4:  # a single year like 2024
        year = int(text) % 100
        return f"{year:02d}{(year + 1) % 100:02d}"
    if text.isdigit() and len(text) == 2:
        year = int(text)
        return f"{year:02d}{(year + 1) % 100:02d}"
    raise ValueError(f"cannot read season {season!r}; try 2024-25 or 2425")


def _iso_date(value: str) -> str | None:
    text = (value or "").strip()
    if not text:
        return None
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _number(value: str | None) -> int | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_main_csv(text: str) -> list[dict]:
    """Parse a full-statistics league file into our row format."""
    rows = list(csv.DictReader(io.StringIO(text)))
    out: list[dict] = []
    for row in rows:
        date = _iso_date(row.get("Date", ""))
        home = (row.get("HomeTeam") or "").strip()
        away = (row.get("AwayTeam") or "").strip()
        if not (date and home and away):
            continue
        record: dict = {"date": date, "home": home, "away": away}
        for stat, (hcol, acol) in _MAIN_STATS.items():
            h, a = _number(row.get(hcol)), _number(row.get(acol))
            if h is None or a is None:
                continue
            record[f"home_{stat}"] = h
            record[f"away_{stat}"] = a
        if "home_goals" not in record:
            continue  # a match without a score is not a result
        # Total cards is what card markets are usually priced on; keep the
        # yellow and red splits too, since some books price them separately.
        if "home_yellows" in record and "home_reds" in record:
            record["home_cards"] = record["home_yellows"] + record["home_reds"]
            record["away_cards"] = record["away_yellows"] + record["away_reds"]
        out.append(record)
    return out


def parse_extra_csv(text: str, seasons: Sequence[str] | None = None) -> list[dict]:
    """Parse a goals-only league file, optionally filtered to some seasons.

    These files hold every season in one download, and the season column reads
    either "2024" or "2024/2025" depending on whether the league runs across a
    calendar boundary, so both spellings are matched.
    """
    rows = list(csv.DictReader(io.StringIO(text)))
    wanted = set()
    for season in seasons or ():
        code = normalise_season(season)
        start = 2000 + int(code[:2])
        wanted.add(str(start))
        wanted.add(f"{start}/{start + 1}")

    out: list[dict] = []
    for row in rows:
        if wanted and (row.get("Season") or "").strip() not in wanted:
            continue
        date = _iso_date(row.get("Date", ""))
        home = (row.get("Home") or "").strip()
        away = (row.get("Away") or "").strip()
        hg, ag = _number(row.get("HG")), _number(row.get("AG"))
        if not (date and home and away) or hg is None or ag is None:
            continue
        out.append({
            "date": date, "home": home, "away": away,
            "home_goals": hg, "away_goals": ag,
        })
    return out


def _download(url: str, timeout: float = 30.0) -> str:
    request = urllib.request.Request(
        url, headers={"User-Agent": "claudebet/0.1 (+results importer)"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8-sig", errors="replace")


def fetch(
    code: str,
    seasons: Iterable[str] = ("2425",),
    timeout: float = 30.0,
) -> tuple[list[dict], list[str]]:
    """Download one league's results. Returns ``(rows, notes)``.

    ``notes`` carries anything you need to know about what you just got --
    missing seasons, or a feed that has no corners in it.
    """
    key = code.strip().upper()
    if key not in LEAGUES:
        raise ValueError(
            f"unknown league {code!r}. Known codes: {', '.join(sorted(LEAGUES))}"
        )
    league = LEAGUES[key]
    notes: list[str] = []
    rows: list[dict] = []

    if league.detailed:
        for season in seasons:
            season_code = normalise_season(season)
            url = _MAIN_URL.format(season=season_code, code=key)
            try:
                rows.extend(parse_main_csv(_download(url, timeout)))
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    notes.append(f"season {season_code} is not published for {key}")
                    continue
                raise
    else:
        rows = parse_extra_csv(_download(_EXTRA_URL.format(code=key), timeout), list(seasons))
        notes.append(
            f"{league.name} is published with goals only -- no corners, shots "
            "or cards. Match result, goal totals and both-teams-to-score will "
            "work; corner and card markets will not."
        )

    rows.sort(key=lambda r: r["date"])
    return rows, notes


def write_results_csv(rows: Sequence[dict], path: str | Path) -> Path:
    """Write downloaded rows in the layout :func:`load_match_log` expects."""
    if not rows:
        raise ValueError("nothing to write")
    columns: list[str] = ["date", "home", "away"]
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    # Only keep statistics that arrived with both halves present.
    paired = ["date", "home", "away"] + [
        c for c in columns
        if c.startswith("home_") and f"away_{c[5:]}" in columns
    ]
    paired += [f"away_{c[5:]}" for c in paired if c.startswith("home_")]

    target = Path(path)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=paired, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return target
