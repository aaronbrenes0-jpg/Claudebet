"""Scan a whole day of fixtures at one book and rank what is worth betting.

Built for the common real-world case: you have exactly one bookmaker, so there
is no second price to compare against and no arbitrage to find. Everything then
rests on one question -- is your model's probability better than the book's? --
and that is a much harder place to stand than shopping between books. This
module is written to be honest about that rather than to manufacture bets.

**The distinction this module exists to make.** A trend like "over 4.5 shots on
target in five straight games" describes something *likely*. Whether it is a
good *bet* depends entirely on the price. At odds of 1.15 you need it to land
87% of the time just to break even, and the book set 1.15 precisely because it
can see the same five games you can. Likely and profitable are different
things, and the scan reports them in separate columns so you can never confuse
them: ``MODEL`` is how often we think it happens, ``NEEDS`` is how often it has
to happen for the price to be worth taking. Only the gap between them is money.

**Where a single-book edge realistically lives.** Not in the match result of a
big league -- that number is fought over by everyone and the book will be at
least as right as you. It lives in the markets a recreational book prices with
a simple formula and does not move much: corners, cards, shots, team totals,
and smaller competitions. :data:`MARKET_TRUST` encodes that, letting the model
speak louder on side markets than on the main line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from .blend import blend as blend_distributions
from .devig import devig
from .edge import evaluate
from .form import FormModel, MatchLog, MatchProjection
from .odds import decimal_to_prob, parse_odds
from .trends import hit_rate
from .trends import _btts, _over_match, _over_team, _team_unbeaten, _team_wins
from .trends import Condition

__all__ = [
    "Fixture",
    "ScanConfig",
    "Candidate",
    "ScanResult",
    "scan",
    "load_fixtures",
    "write_fixtures_template",
    "MARKET_TRUST",
    "resolve_market",
]

# How much of the blended probability the model is allowed to contribute, by
# market family. A recreational book is sharp on the match result and lazy on
# the peripheral counts, so the model earns more say on the latter. These are
# starting points -- once you have graded a few hundred bets, replace them with
# what claudebet.blend.weight_from_evidence says you have actually earned.
MARKET_TRUST: dict[str, float] = {
    "1X2": 0.20,
    "dc": 0.20,
    "goals": 0.22,
    "btts": 0.28,
    "ah": 0.25,
    "corners": 0.40,
    "cards": 0.40,
    "shots": 0.40,
    "shots_on_target": 0.40,
}
DEFAULT_TRUST = 0.35


@dataclass
class Fixture:
    """One upcoming match and the prices your book is showing on it."""

    home: str
    away: str
    kickoff: str = ""
    competition: str = ""
    neutral: bool = False
    # {market key: {selection: decimal odds}}
    markets: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.markets = {
            key: {sel: parse_odds(price) for sel, price in prices.items()}
            for key, prices in self.markets.items()
        }

    @property
    def label(self) -> str:
        return f"{self.home} v {self.away}"


@dataclass
class ScanConfig:
    """Settings for a single-book scan.

    The defaults are stricter than the multi-book path, deliberately. Without a
    second price there is no independent check on the model, so the required
    edge is larger and the uncertainty charged against every estimate is
    higher.
    """

    window: int = 5
    min_edge: float = 0.03  # 3 points, vs 1.5 when several books can be compared
    kelly_multiplier: float = 0.25
    max_stake_fraction: float = 0.02
    max_slate_exposure: float = 0.10
    confidence_z: float = 1.0
    extra_stderr: float = 0.12  # no cross-book check, so a wider error bar
    trust: Mapping[str, float] | None = None  # override MARKET_TRUST
    trust_scale: float = 1.0  # multiply every trust value, e.g. 0.5 to halve
    min_matches: int = 40  # below this the form model is not worth running
    max_disagreement: float = 0.15
    """Refuse to bet when the model and the book differ by more than this.

    A book prices a match with team news, lineups, weather and the weight of
    everyone else's money. If your model says 51% where it says 30%, the
    explanation is essentially never that you have found a twenty-point edge --
    it is that you are missing something the book knows, or that your history
    is too thin to rate one of these teams. Huge disagreements are the
    signature of a broken input, and treating them as opportunities is the
    fastest way to lose a bankroll. These are surfaced for you to investigate,
    never staked.
    """


@dataclass
class Candidate:
    """One bettable selection, fully diagnosed."""

    match: str
    market: str
    selection: str
    odds: float
    model_prob: float
    book_prob: float  # the book's own opinion, with its margin removed
    blended_prob: float
    needs: float  # break-even probability at this price
    edge: float
    ev: float  # at the blended estimate, so it reconciles with the columns
    ev_after_margin: float  # after the safety haircut; this drives the ranking
    stake_fraction: float
    stake: float
    recent: str  # e.g. "4/5"
    trust: float
    verdict: str
    reason: str

    @property
    def priced_in(self) -> bool:
        """The trend is real but the price already reflects it."""
        return self.verdict != "bet" and self.recent not in ("", "-")


@dataclass
class ScanResult:
    date: str
    book: str
    bets: list[Candidate]
    priced_in: list[Candidate]
    all_candidates: list[Candidate]
    fixtures: int
    markets_priced: int
    near_misses: list[Candidate] = field(default_factory=list)
    flagged: list[Candidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def report(self, bankroll: float = 0.0, max_priced_in: int = 8) -> str:
        out: list[str] = []
        head = f"== {self.book} scan"
        if self.date:
            head += f": {self.date}"
            out.append(head + " ==")
        else:
            out.append(head + " ==")
        out.append(
            f"{self.fixtures} matches, {self.markets_priced} prices checked, "
            f"{len(self.all_candidates)} selections considered"
        )
        for w in self.warnings:
            out.append(f"!  {w}")

        out.append("")
        out.append("TOP BETS -- ranked by value, not by how likely they are to win")
        if not self.bets:
            out.append("  nothing clears the threshold today. That is the normal")
            out.append("  outcome and it is the tool working, not failing.")
        else:
            out.append(
                f"  {'#':<3}{'MATCH':<30}{'BET':<26}{'ODDS':>6}"
                f"{'MODEL':>7}{'NEEDS':>7}{'EDGE':>7}{'EV':>7}{'RECENT':>8}{'STAKE':>9}"
            )
            for i, c in enumerate(self.bets, start=1):
                out.append(
                    f"  {i:<3}{c.match[:29]:<30}{(c.market + ' ' + c.selection)[:25]:<26}"
                    f"{c.odds:>6.2f}{c.blended_prob:>7.1%}{c.needs:>7.1%}"
                    f"{c.edge:>+7.1%}{c.ev:>+7.1%}{c.recent:>8}{c.stake:>9.2f}"
                )
            out.append("")
            out.append(
                "  MODEL is how often we think it happens. NEEDS is how often it"
            )
            out.append(
                "  must happen for the price to be fair. The gap is the only money."
            )

        if self.near_misses:
            out.append("")
            out.append("CLOSE, BUT NOT ENOUGH -- shown so you can see the near misses")
            for c in self.near_misses[:5]:
                out.append(
                    f"  {c.match[:28]:<29} {(c.market + ' ' + c.selection)[:24]:<25} "
                    f"odds {c.odds:>5.2f}  edge {c.edge:+.1%}  -> {c.reason}"
                )

        if self.flagged:
            out.append("")
            out.append("FLAGGED -- the model disagrees with the book far too much")
            for c in self.flagged[:5]:
                out.append(
                    f"  {c.match[:28]:<29} {(c.market + ' ' + c.selection)[:24]:<25} "
                    f"we say {c.model_prob:.0%}, book says {c.book_prob:.0%}"
                )
            out.append("")
            out.append(
                "  A gap that size is almost always your data, not their mistake --"
            )
            out.append(
                "  a team with too few games logged, a missing result, a name typo."
            )
            out.append("  Never staked. Go and look at those teams in your file.")

        if self.priced_in:
            out.append("")
            out.append("ALREADY PRICED IN -- real trends, but the odds are too short")
            for c in self.priced_in[:max_priced_in]:
                loss = c.ev
                out.append(
                    f"  {c.match[:28]:<29} {(c.market + ' ' + c.selection)[:24]:<25} "
                    f"hit {c.recent:<6} odds {c.odds:>5.2f}  "
                    f"needs {c.needs:.0%}, we make it {c.blended_prob:.0%}  "
                    f"-> {loss:+.1%} per unit"
                )
            out.append("")
            out.append(
                "  These are the bets that feel good and lose slowly. The streak is"
            )
            out.append(
                "  real; the book already knows. Betting them costs you the gap."
            )

        if self.bets and bankroll:
            total = sum(c.stake for c in self.bets)
            out.append("")
            out.append(
                f"total staked {total:.2f} of {bankroll:.2f} "
                f"({total / bankroll:.1%} of bankroll)"
            )
        return "\n".join(out)


# -- market key grammar --------------------------------------------------


def resolve_market(projection: MatchProjection, key: str) -> dict[str, float] | None:
    """Model probabilities for a market key, or ``None`` if unavailable.

    The grammar matches how the odds file is written::

        1X2                  home / draw / away
        dc                   1X / 12 / X2
        btts                 yes / no
        ah_-0.5              home / away        (handicap on the home team)
        corners_9.5          over / under       (match total)
        shots_home_4.5       over / under       (one team's total)
    """
    key = key.strip()
    lowered = key.lower()
    try:
        if lowered in ("1x2", "match", "result"):
            return projection.match_odds()
        if lowered in ("dc", "double_chance"):
            return projection.double_chance()
        if lowered == "btts":
            return projection.btts()
        if lowered.startswith("ah_"):
            market = projection.asian_handicap(float(key[3:]))
            return {"home": market["home"], "away": market["away"]}

        parts = key.split("_")
        if len(parts) < 2:
            return None
        line = float(parts[-1])
        if len(parts) >= 3 and parts[-2].lower() in ("home", "away"):
            stat = "_".join(parts[:-2])
            side = parts[-2].lower()
            if stat not in projection.rates:
                return None
            dist = projection.team_distribution(stat, side)
            market = dist.total_market(line)
        else:
            stat = "_".join(parts[:-1])
            if stat not in projection.rates:
                return None
            market = projection.totals(stat, [line])[line]
        return {"over": market["over"], "under": market["under"]}
    except (ValueError, KeyError):
        return None


def _condition_for(key: str, selection: str) -> Condition | None:
    """A historical test matching the market, so we can report a hit rate."""
    lowered = key.strip().lower()
    if lowered in ("1x2", "match", "result") and selection == "home":
        return Condition("wins", _team_wins)
    if lowered in ("dc", "double_chance"):
        return Condition("avoids defeat", _team_unbeaten)
    if lowered == "btts" and selection == "yes":
        return Condition("both teams score", _btts)

    parts = key.split("_")
    if len(parts) < 2:
        return None
    try:
        line = float(parts[-1])
    except ValueError:
        return None
    if len(parts) >= 3 and parts[-2].lower() in ("home", "away"):
        stat = "_".join(parts[:-2])
        test = _over_team(stat, line)
    else:
        stat = "_".join(parts[:-1])
        test = _over_match(stat, line)
    if selection == "under":
        base = test

        def inverted(record, is_home):
            value = base(record, is_home)
            return None if value is None else not value

        return Condition(f"under {line:g} {stat}", inverted)
    if selection != "over":
        return None
    return Condition(f"over {line:g} {stat}", test)


def _recent_record(
    log: MatchLog, fixture: Fixture, key: str, selection: str, window: int
) -> str:
    """How often this selection would have landed in the teams' recent games."""
    condition = _condition_for(key, selection)
    if condition is None:
        return "-"
    parts = key.split("_")
    team_specific = len(parts) >= 3 and parts[-2].lower() in ("home", "away")
    if team_specific:
        team = fixture.home if parts[-2].lower() == "home" else fixture.away
        hits, played, _ = hit_rate(log, team, condition, window)
        return f"{hits}/{played}" if played else "-"
    if key.strip().lower() in ("1x2", "match", "result", "dc"):
        hits, played, _ = hit_rate(log, fixture.home, condition, window)
        return f"{hits}/{played}" if played else "-"
    # Match-level market: pool both teams' recent games.
    hits = played = 0
    for team in (fixture.home, fixture.away):
        h, p, _ = hit_rate(log, team, condition, window)
        hits += h
        played += p
    return f"{hits}/{played}" if played else "-"


def _trust_for(key: str, config: ScanConfig) -> float:
    table = dict(MARKET_TRUST)
    if config.trust:
        table.update(config.trust)
    lowered = key.strip().lower()
    if lowered in table:
        return min(1.0, table[lowered] * config.trust_scale)
    if lowered.startswith("ah_"):
        return min(1.0, table.get("ah", DEFAULT_TRUST) * config.trust_scale)
    parts = lowered.split("_")
    for cut in range(len(parts) - 1, 0, -1):
        stat = "_".join(parts[:cut])
        if stat in table:
            return min(1.0, table[stat] * config.trust_scale)
    return min(1.0, DEFAULT_TRUST * config.trust_scale)


# -- the scan ------------------------------------------------------------


def scan(
    log: MatchLog,
    fixtures: Sequence[Fixture],
    config: ScanConfig | None = None,
    bankroll: float = 0.0,
    book: str = "book",
    date: str = "",
) -> ScanResult:
    """Price every market on every fixture and rank by value."""
    cfg = config or ScanConfig()
    warnings: list[str] = []

    if len(log) < cfg.min_matches:
        warnings.append(
            f"only {len(log)} historical matches; the model needs a few hundred "
            "before its probabilities mean much. Treat today as practice."
        )

    model = FormModel(window=cfg.window).fit(log)
    known = set(log.teams())

    candidates: list[Candidate] = []
    priced = 0
    missing_teams: set[str] = set()

    for fixture in fixtures:
        if fixture.home not in known or fixture.away not in known:
            missing_teams.add(
                fixture.home if fixture.home not in known else fixture.away
            )
            continue
        projection = model.project(fixture.home, fixture.away, neutral=fixture.neutral)

        for key, prices in fixture.markets.items():
            model_probs = resolve_market(projection, key)
            if model_probs is None:
                warnings.append(
                    f"{fixture.label}: no model for market {key!r} -- check the name "
                    "or add the statistic to your results file"
                )
                continue
            usable = {s: o for s, o in prices.items() if s in model_probs}
            if not usable:
                continue
            priced += len(usable)

            # The book's own view, with its margin stripped out. With one book
            # this is the only opinion available to disagree with.
            if len(usable) >= 2:
                order = list(usable)
                fair = devig([usable[s] for s in order])
                book_probs = dict(zip(order, fair.probs))
            else:
                only = next(iter(usable))
                book_probs = {only: decimal_to_prob(usable[only])}

            trust = _trust_for(key, cfg)
            complete = set(model_probs) == set(usable)
            for selection, odds in usable.items():
                book_prob = book_probs.get(selection, decimal_to_prob(odds))
                if complete and len(usable) >= 2:
                    blended = blend_distributions(
                        {s: model_probs[s] for s in usable}, book_probs, trust
                    )[selection]
                else:
                    # Partial market: blend the two numbers directly rather
                    # than pretending we have a full distribution.
                    blended = (
                        model_probs[selection] ** trust * book_prob ** (1 - trust)
                    )

                result = evaluate(
                    blended,
                    odds,
                    outcome=selection,
                    bankroll=bankroll,
                    prob_stderr=cfg.extra_stderr,
                    kelly_multiplier=cfg.kelly_multiplier,
                    confidence_z=cfg.confidence_z,
                    min_edge=cfg.min_edge,
                    max_stake_fraction=cfg.max_stake_fraction,
                )

                verdict, reason = result.verdict, result.reason
                stake_fraction, stake = result.stake_fraction, result.stake
                # How far apart we are allowed to be before this reads as a
                # broken input rather than an edge. Scaled by trust: a book is
                # very hard to beat by ten points on the match result, but a
                # recreational corners line genuinely can be that far out.
                tolerance = cfg.max_disagreement * (0.5 + 2.0 * trust)
                gap = abs(model_probs[selection] - book_prob)
                if gap > tolerance:
                    verdict = "check"
                    reason = (
                        f"model and book differ by {gap:.0%}, past the {tolerance:.0%} "
                        f"we allow on this market -- more likely a gap in your results "
                        "file than an edge"
                    )
                    stake_fraction = stake = 0.0

                candidates.append(
                    Candidate(
                        match=fixture.label,
                        market=key,
                        selection=selection,
                        odds=odds,
                        model_prob=model_probs[selection],
                        book_prob=book_prob,
                        blended_prob=blended,
                        needs=result.breakeven_prob,
                        edge=result.edge_pp,
                        ev=blended * odds - 1.0,
                        ev_after_margin=result.ev_per_unit,
                        stake_fraction=stake_fraction,
                        stake=stake,
                        recent=_recent_record(
                            log, fixture, key, selection, cfg.window
                        ),
                        trust=trust,
                        verdict=verdict,
                        reason=reason,
                    )
                )

    if missing_teams:
        warnings.append(
            "no history for "
            + ", ".join(sorted(missing_teams))
            + " -- those matches were skipped"
        )

    candidates.sort(key=lambda c: -c.ev_after_margin)
    bets = [c for c in candidates if c.verdict == "bet"]

    # Cap total exposure across the day.
    total = sum(c.stake_fraction for c in bets)
    if total > cfg.max_slate_exposure and total > 0:
        scale = cfg.max_slate_exposure / total
        for c in bets:
            c.stake_fraction *= scale
            c.stake *= scale
        warnings.append(
            f"total exposure scaled down to {cfg.max_slate_exposure:.0%} of bankroll"
        )

    # Trends that landed often but are not worth backing at the price.
    priced_in = [
        c
        for c in candidates
        if c.verdict != "bet" and _hit_fraction(c.recent) >= 0.8 and c.ev < 0
    ]
    priced_in.sort(key=lambda c: (-_hit_fraction(c.recent), c.ev))

    near_misses = [c for c in candidates if c.verdict == "pass" and c.ev > 0]
    flagged = [c for c in candidates if c.verdict == "check"]

    return ScanResult(
        date=date,
        book=book,
        bets=bets,
        priced_in=priced_in,
        near_misses=near_misses,
        flagged=flagged,
        all_candidates=candidates,
        fixtures=len(fixtures),
        markets_priced=priced,
        warnings=warnings,
    )


def _hit_fraction(recent: str) -> float:
    if "/" not in recent:
        return 0.0
    hits, _, played = recent.partition("/")
    try:
        return float(hits) / float(played) if float(played) else 0.0
    except ValueError:
        return 0.0


# -- input files ---------------------------------------------------------


def load_fixtures(path: str | Path) -> tuple[list[Fixture], str, str]:
    """Read a day's fixtures and prices. Returns (fixtures, date, book)."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        data = {"matches": data}
    fixtures = [
        Fixture(
            home=m["home"],
            away=m["away"],
            kickoff=m.get("kickoff", ""),
            competition=m.get("competition", ""),
            neutral=bool(m.get("neutral", False)),
            markets=m.get("markets", {}),
        )
        for m in data.get("matches", [])
    ]
    return fixtures, data.get("date", ""), data.get("book", "book")


_FIXTURES_TEMPLATE = {
    "date": "2026-07-26",
    "book": "doradobet",
    "_help": (
        "One entry per match you are considering. Under 'markets', use the key "
        "names shown here and type the decimal odds straight off the app. "
        "Include only the markets you care about -- more is better but nothing "
        "is required. Keys: 1X2, dc, btts, ah_-0.5, goals_2.5, corners_9.5, "
        "cards_4.5, shots_on_target_home_4.5 (any stat in your results file, "
        "optionally with home/away, then the line)."
    ),
    "matches": [
        {
            "home": "Inter Miami",
            "away": "Chicago Fire",
            "kickoff": "20:30",
            "markets": {
                "1X2": {"home": 1.75, "draw": 3.90, "away": 4.20},
                "goals_2.5": {"over": 1.85, "under": 1.95},
                "btts": {"yes": 1.72, "no": 2.05},
                "corners_9.5": {"over": 1.90, "under": 1.90},
                "shots_on_target_home_4.5": {"over": 1.18, "under": 4.50},
            },
        },
        {
            "home": "Orlando City",
            "away": "Atlanta United",
            "kickoff": "22:00",
            "markets": {
                "1X2": {"home": 2.05, "draw": 3.50, "away": 3.60},
                "corners_10.5": {"over": 2.00, "under": 1.80},
            },
        },
    ],
}


def write_fixtures_template(path: str | Path) -> Path:
    target = Path(path)
    target.write_text(json.dumps(_FIXTURES_TEMPLATE, indent=2) + "\n")
    return target
